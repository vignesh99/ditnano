import numpy as np
import torch
import torch.distributed as dist
import pickle
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from models.models import DiT_models
from models.edm import EDMPrecond
from diffusion import create_diffusion
from utils.losses import loss_dict
from utils.diffutils import requires_grad
from utils.datasets import PairedDataset, PairedCondDataset

def get_data(args, rank, device, world_size=4) :
    '''
    Get training data loader (for class conditional and unconditional)
    '''
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    if args.dataset == 'imagenet' :
        latent_size = args.image_size // 8
    else :
        #for CIFAR-10 (image size is 32x32)
        latent_size = args.image_size
    #one-hot vectors for EDM labels
    eye_matrix = torch.eye(args.num_classes, device=device)
        
    if args.cond:
        dataset = PairedCondDataset(args.data_path, world_size=world_size, rank=rank)
    else:
        dataset = PairedDataset(args.data_path, world_size=world_size, rank=rank)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    return loader, dataset, sampler, latent_size

#----------------------------------------------------------------#
def get_models(args, device, rank, latent_size=32, multione=None, trainsetup='GET') :
    '''
    Get student and teacher models for performing distillation
    '''
    #Student models
    model = DiT_models[args.model](input_size=latent_size,
                                   multione=multione,
                                   trainsetup=trainsetup,
                                   num_classes=args.num_classes,    #args based inputs
                                   depth=args.depth,
                                   num_heads=args.num_heads,
                                   hidden_size=args.hidden_size,
                                   patch_size=args.patch_size,)
    ema =  copy.deepcopy(model).to(device)  # EMA of the model for use after training
    requires_grad(ema, False)
    #Teacher models
    if "edm" in args.teacher :
        #EDM hyperparams directly taken from DMD implementation on GitHub
        #https://github.com/devrimcavusoglu/dmd
        teacher = EDMPrecond(img_resolution=32,
                             img_channels=3,
                             label_dim=10,
                             resample_filter=[1, 1],
                             embedding_type="positional",
                             augment_dim=9,
                             dropout=0.13,
                             model_type="SongUNet",
                             encoder_type="standard",
                             channel_mult_noise=1,
                             model_channels=128,
                             channel_mult=(2, 2, 2),
                             )
        #load this for EDM "../pretrained/edm-cifar10-32x32-cond-vp.pkl"
        with open_url(args.teacher_load) as f :
            teacherckpt = pickle.load(f)['ema']
        teacher.load_state_dict(teacherckpt.state_dict(), strict=True)
        del teacher.model.map_augment    #remove params to avoid gradient error
        teacher.model.map_augment = None
        #fake teacher - DMD implementation
        faketeacher = copy.deepcopy(teacher)
        faketeacher = DDP(faketeacher.to(device), device_ids=[rank])
        faketeacher.requires_grad_(True)    #can be removed 

    elif "DiT" in args.teacher :
        teacher = DiT_models[args.teacher](input_size=latent_size,
                                           num_classes=args.num_classes
                                           )
        teacherckpt = torch.load(args.teacher_load, map_location=torch.device('cpu'))
        teacher.load_state_dict(teacherckpt['ema'])
        faketeacher = None

    else :
        teacher, faketeacher = None, None

    #distributed training
    model = DDP(model.to(device), device_ids=[rank])
    #set models to train and eval
    model.train()
    if teacher is not None :
        teacher = teacher.to(device)
        teacher.requires_grad_(False)
        teacher.eval() 
    if faketeacher is not None :
        faketeacher.train()
    ema.eval()    # EMA model should always be in eval mode

    return model, ema, teacher, faketeacher

#----------------------------------------------------------------#
def get_trainsetup(args, model, faketeacher, device) :
    # Setup optimizer for DiT-Nano 
    opt = torch.optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    fakeopt = None
    if "edm" in args.teacher :
        #optimizer for fake teacher if using DMD
        fakeopt = torch.optim.AdamW(faketeacher.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
    
    #get 1-step loss function or multi-step diffusion process
    if args.num_sampling_steps == 1 or args.trainsetup == 'EDMMS':
        loss_fn = loss_dict[args.loss]().to(device)
        diffusion = None
    else :
        loss_fn = None
        diffusion = create_diffusion(timestep_respacing="", # default: linear noise schedule
                                     diffusion_steps=args.num_sampling_steps,
                                     dataset=args.dataset,
                                     staticdist=args.staticdist)  
        #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    return opt, fakeopt, loss_fn, diffusion

#----------------------------------------------------------------#
def trainbatch(args, x, z, c, model, teacher, faketeacher, opt, fakeopt, loss_fn, diffusion, mi1steps=3, vae=None, device='cuda') :
    '''
    Do one epoch of training based on chosen setup
    '''
    #setup 1: DiT paper setup (multi-step sampling in latent space)
    if args.trainsetup == 'DiT' :
        loss, model, opt = trainDiTsetup(args, x, c,
                                         model, opt, diffusion, vae,
                                         device)
    #setup 2: Pixel-space DDPM setup (multi-step sampling)
    elif args.trainsetup == 'PixelDDPM' :
        loss, model, opt = trainPixelDDPMsetup(args, x, z, c,
                                               model, opt, diffusion,
                                               device)
    #setup 3: DMD setup (one-step sampling with special DM loss)
    elif args.trainsetup == 'DMD' :
        loss, model, faketeacher, opt, fakeopt = trainDMDsetup(args, x, z, c,
                                                               model, teacher, faketeacher,
                                                               opt, fakeopt, loss_fn,
                                                               device)
    #setup 4: Layer-wise distillation (one-step sampling)
    elif args.trainsetup == 'Layer' :
        loss, model, opt = trainLayersetup(args, x, z, c,
                                           model, teacher, opt, loss_fn,
                                           device)
    #setup 5: Multi-step sampling in a single step
    elif args.trainsetup == 'MultiOne' :
        loss, model, opt = trainMultiOnesetup(args, x, z, c,
                                              model, opt, loss_fn, mi1steps,
                                              device)
    #setup 6: Multi-step distillation setup from EDM
    elif args.trainsetup == 'EDMMS' :
        loss, model, opt = trainEDMMSsetup(args, x, z, c,
                                           model, opt, loss_fn,
                                           device)
    #setup 7: Regular one-step distillation setup from GET
    elif args.trainsetup == 'GET' :
        loss, model, opt = trainGETsetup(args, x, z, c,
                                         model, opt, loss_fn,
                                         device)
    #add missing parameters
    if args.trainsetup != 'DMD' :
        faketeacher = None
        fakeopt = None

    return loss, model, opt, faketeacher, fakeopt

#----------------------------------------------------------------#
def trainDiTsetup(args, x, c, model, opt, diffusion, vae, device) :
    '''
    Latent space multi-step diffusion for ImageNet
    '''
    assert args.num_sampling_steps != 1    #multi-step sampling
    assert args.dataset == 'imagenet'
    
    with torch.no_grad():
        # Map input images to latent space + normalize latents:
        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    model_kwargs = dict(y=c)
    
    diff_loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
    loss = diff_loss_dict["loss"].mean()
    opt.zero_grad()    #optimizer
    loss.backward()
    opt.step()
    
    return loss, model, opt

#----------------------------------------------------------------#
def trainPixelDDPMsetup(args, x, z, c, model, opt, diffusion, device) :
    '''
    Pixel space multi-step diffusion for CIFAR-10
    '''
    assert args.num_sampling_steps != 1    #multi-step sampling
    assert args.dataset == 'cifar10'
    
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    #if static dataset is used for multi-step distillation
    if args.staticdist :
        T = torch.LongTensor(
            [diffusion.num_timesteps-1]).repeat(x.shape[0]).to(device)
        x_T = z
    else :
        T = None
        x_T = None
    #compute diffusion denoising loss
    model_kwargs = dict(y=c)
    diff_loss_dict = diffusion.training_losses(model, x, t,
                                               model_kwargs,
                                               x_T=z, T=T)
    loss = diff_loss_dict["loss"].mean()
    opt.zero_grad()    #optimizer
    loss.backward()
    opt.step()

    return loss, model, opt

#----------------------------------------------------------------#
def trainDMDsetup(args, x, z, c, model, teacher, faketeacher, opt, fakeopt, loss_fn, device) :
    '''
    Pixel space one-step diffusion distillation using distribution matching loss
    '''
    assert args.num_sampling_steps == 1    #one-step sampling
    assert args.dataset == 'cifar10'
    
    #implement DMD loss
    t = torch.zeros((x.shape[0],), device=device)
    z_dmd = torch.randn_like(x, device=device)
    x_dmd = model(z_dmd, t, c)    #for dmd loss
    x_pred = model(z, t, c)       #for regression loss
    
    faketeacher.requires_grad_(False)    #disable gradient for DM loss
    edmlabels = eye_matrix[c]
    loss = loss_fn(mu_real=teacher,      #Distribution-Matching loss
                   mu_fake=faketeacher,
                   x=x_dmd,
                   x_ref=x_pred,
                   y_ref=x,
                   labels=edmlabels)
    faketeacher.requires_grad_(True)     #renable gradient for denoising loss
    opt.zero_grad()    #optimizer
    loss.backward()
    if args.maxnorm is not None :        #based on DMD paper
        nn.utils.clip_grad_norm_(model.parameters(),
                                 args.maxnorm)
    opt.step()
    
    #implement diffusion denoising loss on fake model
    loss_fn_diff = loss_dict['denoise']().to(device)
    fakeloss = loss_fn_diff(mu_fake=faketeacher,
                            x=x_dmd,
                            labels=edmlabels)
    fakeopt.zero_grad()    #optimizer
    fakeloss.backward()
    if args.maxnorm is not None :
        nn.utils.clip_grad_norm_(faketeacher.parameters(),
                                 args.maxnorm)
    fakeopt.step()

    return loss, model, faketeacher, opt, fakeopt

#----------------------------------------------------------------#
def trainLayersetup(args, x, z, c, model, teacher, opt, loss_fn, device) :
    '''
    Pixel space one-step diffusion layer-wise distillation 
    '''
    assert args.num_sampling_steps == 1    #one-step sampling
    assert args.dataset == 'cifar10'
    
    t = torch.zeros((x.shape[0],), device=device)
    x_pred, attn_stud = model(z, t, c)
    x_teach, attn_teach = teacher(z, t, c)
    loss = loss_fn(x_stud=x_pred,
                   x_true=x,
                   x_teach=x_teach,
                   attn_stud=attn_stud,
                   attn_teach=attn_teach)
    opt.zero_grad()    #optimizer
    loss.backward()
    opt.step()

    return loss, model, opt

#----------------------------------------------------------------#
def trainMultiOnesetup(args, x, z, c, model, opt, loss_fn, mi1steps, device) :
    '''
    Pixel space mapping multi-step diffusion into a single step
    distillation

    forward ODE discretization through EDM setup
    x_t = \sqrt{alpha_t}x_0 + \sqrt{1-alpha_t}z
    z = (x_T - \sqrt{alpha_T}x_0) / \sqrt{1-alpha_T}
    \alpha_t = 1/(1+\sigma_t^2)
    '''
    assert args.num_sampling_steps == 1    #one-step sampling
    assert args.dataset == 'cifar10'
    sigmatime = args.sigmatime

    #obtaining intermediate noisy samples
    sigma_min, sigma_max = 0.002, 80
    rho = 7.0
    revtsteps = torch.arange(1, mi1steps)    #time steps from reverse diffuion perspective
    #sclar input to timestep embedding
    tsteps = (torch.concat((torch.flip(revtsteps, dims=(0,)), torch.tensor([0])))/mi1steps).to(device)
    #since reverse diffusion idx=0 is noise and idx=1 is image. tidx != actual idx.
    #EDM noise setup
    sigmas = (sigma_max**(1/rho) + (revtsteps/(mi1steps)) * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
    sigmas = torch.concat((sigmas, torch.tensor([0]))).to(device)    #add 0 since final image has no noise
    alphas = 1.0 / (1.0 + sigmas**2)    #using SNR definition
    alpha_T = torch.tensor(1.0 / (1.0 + sigma_max**2)).to(device)
    noise = (z - torch.sqrt(alpha_T)*x) / torch.sqrt(1-alpha_T)    #finding ODE noise using final-step noise
    xint = []
        
    #layers to tap into
    timeip = sigmas if sigmatime else tsteps    #timestep embedding input
    for idx in range(mi1steps) :
        #get intermediate output
        if idx < mi1steps-1 :
            xint.extend(torch.sqrt(alphas[idx])*x + torch.sqrt(1-alphas[idx])*noise)
        #timestep embedding tensor input    
        if idx == 0 :
            #since cannot concat in first iteration
            t = (timeip[idx])*torch.ones((x.shape[0],)).to(device)
        else :
            #intermediate time index
            t = torch.cat((t, (timeip[idx])*torch.ones((x.shape[0],)).to(device)), dim=0)
    t = t.to(device)
    
    x_pred, xint_pred = model(z, t, c)
    xint = torch.stack(xint)    #match the shape/type of model output
    loss = loss_fn(x_pred, x, xint_pred, xint)
    
    opt.zero_grad()    #optimizer
    loss.backward()
    opt.step()
    faketeacher = None

    return loss, model, opt

#----------------------------------------------------------------#
def trainEDMMSsetup(args, x, z, c, model, opt, loss_fn, device) :
    '''
    Pixel space one-step diffusion distillation using GET setup
    '''
    assert args.num_sampling_steps != 1    #one-step sampling
    assert args.dataset == 'cifar10'
    
    #code block from EDM
    P_mean, P_std, sigma_data = -1.2, 1.2, 0.5
    rnd_normal = torch.randn([z.shape[0], 1, 1, 1], device=z.device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    z = z * sigma
    
    t = torch.zeros((x.shape[0],), device=device)
    x_pred,_ = model(z+x, t, c, sigma)
    loss = loss_fn(x_pred, x)
    
    opt.zero_grad()    #optimizer
    loss.backward()
    opt.step()
    faketeacher = None

    return loss, model, opt

#----------------------------------------------------------------#
def trainGETsetup(args, x, z, c, model, opt, loss_fn, device) :
    '''
    Pixel space one-step diffusion distillation using GET setup
    '''
    assert args.num_sampling_steps == 1    #one-step sampling
    assert args.dataset == 'cifar10'
    
    t = torch.zeros((x.shape[0],), device=device)
    x_pred,_ = model(z, t, c)
    loss = loss_fn(x_pred, x)
    
    opt.zero_grad()    #optimizer
    loss.backward()
    opt.step()
    faketeacher = None

    return loss, model, opt

#----------------------------------------------------------------#

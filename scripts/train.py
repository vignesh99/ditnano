#general libraries
import sys
import argparse
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
import time
import numpy as np
from glob import glob #?
import torch
import torch.nn as nn
# the first flag below was False when DiT authors tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torchprofile import profile_macs #?
import copy
import pickle
import warnings

#custom libraries
sys.path.append("../")  #load libraries from main folder
from utils.diffutils import (
    create_logger, save_ckpt, 
    update_ema, requires_grad,  
    sample_image, sample_fid, compute_fid_is,
    open_url, get_fixed_generator_sigma,
        )
from utils.trainutils import get_models, get_data, get_trainsetup, trainbatch
from models.models import DiT_models
from utils.losses import loss_dict
from diffusers.models import AutoencoderKL
from models.edm import EDMPrecond

#------------------------------------------------------------------------------------#
def main(args):
    '''
    Model training.
    '''
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP (distributed data parallel)
    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert args.global_batch_size % world_size == 0, f'Batch size must be divisible by world size.'
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f'Starting rank={rank}, seed={seed}, world_size={world_size}.')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #ignore file depreciated warnings
    #warnings.filterwarnings("ignore", category=UserWarning) 

    # Setup an experiment folder
    if rank == 0 :
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f'{args.results_dir}/*'))
        model_string_name = args.model.replace('/', '-')
        experiment_dir = f'{args.results_dir}/{experiment_index:03d}-{model_string_name}-{args.name}'

        checkpoint_dir = f'{experiment_dir}/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        sample_dir = f'{experiment_dir}/samples'
        os.makedirs(sample_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f'Experiment directory created at {experiment_dir}')
    else:
        logger = create_logger()

    #----------------------------------------------------------------#       
    # Setup data
    loader, dataset, sampler, latent_size = get_data(args, rank, device, world_size)
    logger.info(f'Dataset contains {len(dataset):,} images ({dataset.data_dir})')
    
    # Create models
    multione = torch.tensor([1, 3]) if args.trainsetup == 'MultiOne' else None
    model, ema, teacher, faketeacher = get_models(args, device, rank, latent_size,
                                                  multione=multione,
                                                  trainsetup=args.trainsetup)
    
    #get optimizers, loss and diffusion process
    opt, fakeopt, loss_fn, diffusion = get_trainsetup(args, model, faketeacher, device)
    
    # Get params and FLOPS (need to implement)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    dist.barrier()
    
    #----------------------------------------------------------------#       
    # Prepare models for training
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights     
    # Variables for monitoring/logging purposes
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_loss_reg = 0
    loss_list = []
    loss_reg_list = []
    total_steps = args.epochs * (len(dataset) / args.global_batch_size)

    # Resume from the prev checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=torch.device('cpu'))
        model.module.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        train_steps = max(args.resume_iter, 0)
        
        logger.info(f'Resume from {args.resume}..')
    
    start_time = time.time()
    logger.info(f'Training for {args.epochs} epochs...')
    
    #------------------------------------------------------------------------------------#        
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f'Beginning epoch {epoch}...')

        #load batch of data
        for data in loader:
            # Unpack data
            if args.cond:
                z, x, c = data
                z, x, c = z.to(device), x.to(device), c.to(device).max(dim=1)[1]
            else:
                z, x = data
                z, x, c = z.to(device), x.to(device), None

            #Train one batch
            loss, model, opt, faketeacher, fakeopt = trainbatch(args,
                                                                x=x, z=z, c=c,
                                                                model=model,
                                                                teacher=teacher,
                                                                faketeacher=faketeacher,
                                                                opt=opt,
                                                                fakeopt=fakeopt,
                                                                loss_fn=loss_fn,
                                                                diffusion=diffusion,
                                                                mi1steps=3,
                                                                vae=None,
                                                                device=device)

            #----------------------------------------------------------------#                    
            # LR Warmup
            if train_steps < args.warmup_iter:
                curr_lr = args.lr * (train_steps+1) / args.warmup_iter
                opt.param_groups[0]['lr'] = curr_lr
                
            update_ema(ema, model.module, decay=args.ema_decay)
            loss_list.append(loss.item())
            running_loss += loss_list[-1]
            log_steps += 1
            train_steps += 1
            
            #------------------------------------------------------------------------#            
            # Log training progress
            if train_steps % args.log_every == 0:
                # Measure training speed
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(f'(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}')
                
                # Reset monitoring variables
                loss_list = []
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = f'{checkpoint_dir}/{train_steps:07d}.pth'
                    save_ckpt(args, model, ema, opt, checkpoint_path)
                    logger.info(f'Saved checkpoint to {checkpoint_path}')
                dist.barrier()

            # Save the latest checkpoint
            if train_steps % args.save_latest_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = f'{checkpoint_dir}/latest.pth'
                    save_ckpt(args, model, ema, opt, checkpoint_path)
                    logger.info(f'Saved latest checkpoint to {checkpoint_path}')
                dist.barrier()

            # Sample images
            if train_steps % args.sample_every == 0 and train_steps > 0:
                if rank == 0:
                    image_path = f'{sample_dir}/{train_steps}.png'
                    #sample_image(args, ema, device, image_path, cond=args.cond, cfg=args.cfg)
                    logger.info(f'Saved samples to {image_path}')
                dist.barrier()
            
            # Compute FID and IS
            if train_steps % args.eval_every == 0 and train_steps > 0:
                images = sample_fid(args, ema, device, rank, cond=args.cond, cfg=args.cfg, mi1steps=4)#, set_train=True)

                # In case you want to sample from the online model
                # images = sample_fid(args, model.module, device, rank, cond=args.cond, set_grad=True)
                
                # DDP sync
                all_images = [torch.zeros_like(images) for _ in range(world_size)]
                dist.gather(images, all_images if rank == 0 else None, dst=0)
                if rank == 0:
                    FID, IS = compute_fid_is(args, all_images, rank)
                    logger.info(f'FID {FID:0.2f}, IS {IS:0.2f} at iters {train_steps}.')
                
                del images, all_images
                dist.barrier()

            # Check training schedule
            if train_steps > total_steps:
                break

    if rank == 0:
        checkpoint_path = f'{checkpoint_dir}/final.pth'
        save_ckpt(args, model, ema, opt, checkpoint_path)
        logger.info(f'Saved final checkpoint to {checkpoint_path}')
    dist.barrier()
    
    # Finish training
    dist.destroy_process_group()

#------------------------------------------------------------------------------------#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #data and model
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--results-dir', type=str, default='../results-train')
    parser.add_argument('--model', type=str, choices=list(DiT_models.keys()), default='DiT-N/2')
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--cond', action='store_true', help='Run conditional model.')
    parser.add_argument('--num-classes', type=int, default=10) #vs49: 1000 for ImageNet
    parser.add_argument('--num-sampling-steps', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--patch-size', type=int, default=2)
    parser.add_argument('--loss', type=str, choices=['l1', 'l2', 'lpips', 'dists', 'gen', 'lay', 'multione'], default='lpips')
    parser.add_argument('--vae', type=str, choices=['ema', 'mse'], default='ema')
    parser.add_argument('--teacher', type=str, default='None', choices=['edm', 'DiT-S/2', 'None'])
    parser.add_argument('--teacher-load', type=str, default="None")
    #training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-iter', type=int, default=0, help="warmup for the given iterations")
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--global-batch-size', type=int, default=256)
    parser.add_argument('--global-seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--ckpt-every', type=int, default=100000)
    parser.add_argument('--save-latest-every', type=int, default=20000)
    parser.add_argument('--maxnorm', type=float, default=10.0)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--trainsetup', type=str, choices=['DiT', 'PixelDDPM', 'DMD', 'Layer', 'GET', 'MultiOne', 'EDMMS'], default='GET')
    parser.add_argument('--sigmatime', action='store_true', help='Use SNR values as input to timestep embedding')
    #evaluation
    parser.add_argument('--sample-every', type=int, default=1000000)
    parser.add_argument('--eval-every', type=int, default=12500)
    parser.add_argument('--eval-samples', type=int, default=50000)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--prng', action='store_true', help='Run with prng instead of randn')
    parser.add_argument('--cfg', action='store_true', help='Perform classifier-free guidance during inference')
    #miscellaneous
    parser.add_argument('--stat-path', type=str, default='YOUR_STAT_PATH/cifar10.test.npz')
    parser.add_argument('--teacher-path', type=str, default='pretrained/edm-cifar10-32x32-cond-vp.pkl')
    parser.add_argument('--resume', help="restore checkpoint for training")
    parser.add_argument('--resume-iter', type=int, default=-1, help="resume from the given iterations")
    parser.add_argument('--staticdist', action='store_true', help='Latent space distillation loss')
    parser.add_argument('--latent', action='store_true', help='Perform diffusion in latent space')
    
    args = parser.parse_args()
    main(args)

import sys
import argparse
import os
import re #?
import time
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.append("../")  #load libraries from main folder
from utils.diffutils import (
        create_logger, requires_grad,
        sample_image, sample_fid, compute_fid_is
        )
from models.models import DiT_models

def main(args):
    '''
    Model evaluation.
    '''
    # Setup DDP
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f'Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup an experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        resume_dir = re.split('/|\.', args.resume)
        folder_name = f'eval-{resume_dir[-4]}-{resume_dir[-2]}-{args.name}'
        experiment_dir = f'{args.results_dir}/{folder_name}'  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f'Experiment directory created at {experiment_dir}')
    else:
        logger = create_logger()

    # Create model
    model = DiT_models[args.model](input_size=args.image_size,
                                   num_classes=args.num_classes,
                                   depth=args.depth,
                                   num_heads=args.num_heads,
                                   hidden_size=args.hidden_size,
                                   patch_size=args.patch_size,
                                   multione=torch.tensor([3]) if 'MultiOne' in args.trainsetup else None,
                                   trainsetup=args.trainsetup,).to(device)
    ema = DiT_models[args.model](input_size=args.image_size,
                                 num_classes=args.num_classes,
                                 depth=args.depth,
                                 num_heads=args.num_heads,
                                 hidden_size=args.hidden_size,
                                 patch_size=args.patch_size,
                                 multione=torch.tensor([3]) if 'MultiOne' in args.trainsetup else None,
                                 trainsetup=args.trainsetup,).to(device)
    requires_grad(ema, False)
    
    # Setup DDP
    model = DDP(model.to(device), device_ids=[rank])
    logger.info(f'Model Parameters: {sum(p.numel() for p in model.parameters()):,}')

    model.eval()
    ema.eval()
    
    # Resume from the given checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=torch.device('cpu'))
        model.module.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        logger.info(f'Resume from {args.resume}..')

    # Sample images
    '''
    if rank == 0:
        image_path = f'{experiment_dir}/samples.png'
        sample_image(args, ema, device, image_path, cond=args.cond, cfg=args.cfg)
        #sample_image(args, model, device, image_path, cond=args.cond)
        logger.info(f'Saved samples to {image_path}')
    dist.barrier()
    '''

    # Compute FID and IS
    start_time = time.time()
    images = sample_fid(args, ema, device, rank, cond=args.cond, cfg=args.cfg)
    #images = sample_fid(args, model, device, rank, cond=args.cond)
    end_time = time.time()
    logger.info(f'Time for sampling 50k images {end_time-start_time:.2f}s.')

    # DDP sync for FID evaluation
    all_images = [torch.zeros_like(images) for _ in range(dist.get_world_size())]
    dist.gather(images, all_images if rank == 0 else None, dst=0)
    if rank == 0:
        FID, IS = compute_fid_is(args, all_images, rank)
        logger.info(f'FID {FID:0.2f}, IS {IS:0.2f}.')
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='../results-eval')
    parser.add_argument('--name', type=str, default='debug')

    parser.add_argument('--model', type=str, choices=list(DiT_models.keys()), default='DiT-N/2')
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--patch-size', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='cifar10')

    parser.add_argument('--cond', action='store_true', help='Run conditional model.')
    parser.add_argument('--num-classes', type=int, default=10)

    parser.add_argument('--global-seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--eval-samples', type=int, default=50000)
    parser.add_argument('--stat-path', type=str, default='YOUR_STAT_PATH/cifar10.test.npz')
    parser.add_argument('--num-sampling-steps', type=int, default=1)

    parser.add_argument('--resume', help="restore checkpoint for training")
    #my own
    parser.add_argument('--prng', action='store_true', help='Run with prng instead of randn')
    parser.add_argument('--cfg', action='store_true', help='Run with classifier-free guidance')
    parser.add_argument('--trainsetup', type=str, choices=['DiT', 'PixelDDPM', 'DMD', 'Layer', 'GET', 'MultiOne'], default='GET')
    parser.add_argument('--sigmatime', action='store_true', help='Use SNR values as input to timestep embedding')

    args = parser.parse_args()
    main(args)

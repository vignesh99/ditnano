from pylab import *
from collections import OrderedDict
import logging
import torch
import torch.distributed as dist
import re
from PIL import Image
from pytorch_gan_metrics import get_inception_score_and_fid

from diffusion import create_diffusion
from prng.prng_helper import prng

def create_logger(logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    '''
    Step the EMA model towards the current model.
    '''
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    '''
    Set requires_grad flag for all parameters in a model.
    '''
    for p in model.parameters():
        p.requires_grad = flag

def save_ckpt(args, model, ema, opt, checkpoint_path):
    '''
    Save a checkpoint containing the online model, EMA, and optimizer states.
    '''
    checkpoint = {
            'args': args,
            'model': model.module.state_dict(),
            'ema': ema.state_dict(),
            'opt': opt.state_dict(),
            }
    torch.save(checkpoint, checkpoint_path)


def sample_image(args, model, device, image_path, set_train=False, cond=False, cfg=False):
    '''
    sample a batch of images for visualization.
    set set_train to true if you are using the online model for sampling.
    '''
    model.eval()
    
    n_row, n_col = 1,1#5, 2 #vs49: change from 16 to 1 to save memory 
    size = args.image_size

    #z = torch.randn(n_row*n_col, 3, size, size).to(device)
    z = torch.randn(10, 3, size, size).to(device)
    z = [z[3]]*(n_row*n_col)
    z = torch.stack(z).to(device)
    #vs49: need to modify for multi-step
    t = torch.zeros((n_row*n_col,)).to(device)
    #c = torch.randint(0, args.num_classes, (n_row*n_row,)).to(device) if cond else None
    #c = 7*torch.ones((n_row*n_row,)).to(torch.int).to(device) if cond else None  #vs49: to see one class clearly
    c = torch.arange(10, 11).reshape(n_row*n_col).to(torch.int).to(device) if cond else None
    print(c)
    with torch.no_grad():
        if not cfg :
            x,_ = model(z, t, c)
        else :
            z = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            c_null = torch.tensor([args.num_classes]*(n_row*n_col), device=device)
            c = torch.cat([c, c_null], 0)
            x,_ = model.forward_with_cfg(z, t, c, cfg_scale=1.5)
                
    x = x.view(2*n_row, n_col, 3, size, size)
    x = x[0]
    x = (x * 127.5 + 128).clip(0, 255).to(torch.uint8)
    images = x.permute(2, 0, 3, 1).reshape(n_row*size, n_col*size, 3).cpu().numpy()
    #images = x.permute(0, 3, 1, 4, 2).reshape(2*n_row*size, n_col*size, 3).cpu().numpy()
    
    Image.fromarray(images, 'RGB').save(image_path)
    del images, x, z, c
    torch.cuda.empty_cache()

    if set_train:
        model.train()


def num_to_groups(num, divisor):
    '''
    Compute number of samples in each batch to evenly divide the total eval samples.
    '''
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def sample_fid(args, model, device, rank, set_train=False, cond=False, cfg=False, mi1steps=3, multione=None):
    '''
    Sample args.eval_samples images in parallel for FID and IS calculation. Default 50k images.
    Set set_train to True if you are using the online model for sampling.
    '''
    #Set up diffusion process
    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    # Set up batches for each node
    #vs49: commented for smmoth running
    #assert args.eval_samples % dist.get_world_size() == 0
    samples_per_node = args.eval_samples // dist.get_world_size()
    batches = num_to_groups(samples_per_node, args.eval_batch_size)
    
    # Dist EMA/online evaluation
    # No need to use the DDP wrapper here
    # As we do not need grad sycn (by DDP)
    model.eval()
    model = model.to(device)
    
    n_cls = args.num_classes
    size = args.image_size
    #multione = torch.tensor([1, 3, 5]) if 'MutliOne' in args.trainsetup else None

    images = []
    #use prng for image generation if required
    if args.prng :
        randn_prng = prng(args.eval_samples)
        prng_count = 0
    with torch.no_grad():
        for n in batches:
            if args.prng :
                z = randn_prng[int(prng_count*n): int((prng_count+1)*n)].to(device)
                prng_count += 1
            else :
                z = torch.randn(n, 3, size, size).to(device)
            #vs49: need to modify for multi-step
            if 'MultiOne' in args.trainsetup :
                sigmatime = args.sigmatime
                #timestep embedding setup
                sigma_min, sigma_max = 0.002, 80
                rho = 7.0
                revtsteps = torch.arange(1, mi1steps)
                tsteps = (torch.concat((torch.flip(revtsteps, dims=(0,)), torch.tensor([0])))/mi1steps).to(device)
                sigmas = (sigma_max**(1/rho) + (revtsteps/(mi1steps)) * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
                sigmas = torch.concat((sigmas, torch.tensor([0]))).to(device)
                timeip = sigmas if sigmatime else tsteps
                
                for idx in range(mi1steps) :
                    if idx == 0 :
                        #since cannot concat in first iteration
                        t = (timeip[idx])*torch.ones((z.shape[0],)).to(device)
                    else :
                        #intermediate time index
                        t = torch.cat((t, (timeip[idx])*torch.ones((z.shape[0],)).to(device)), dim=0)
                t = t.to(device)
            elif 'EDMMS' in args.trainsetup :
                #define parameters
                rho = 7
                S_churn, S_min, S_max, S_noise = 0, 0, float('inf'), 1
                sigma_max, sigma_min = 80, 0.002
                num_steps = args.num_sampling_steps
                step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
                t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
                t = torch.zeros((n,)).to(device)
            else :
                t = torch.zeros((n,)).to(device)

            c = torch.randint(0, n_cls, (n,)).to(device) if cond else 10*torch.ones(n, dtype=torch.int).to(device)
            if args.num_sampling_steps == 1 :
                #z = 80*z
                if not cfg :
                    x,_ = model(z, t, c)
                else :
                    if 'MultiOne' not in args.trainsetup :
                        t = torch.cat([t, t], 0)
                    else :
                        t = t.split(z.shape[0], dim=0)
                        #vectorize this piece of code to avoid for-loop
                        for idx in range(len(t)) :
                            tnew = t[0] if idx == 0 else torch.cat((tnew, t[idx]), dim=0)
                            tnew = torch.cat((tnew, t[idx]), dim=0)
                        t = tnew.to(device)
                        
                    z = torch.cat([z, z], dim=0)
                    c_null = torch.tensor([args.num_classes]*n, device=device)
                    c = torch.cat([c, c_null], dim=0)
                    x,_ = model.forward_with_cfg(z, t, c, cfg_scale=1.5)
                images.append(x)
            elif 'EDMMS' in args.trainsetup :
                #sampling loop
                t_next = t_steps[0]
                x_next = z * t_next
                for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                    x_cur = x_next
                    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
                    t_hat = t_cur + gamma * t_cur
                    x_hat =  x_cur + (t_hat ** 2 - t_cur ** 2).clip(min=0).sqrt() * S_noise * torch.randn_like(x_cur)
                    # Euler step.
                    h = t_next - t_hat
                    t_hat_tensor = (t_hat*torch.zeros((n,)).to(device)).view(n, 1, 1, 1)
                    denoised,_ = model(x_hat, t, c, t_hat_tensor)
                    d_cur = (1 / t_hat + t_hat) * x_hat - 1 / t_hat * denoised
                    x_next = x_hat + h * d_cur
                x = x_next
                images.append(x)
        
            else :
                model_kwargs = dict(y=c)
                x = diffusion.p_sample_loop(model.forward,
                                            z.shape,
                                            z, clip_denoised=False,
                                            model_kwargs=model_kwargs,
                                            progress=False,
                                            device=device)
                images.append(x)
                
    images = torch.cat(images, dim=0)
    
    torch.cuda.empty_cache()
    if set_train:
        model.train()

    return images


def compute_fid_is(args, all_images, rank):
    '''
    Compute FID and IS using provided images.
    '''
    # Post-process to images.
    all_images = torch.cat(all_images, dim=0)
    '''
    batch_size = 1024  # Adjust batch size to fit into memory
    all_images = all_images.split(batch_size)
    results = []
    for batch in all_images:
        result = (batch * 127.5 + 128).clip(0, 255).to(torch.uint8).float().div(255).cpu()
        results.append(result)
    all_images = torch.cat(results)
    print("Done normalizing")
    '''
    all_images = (all_images * 127.5 + 128).clip(0, 255).to(torch.uint8).float().div(255).cpu()
    
    # Compute FID & IS
    (IS, IS_std), FID = get_inception_score_and_fid(all_images, args.stat_path)
    torch.cuda.empty_cache()

    return FID, IS
 

def get_sigmas_karras(n, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    # from https://github.com/crowsonkb/k-diffusion
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

def get_fixed_generator_sigma(size, device) :
    """
    Returns sigmas of size `size` with fixed sigmas for the generator. In the paper, it
    is fixed to T-1'th timestep for generator. In practice EDM models are fed sigma value at timestep t.
    """
    sigma = get_sigmas_karras(n=1000, sigma_min=0.002, sigma_max=80.0).to(device)[1]  # sigma_(T-1)
    return torch.tile(sigma, (1, size))

def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True):
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        safe_name = safe_name[:min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)

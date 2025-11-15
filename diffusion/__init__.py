# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(timestep_respacing,
                     noise_schedule="linear", 
                     use_kl=False,
                     sigma_small=False, #vs49: make True for beta as variance instead of beta~
                     predict_xstart=False,
                     learn_sigma=True, #vs49: make False if variance is not predicted by model
                     rescale_learned_sigmas=False,
                     diffusion_steps=1000,
                     dataset='cifar10',
                     staticdist=False,        
                     ):
    #variable "diffusion_steps" controls the "betas" shape[0]
    #which in turn controls the Gaussiandiffusion.num_timesteps value
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            #vs49: change to PREVIOUS_X for predicting x_{t-1} instead of epsilon
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            #gd.ModelMeanType.PREVIOUS_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        # rescale_timesteps=rescale_timesteps,
        dataset=dataset,
        staticdist=staticdist,
    )

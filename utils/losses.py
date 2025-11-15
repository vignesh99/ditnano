import torch
import torch.nn as nn
import torch.nn.functional as F
from piq import LPIPS, DISTS
from utils.diffutils import get_sigmas_karras


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_pred, x):
        return (x_pred - x).abs().mean()

class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_pred, x):
        return ((x_pred - x) ** 2).mean()

class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = LPIPS()

    def forward(self, x_pred, x):
        x_pred = F.interpolate(x_pred, size=224, mode="bilinear")
        x = F.interpolate(x.float(), size=224, mode="bilinear")
        
        x_pred = (x_pred + 1) / 2
        x = (x + 1) / 2

        return self.loss(x_pred, x)


class DISTSLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = DISTS()

    def forward(self, x_pred, x):
        x_pred = F.interpolate(x_pred, size=224, mode="bilinear")
        x = F.interpolate(x.float(), size=224, mode="bilinear")
        
        x_pred = (x_pred + 1) / 2
        x = (x + 1) / 2

        return self.loss(x_pred, x)

class DMDLoss(nn.modules.loss._Loss):
    def __init__(self, train_timesteps=1000, minstepp=0.02, rho=7.0):
        super().__init__()
        self.train_timesteps = train_timesteps
        self.minstepp = 0.02    #min-step percentage
        self.maxstepp = 1-self.minstepp
        karras_sigmas = torch.flip(get_sigmas_karras(self.train_timesteps,
                                                     rho=rho
                                                     ), dims=[0]
                                   )    # small sigma first, large sigma later 
        self.register_buffer("karras_sigmas", karras_sigmas)
        
    def forward(self, x, mu_real, mu_fake, labels) :
        batch_size = x.shape[0]
        with torch.no_grad() :
            #do forward diffusion
            timesteps = torch.randint(int(self.minstepp * self.train_timesteps),
                                      int(self.maxstepp * self.train_timesteps),
                                      [batch_size, 1, 1, 1],
                                      device=x.device
                                      )    #t ~ U(Tmin, Tmax)
            noise = torch.randn_like(x, device=x.device)
            timestep_sigma = self.karras_sigmas[timesteps]
            noisy_x = x + timestep_sigma.reshape(-1, 1, 1, 1) * noise
            
            #obtain images from true teacher and fake teacher
            pred_fake_image = mu_fake(noisy_x, timestep_sigma, class_labels=labels)
            pred_real_image = mu_real(noisy_x, timestep_sigma, class_labels=labels)
            
            #take loss between input and gradient-updated input
            weight_factor = torch.abs(x - pred_real_image).mean(dim=[1, 2, 3],
                                                                keepdim=True)  # Eqn. 8
            grad = (pred_fake_image - pred_real_image) / weight_factor
            diff = (x - grad).detach()  # stop-gradient
            
        return 0.5 * F.mse_loss(x, diff, reduction="mean")

class GeneratorLoss(nn.modules.loss._Loss):
    def __init__(self, train_timesteps=1000, lambda_reg=0.5) :
        super().__init__(self)
        self.dmd_loss = DMDLoss(train_timesteps)
        self.lpips = LPIPSLoss()
        self.l1 = L1Loss()
        self.lambda_reg = lambda_reg

    def forward(self, mu_real, mu_fake, x, x_ref, y_ref, labels):
        loss_kl = self.dmd_loss(x=x,
                                mu_real=mu_real,
                                mu_fake=mu_fake,
                                labels=labels)
        loss_reg = self.lpips(x_ref, y_ref)
        #loss_reg = self.l1(x_ref, y_ref)
        #print(loss_reg.item())
        return self.lambda_reg * loss_reg + (1 - self.lambda_reg) * loss_kl#, loss_kl

class DenoisingLoss(nn.modules.loss._Loss):
    def __init__(self, train_timesteps=1000, rho=7.0):
        super().__init__()
        self.train_timesteps = train_timesteps
        karras_sigmas = torch.flip(get_sigmas_karras(self.train_timesteps,
                                                     rho=rho
                                                     ), dims=[0]
                                   )    # small sigma first, large sigma later
        self.register_buffer("karras_sigmas", karras_sigmas)
    
    def forward(self, mu_fake, x, labels):
        #do forward diffusion
        batch_size = x.shape[0]
        x = x.detach()    #no gradient to generator
        timesteps = torch.randint(0,    #start with 1 due to divide by 0?
                                  self.train_timesteps,
                                  [batch_size, 1, 1, 1],
                                  device=x.device
                                  )
        noise = torch.randn_like(x)
        timestep_sigma = self.karras_sigmas[timesteps]
        noisy_x = x + timestep_sigma.reshape(-1, 1, 1, 1) * noise
        
        pred_fake_image = mu_fake(noisy_x, timestep_sigma, class_labels=labels)
        # Algorithm SNR + 1 / sigma_data^2 for EDM (sigma_data = 0.5)
        weight = (1 / timestep_sigma**2) + (1 / mu_fake.module.sigma_data**2)
        v = pred_fake_image
        #print("generator: mean = ", x.mean().item(), "variance = ", x.var().item(), "min = ", x.min().item(), "max = ", x.max().item())
        #print("fake model: mean = ", v.mean().item(), "variance = ", v.var().item(), "min = ", v.min().item(), "max = ", v.max().item())
        
        return torch.mean(weight * (pred_fake_image - x) ** 2)

class LayerwiseLoss(nn.modules.loss._Loss):
    def __init__(self, lambda_dist=0.25, lambda_layer=0.1) :
        super().__init__(self)
        self.lmain = LPIPSLoss()
        self.ldist = LPIPSLoss()
        self.llayer = L2Loss() #LPIPSLoss()
        #self.scalemain = (1 - lambda_dist - lambda_layer)
        self.scaledist = lambda_dist
        self.scalelayer = lambda_layer
        
    def forward(self, x_stud, x_true, x_teach, attn_stud, attn_teach) :
        #compute loss with EDM dataset and DiT teacher output
        #mainloss = self.lmain(x_stud, x_true)
        distloss = self.ldist(x_stud, x_teach)
        #make DiT teacher attention tensor shape-compatabile with DiT-Nano
        #L, B, H, N, Dh = attn_teach.shape
        #taking mean across all teacher attention heads
        '''
        assert H % 3 == 0
        attn_teach = attn_teach.view(L, B, H//3, 3, N, Dh)
        attn_teach = attn_teach.mean(dim=3)    #automatically eliminates the dimension
        attn_teach = attn_teach[1::2]    #skip alternate layers for layer matching
        assert attn_teach.shape == attn_stud.shape
        '''
        #compute layer-wise loss
        #layerloss = self.llayer(attn_stud, attn_teach)

        #loss = self.scalemain * mainloss + self.scaledist * distloss + self.scalelayer * layerloss
        #loss = 0.5*mainloss + 0.25*distloss + 0.25*layerloss
        loss = distloss
        return loss

class MultiOneLoss(nn.Module) :
    def __init__(self, scale=0.5) :
        super().__init__()
        self.final = LPIPSLoss()
        self.layer = LPIPSLoss()
        self.scale = scale

    def forward(self, x_pred, x, xint_pred, xint) :
        final = self.final(x_pred, x)
        layer = self.layer(xint_pred, xint)
        #exit()

        return (1 - self.scale) * final + self.scale * layer
    
loss_dict = {
    'l1': L1Loss,
    'l2': L2Loss,
    'lpips': LPIPSLoss,
    'dists': DISTSLoss,
    'gen': GeneratorLoss,
    'dmd': DMDLoss,
    'denoise': DenoisingLoss,
    'gan': None,
    'lay': LayerwiseLoss,
    'multione': MultiOneLoss,
        }

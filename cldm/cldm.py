import os
import einops
import torch
import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import ToTensor, ToPILImage
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
import time
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torchvision.transforms.functional import crop
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from MySDSR.dataload import feed_data
import yaml
from PIL import Image

class ImageSpliterTh:
    def __init__(self, im, pch_size, stride, sf=1):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
        '''
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf

        bs, chn, height, width= im.shape
        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.length = self.__len__()
        self.num_pchs = 0

        self.im_ori = im
        self.im_res = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)
        self.pixel_count = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [0,]
        else:
            starts = list(range(0, length, self.stride))
            for i in range(len(starts)):
                if starts[i] + self.pch_size > length:
                    starts[i] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_pchs < self.length:
            w_start_idx = self.num_pchs // len(self.height_starts_list)
            w_start = self.width_starts_list[w_start_idx]
            w_end = w_start + self.pch_size

            h_start_idx = self.num_pchs % len(self.height_starts_list)
            h_start = self.height_starts_list[h_start_idx]
            h_end = h_start + self.pch_size

            pch = self.im_ori[:, :, h_start:h_end, w_start:w_end,]

            h_start *= self.sf
            h_end *= self.sf
            w_start *= self.sf
            w_end *= self.sf

            self.w_start, self.w_end = w_start, w_end
            self.h_start, self.h_end = h_start, h_end

            self.num_pchs += 1
        else:
            raise StopIteration()

        return pch, (h_start, h_end, w_start, w_end)

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: n x c x pch_size x pch_size, float
            index_infos: (h_start, h_end, w_start, w_end)
        '''
        if index_infos is None:
            w_start, w_end = self.w_start, self.w_end
            h_start, h_end = self.h_start, self.h_end
        else:
            h_start, h_end, w_start, w_end = index_infos

        self.im_res[:, :, h_start:h_end, w_start:w_end] += pch_res
        self.pixel_count[:, :, h_start:h_end, w_start:w_end] += 1

    def gather(self):
        assert torch.all(self.pixel_count != 0)
        return self.im_res.div(self.pixel_count)

def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class ControlLDM(LatentDiffusion):

    def __init__(self, structcond_stage_config, deg_locked, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Encoder_degrad = instantiate_from_config(structcond_stage_config)
        self.deg_locked = deg_locked
        with open('./MySDSR/config.yaml', encoding='utf-8')as file:
            content = file.read()
            self.data = yaml.load(content, Loader=yaml.FullLoader)
    @torch.no_grad()    
    def get_input(self, batch, k, bs=None, *args, **kwargs):  
        im_lq, im_gt = feed_data(batch, self.data)
        
        im_gt = (im_gt - 0.5) / 0.5
        lr = (im_lq - 0.5) / 0.5

        lr = einops.rearrange(lr, 'b c h w  -> b h w c')

        im_gt = einops.rearrange(im_gt, 'b c h w  -> b h w c') 
        batch = dict(jpg=im_gt, lr=lr, txt=batch['txt'])
        z, all_conds = super().get_input(batch, k, bs=None, *args, **kwargs)  
        lr = im_lq
        lr = lr.to(self.device)
 
        lr = lr.to(memory_format=torch.contiguous_format).float()
        

        return z,  dict(lr=lr)

    @torch.no_grad()
    def get_input_test(self, batch, k, bs=None, *args, **kwargs):
        im_lq, im_gt = batch['im_lq'], batch['im_gt']
  
        im_gt = (im_gt - 0.5) / 0.5
        lr = (im_lq - 0.5) / 0.5
   
        lr = einops.rearrange(lr, 'b c h w  -> b h w c')
   
        im_gt = einops.rearrange(im_gt, 'b c h w  -> b h w c') 
        batch = dict(jpg=im_gt, lr=lr, txt=batch['txt'])
        z, all_conds = super().get_input(batch, k, bs=None, *args, **kwargs)  #del clip
        lr = im_lq
        lr = lr.to(self.device)

        lr = lr.to(memory_format=torch.contiguous_format).float()
        

        return z,  dict(lr=lr)
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        lr = cond['lr']
    
        z_lr = cond['latent']
        struc_c, context = self.Encoder_degrad(z_lr, t, lr)
        
        eps = diffusion_model(x=x_noisy, timesteps=t, context=context, struct_cond=struc_c)
        

        return eps

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images = self.log_images_test(batch)
        for k in images:
                N = min(images[k].shape[0], 4)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    
                    images[k] = torch.clamp(images[k], -1., 1.)
        root = os.path.join('./test', "test")
        for k in images:
            subfolder_path = os.path.join(root, k)
            os.makedirs(subfolder_path, exist_ok=True)
            for i in range(images[k].size(0)):
                img = images[k][i].detach().cpu().numpy().transpose(1, 2, 0)
                img = ((img + 1.0) / 2.0 * 255).astype(np.uint8)

                filename = os.path.basename(batch['txt'][0])
                path = os.path.join(subfolder_path, filename)
                Image.fromarray(img).save(path)

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=1.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, cond = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat = cond['lr']
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        lrup = torch.nn.functional.interpolate(log["control"], scale_factor=4, mode='bicubic', align_corners=False)
        lr_posterior = self.encode_first_stage(lrup)
        z_lr = self.get_first_stage_encoding(lr_posterior).detach()
        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond=cond,
                                                     batch_size=N, ddim=use_ddim, x0=z_lr,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log
    
    @torch.no_grad()
    def log_images_test(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=1.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=1.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input_test(batch, self.first_stage_key, bs=N)
        c_cat = c["lr"][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        lr = c_cat 
        
        lr_up = nn.functional.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)
        ori_h, ori_w =lr_up.shape[2:]
        if not (ori_h % 64 == 0 and ori_w % 64 == 0):
            flag_pad = True
            pad_h = ((ori_h // 64) + 1) * 64 - ori_h
            pad_w = ((ori_w // 64) + 1) * 64 - ori_w
            lr_up = F.pad(lr_up, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False
        lr_up_log = lr_up * 2 -1
        log['lr_up'] = lr_up_log
        lr_latent = self.encode_first_stage(lr_up)
        x0 = self.get_first_stage_encoding(lr_latent).detach()
        b, ch, h, w = x0.shape
        

        if sample:
            if lr_up.shape[2] > 1024 or lr_up.shape[3] > 1024:
                im_spliter = ImageSpliterTh(lr_up, 1024, 800, sf=1)
                for im_lq_pch, index_infos in im_spliter:
                    lr_latent_pch = self.encode_first_stage(im_lq_pch)
                    x0 = self.get_first_stage_encoding(lr_latent_pch).detach()
                    b, ch, h, w = x0.shape
                    # get denoise row
                    c['latent'] = x0
                    samples, z_denoise_row = self.sample_log(cond=c,
                                                            batch_size=N, ddim=use_ddim,x0=x0,
                                                            ddim_steps=ddim_steps, eta=ddim_eta, h=h, w=w)
                    x_samples = self.decode_first_stage(samples)    
                        
                    x_samples = adaptive_instance_normalization(x_samples, im_lq_pch *2-1)
                        
                    im_spliter.update(x_samples, index_infos)
                x_samples = im_spliter.gather()
            # get denoise row
            else:
                c['latent'] = x0
                samples, z_denoise_row = self.sample_log(cond=c,
                                                        batch_size=N, ddim=use_ddim,x0=x0,
                                                        ddim_steps=ddim_steps, eta=ddim_eta, h=h, w=w)
                x_samples = self.decode_first_stage(samples)
                #x_samples = adaptive_instance_normalization(x_samples, lr_up_log)
                x_samples = crop(x_samples, 0, 0, ori_h, ori_w)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid


        return log
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, h, w, **kwargs):
        ddim_sampler = DDIMSampler(self)
        #b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h, w)
        start_time = time.time()
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=True, **kwargs)
        end_time = time.time()
        inference_time = end_time - start_time
        print("Inference time:", inference_time, "seconds")
        #samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates
    
    def configure_optimizers(self):
        lr = self.learning_rate
        model_params = [p for name, p in self.model.diffusion_model.named_parameters() if ('spade' in name  or 'attn2' in name)]
        degrad_params = [p for name, p in self.Encoder_degrad.named_parameters() if 'degrad_encoder' not in name]
        model_param_names = [name for name, _ in self.Encoder_degrad.named_parameters() if 'degrad_encoder' not in name]
        print(model_param_names)
 
        optimizer = torch.optim.AdamW(params=degrad_params+model_params, lr=lr)

        return optimizer
    
    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

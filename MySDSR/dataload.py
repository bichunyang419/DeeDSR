import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
import yaml
from pathlib import Path
from torch.utils import data as data

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


from MySDSR.degradation import *


class RealESRGANDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        # support multiple type of data: file path and meta data, remove support of lmdb
        self.paths = []
        if 'meta_info' in opt:
            with open(self.opt['meta_info']) as fin:
                    paths = [line.strip().split(' ')[0] for line in fin]
                    self.paths = [v for v in paths]
            if 'meta_num' in opt:
                self.paths = sorted(self.paths)[:opt['meta_num']]
        if 'gt_path' in opt:
            if isinstance(opt['gt_path'], str):
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path']).glob('*.'+opt['image_type'])]))
            else:
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['gt_path']) > 1:
                    for i in range(len(opt['gt_path'])-1):
                        self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]).glob('*.'+opt['image_type'])]))
        if 'imagenet_path' in opt:
            class_list = os.listdir(opt['imagenet_path'])
            for class_file in class_list:
                self.paths.extend(sorted([str(x) for x in Path(os.path.join(opt['imagenet_path'], class_file)).glob('*.'+'JPEG')]))
        if 'face_gt_path' in opt:
            if isinstance(opt['face_gt_path'], str):
                face_list = sorted([str(x) for x in Path(opt['face_gt_path']).glob('*.'+opt['image_type'])])
                self.paths.extend(face_list[:opt['num_face']])
            else:
                face_list = sorted([str(x) for x in Path(opt['face_gt_path'][0]).glob('*.'+opt['image_type'])])
                self.paths.extend(face_list[:opt['num_face']])
                if len(opt['face_gt_path']) > 1:
                    for i in range(len(opt['face_gt_path'])-1):
                        self.paths.extend(sorted([str(x) for x in Path(opt['face_gt_path'][0]).glob('*.'+opt['image_type'])])[:opt['num_face']])

        # limit number of pictures for test
        if 'num_pic' in opt:
            if 'val' or 'test' in opt:
                random.shuffle(self.paths)
                self.paths = self.paths[:opt['num_pic']]
            else:
                self.paths = self.paths[:opt['num_pic']]

        if 'mul_num' in opt:
            self.paths = self.paths * opt['mul_num']
            # print('>>>>>>>>>>>>>>>>>>>>>')
            # print(self.paths)

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)
        # filter the dataset and remove images with too low quality
        img_size = os.path.getsize(gt_path)
        img_size = img_size/1024

        while img_gt.shape[0] * img_gt.shape[1] < 384*384 or img_size<100:
            index = random.randint(0, self.__len__()-1)
            gt_path = self.paths[index]

            time.sleep(0.1)  # sleep 1s for occasional server congestion
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_size = os.path.getsize(gt_path)
            img_size = img_size/1024

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.crop_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            # top = (h - crop_pad_size) // 2 -1
            # left = (w - crop_pad_size) // 2 -1
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path, 'txt':''} #'Restore degraded images, with the degradation process as follows: '}
        
        return return_d

    def __len__(self):
        return len(self.paths)
    
class TestDataset(data.Dataset):
    def __init__(self, opt):
        super(TestDataset, self).__init__()
        self.lr_path = opt['lr_path']
        self.hr_path = opt['hr_path']
        self.lr_list = sorted(os.listdir(opt['lr_path']))
        self.hr_list = sorted(os.listdir(opt['hr_path']))
        self.file_client = FileClient('disk')
        
    def __getitem__(self, index):
        img_lr_path = os.path.join(self.lr_path, self.lr_list[index])
        img_bytes_lr = self.file_client.get(img_lr_path)
        img_lq = imfrombytes(img_bytes_lr, float32=True)
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        
        img_hr_path = os.path.join(self.hr_path, self.hr_list[index])
        img_bytes_gt = self.file_client.get(img_hr_path)
        img_gt = imfrombytes(img_bytes_gt, float32=True)
        
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        batch = {'im_lq': img_lq, 'im_gt':img_gt, 'txt': img_lr_path}
        return batch
    
    def __len__(self):
        return len(self.hr_list)
    
def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


                
def feed_data(batch, configs, use_usm=False):

    """Degradation pipeline, modified from Real-ESRGAN:
    https://github.com/xinntao/Real-ESRGAN
    """


    jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
    usm_sharpener = USMSharp().cuda()  # do usm sharpening

    im_gt = batch['gt'].cuda()
    if use_usm:
        im_gt = usm_sharpener(im_gt)
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    kernel1 = batch['kernel1'].cuda()
    kernel2 = batch['kernel2'].cuda()
    sinc_kernel = batch['sinc_kernel'].cuda()

    ori_h, ori_w = im_gt.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(im_gt, kernel1)

    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            configs['degradation']['resize_prob'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, configs['degradation']['resize_range'][1])
    elif updown_type == 'down':
        scale = random.uniform(configs['degradation']['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)

    # add noise
    gray_noise_prob = configs['degradation']['gray_noise_prob']
    if random.random() < configs['degradation']['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=configs['degradation']['noise_range'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )

    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=configs['degradation']['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)

    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs['degradation']['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if random.random() < configs['degradation']['second_blur_prob']:
        out = filter2D(out, kernel2)

    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            configs['degradation']['resize_prob2'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, configs['degradation']['resize_range2'][1])
    elif updown_type == 'down':
        scale = random.uniform(configs['degradation']['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
            out,
            size=(int(ori_h / configs['sf'] * scale),
                    int(ori_w / configs['sf'] * scale)),
            mode=mode,
            )

    # add noise
    gray_noise_prob = configs['degradation']['gray_noise_prob2']
    if random.random() < configs['degradation']['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=configs['degradation']['noise_range2'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )

    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=configs['degradation']['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
            )

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if random.random() < 0.5:
        # resize back + the final sinc filter

        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // configs['sf'],
                        ori_w // configs['sf']),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs['degradation']['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)

    else:

        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs['degradation']['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // configs['sf'],
                        ori_w // configs['sf']),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)
 
    # clamp and round
    im_lq = torch.clamp(out, 0, 1.0)

    # random crop
    gt_size = configs['degradation']['gt_size']
    im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, configs['sf'])

    
    if random.random() < 0.01 or torch.isnan(im_lq).any():
        im_lq = nn.functional.interpolate(im_gt, scale_factor=0.25, mode='bicubic', align_corners=False)
    

    return im_lq, im_gt

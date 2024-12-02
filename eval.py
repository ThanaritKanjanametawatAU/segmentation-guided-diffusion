"""
model evaluation/sampling
"""
import math
import os
import torch
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from copy import deepcopy
import numpy as np

import diffusers
from diffusers import DiffusionPipeline, ImagePipelineOutput, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor 

from utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

####################
# segmentation-guided DDPM
####################

def evaluate_sample_many(
    sample_size,
    config,
    model,
    noise_scheduler,
    eval_dataloader,
    device='cuda'
    ):

    # for loading segs to condition on:
    # setup for sampling
    if config.model_type == "DDPM":
        if config.segmentation_guided:
            pipeline = SegGuidedDDPMPipeline(
                unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                )
        else:
            pipeline = diffusers.DDPMPipeline(unet=model.module, scheduler=noise_scheduler)
    elif config.model_type == "DDIM":
        if config.segmentation_guided:
            pipeline = SegGuidedDDIMPipeline(
                unet=model.module, scheduler=noise_scheduler, eval_dataloader=eval_dataloader, external_config=config
                )
        else:
            pipeline = diffusers.DDIMPipeline(unet=model.module, scheduler=noise_scheduler)


    sample_dir = test_dir = os.path.join(config.output_dir, "samples_many_{}".format(sample_size))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    num_sampled = 0
    # keep sampling images until we have enough
    for bidx, seg_batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        if num_sampled < sample_size:
            if config.segmentation_guided:
                current_batch_size = [v for k, v in seg_batch.items() if k.startswith("seg_")][0].shape[0]
            else:
                current_batch_size = config.eval_batch_size

            if config.segmentation_guided:
                images = pipeline(
                    batch_size = current_batch_size,
                    seg_batch=seg_batch,
                ).images
            else:
                images = pipeline(
                    batch_size = current_batch_size,
                ).images

            # save each image in the list separately
            for i, img in enumerate(images):
                if config.segmentation_guided:
                    # name base on input mask fname
                    img_fname = "{}/condon_{}".format(sample_dir, seg_batch["image_filenames"][i])
                else:
                    img_fname = f"{sample_dir}/{num_sampled + i:04d}.png"
                img.save(img_fname)

            num_sampled += len(images)
            print("sampled {}/{}.".format(num_sampled, sample_size))



def evaluate_generation(config, model, noise_scheduler, eval_dataloader, device='cuda'):
    # Setup for sampling
    if config.model_type == "DDPM":
        pipeline = PCBDiffusionPipeline(
            unet=model.module, 
            scheduler=noise_scheduler, 
            external_config=config
        )
    elif config.model_type == "DDIM":
        pipeline = PCBDiffusionPipeline(
            unet=model.module,
            scheduler=noise_scheduler,
            external_config=config
        )

    # Get a batch of master images to condition on
    eval_batch = next(iter(eval_dataloader))
    master_images = eval_batch['master_images'].to(device)

    # Generate images
    images = pipeline(
        batch_size=config.eval_batch_size,
        master_image=master_images,
    ).images

    # Make a grid out of the images
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    image_grid = make_grid(images, rows=rows, cols=cols)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/samples.png")

    # Also save the conditioning master images for reference
    master_grid = make_grid([img for img in master_images], rows=rows, cols=cols)
    master_grid.save(f"{test_dir}/master_images.png")

def convert_segbatch_to_multiclass(shape, segmentations_batch, config, device):
    # NOTE: this generic function assumes that segs don't overlap
    # put all segs on same channel
    segs = torch.zeros(shape).to(device)
    for k, seg in segmentations_batch.items():
        if k.startswith("seg_"):
            seg = seg.to(device)
            segs[segs == 0] = seg[segs == 0]

    if config.use_ablated_segmentations:
        # randomly remove class labels from segs with some probability
        segs = ablate_masks(segs, config)

    return segs

def ablate_masks(segs, config, method="equal_weighted"):
    # randomly remove class label(s) from segs with some probability 
    if method == "equal_weighted":
        """
        # give equal probability to each possible combination of removing non-background classes
        # NOTE: requires that each class has a value in ({0, 1, 2, ...} / 255)
        # which is by default if the mask file was saved as {0, 1, 2 ,...} and then normalized by default to [0, 1] by transforms.ToTensor()
        # num_segmentation_classes
        """
        class_removals = (torch.rand(config.num_segmentation_classes - 1) < 0.5).int().bool().tolist()
        for class_idx, remove_class in enumerate(class_removals):
            if remove_class:
                segs[(255 * segs).int() == class_idx + 1] = 0

    elif method == "by_class":
        class_ablation_prob = 0.3
        for seg_value in segs.unique():
            if seg_value != 0:
                # remove seg with some probability
                if torch.rand(1).item() < class_ablation_prob:
                    segs[segs == seg_value] = 0
    
    else:
        raise NotImplementedError
    return segs

def add_segmentations_to_noise(noisy_images, segmentations_batch, config, device):
    """
    concat segmentations to noisy image
    """

    if config.segmentation_channel_mode == "single":
        multiclass_masks_shape = (noisy_images.shape[0], 1, noisy_images.shape[2], noisy_images.shape[3])
        segs = convert_segbatch_to_multiclass(multiclass_masks_shape, segmentations_batch, config, device) 
        # concat segs to noise
        noisy_images = torch.cat((noisy_images, segs), dim=1)
        
    elif config.segmentation_channel_mode == "multi":
        raise NotImplementedError

    return noisy_images

####################
# general DDPM
####################
def evaluate(config, epoch, pipeline, master_images=None):
    """Evaluate model by generating samples and saving them as image grids"""
    # Generate samples
    images = pipeline(
        batch_size=config.eval_batch_size,
        master_image=master_images,
    ).images

    # Make a grid out of the images
    cols = min(8, len(images))
    rows = math.ceil(len(images) / cols)
    image_grid = make_grid(images, rows=rows, cols=cols)

    # Save the images
    sample_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    image_grid.save(f"{sample_dir}/sample_{epoch:04d}.png")

    # Also save master images if provided
    if master_images is not None:
        master_images_list = [img for img in master_images]
        master_grid = make_grid(master_images_list, rows=rows, cols=cols)
        master_grid.save(f"{sample_dir}/master_{epoch:04d}.png")

# custom diffusers pipelines for sampling from segmentation-guided models
class SegGuidedDDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for segmentation-guided image generation, modified from DDPMPipeline.
    generates both-class conditioned and unconditional images if using class-conditional model without CFG, or just generates 
    conditional images guided by CFG.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
        eval_dataloader ([`torch.utils.data.DataLoader`]):
            Dataloader to load the evaluation dataset of images and their segmentations. Here only uses the segmentations to generate images.
    """
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, eval_dataloader, external_config):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.eval_dataloader = eval_dataloader
        self.external_config = external_config # config is already a thing

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        seg_batch: Optional[torch.Tensor] = None,
        class_label_cfg: Optional[int] = None,
        translate = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            seg_batch (`torch.Tensor`, *optional*, defaults to None):
                batch of segmentations to condition generation on
            class_label_cfg (`int`, *optional*, defaults to `None`):
                class label to condition generation on using CFG, if using class-conditional model

            OPTIONS FOR IMAGE TRANSLATION:
            translate (`bool`, *optional*, defaults to False):
                whether to translate images from the source domain to the target domain

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if self.external_config.segmentation_channel_mode == "single":
            img_channel_ct = self.unet.config.in_channels - 1
        elif self.external_config.segmentation_channel_mode == "multi":
            img_channel_ct = self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")])

        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                img_channel_ct,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            if self.external_config.segmentation_channel_mode == "single":
                image_shape = (batch_size, self.unet.config.in_channels - 1, *self.unet.config.sample_size)
            elif self.external_config.segmentation_channel_mode == "multi":
                image_shape = (batch_size, self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")]), *self.unet.config.sample_size)
            

        # initiate latent variable to sample from
        if not translate:
            # normal sampling; start from noise
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                image = randn_tensor(image_shape, generator=generator)
                image = image.to(self.device)
            else:
                image = randn_tensor(image_shape, generator=generator, device=self.device)
        else:
            # image translation sampling; start from source domain images, add noise up to certain step, then being there for denoising
            trans_start_t = int(self.external_config.trans_noise_level * self.scheduler.config.num_train_timesteps)

            trans_start_images = seg_batch["images"]

            # Sample noise to add to the images
            noise = torch.randn(trans_start_images.shape).to(trans_start_images.device)
            timesteps = torch.full(
                (trans_start_images.size(0),),
                trans_start_t, 
                device=trans_start_images.device
                ).long()
            image = self.scheduler.add_noise(trans_start_images, noise, timesteps)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            if translate:
                # if doing translation, start at chosen time step given partially-noised image
                # skip all earlier time steps (with higher t)
                if t >= trans_start_t:
                  continue

            # 1. predict noise model_output
            # first, concat segmentations to noise
            image = add_segmentations_to_noise(image, seg_batch, self.external_config, self.device)

            if self.external_config.class_conditional:
                if class_label_cfg is not None:
                    class_labels = torch.full([image.size(0)], class_label_cfg).long().to(self.device)
                    model_output_cond = self.unet(image, t, class_labels=class_labels).sample
                    if self.external_config.use_cfg_for_eval_conditioning:
                        # use classifier-free guidance for sampling from the given class

                        if self.external_config.cfg_maskguidance_condmodel_only:
                            image_emptymask = torch.cat((image[:, :img_channel_ct, :, :], torch.zeros_like(image[:, img_channel_ct:, :, :])), dim=1)
                            model_output_uncond = self.unet(image_emptymask, t, 
                                    class_labels=torch.zeros_like(class_labels).long()).sample
                        else:
                            model_output_uncond = self.unet(image, t, 
                                    class_labels=torch.zeros_like(class_labels).long()).sample

                        # use cfg equation
                        model_output = (1. + self.external_config.cfg_weight) * model_output_cond - self.external_config.cfg_weight * model_output_uncond
                    else:
                        # just use normal conditioning
                        model_output = model_output_cond
               
                else:
                    # or, just use basic network conditioning to sample from both classes
                    if self.external_config.class_conditional:
                        # if training conditionally, evaluate source domain samples
                        class_labels = torch.ones(image.size(0)).long().to(self.device)
                        model_output = self.unet(image, t, class_labels=class_labels).sample
            else:
                model_output = self.unet(image, t).sample
            # output is slightly denoised image

            # 2. compute previous image: x_t -> x_t-1
            # but first, we're only adding denoising the image channel (not seg channel),
            # so remove segs
            image = image[:, :img_channel_ct, :, :]
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # if training conditionally, also evaluate for target domain images
        # if not using chosen class for CFG
        if self.external_config.class_conditional and class_label_cfg is None:
            image_target_domain = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                # first, concat segmentations to noise
                # no masks in target domain so just use blank masks
                image_target_domain = torch.cat((image_target_domain, torch.zeros_like(image_target_domain)), dim=1)

                if self.external_config.class_conditional:
                    # if training conditionally, also evaluate unconditional model and target domain (no masks)
                    class_labels = torch.cat([torch.full([image_target_domain.size(0) // 2], 2), torch.zeros(image_target_domain.size(0)) // 2]).long().to(self.device)
                    model_output = self.unet(image_target_domain, t, class_labels=class_labels).sample
                else:
                    model_output = self.unet(image_target_domain, t).sample

                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                # but first, we're only adding denoising the image channel (not seg channel),
                # so remove segs
                image_target_domain = image_target_domain[:, :img_channel_ct, :, :]
                image_target_domain = self.scheduler.step(
                    model_output, t, image_target_domain, generator=generator
                ).prev_sample

            image = torch.cat((image, image_target_domain), dim=0)
            # will output source domain images first, then target domain images

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

class SegGuidedDDIMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation, modified for seg-guided image gen.
    modified from diffusers.DDIMPipeline.
    generates both-class conditioned and unconditional images if using class-conditional model without CFG, or just generates 
    conditional images guided by CFG.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
        eval_dataloader ([`torch.utils.data.DataLoader`]):
            Dataloader to load the evaluation dataset of images and their segmentations. Here only uses the segmentations to generate images.
    
    """
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, eval_dataloader, external_config):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, eval_dataloader=eval_dataloader, external_config=external_config)
        # ^ some reason necessary for DDIM but not DDPM.

        self.eval_dataloader = eval_dataloader
        self.external_config = external_config # config is already a thing

        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)


    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        seg_batch: Optional[torch.Tensor] = None,
        class_label_cfg: Optional[int] = None,
        translate = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
                DDIM and `1` corresponds to DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
                downstream to the scheduler (use `None` for schedulers which don't support this argument).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            seg_batch (`torch.Tensor`, *optional*):
                batch of segmentations to condition generation on
            class_label_cfg (`int`, *optional*, defaults to `None`):
                class label to condition generation on using CFG, if using class-conditional model

            OPTIONS FOR IMAGE TRANSLATION:
            translate (`bool`, *optional*, defaults to False):
                whether to translate images from the source domain to the target domain

        Example:

        ```py

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        # Sample gaussian noise to begin loop
        if self.external_config.segmentation_channel_mode == "single":
            img_channel_ct = self.unet.config.in_channels - 1
        elif self.external_config.segmentation_channel_mode == "multi":
            img_channel_ct = self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")])

        if isinstance(self.unet.config.sample_size, int):
            if self.external_config.segmentation_channel_mode == "single":
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels - 1,
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
            elif self.external_config.segmentation_channel_mode == "multi":
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")]),
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
        else:
            if self.external_config.segmentation_channel_mode == "single":
                image_shape = (batch_size, self.unet.config.in_channels - 1, *self.unet.config.sample_size)
            elif self.external_config.segmentation_channel_mode == "multi":
                image_shape = (batch_size, self.unet.config.in_channels - len([k for k in seg_batch.keys() if k.startswith("seg_")]), *self.unet.config.sample_size)
            
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # initiate latent variable to sample from
        if not translate:
            # normal sampling; start from noise
            image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        else:
            # image translation sampling; start from source domain images, add noise up to certain step, then being there for denoising
            trans_start_t = int(self.external_config.trans_noise_level * self.scheduler.config.num_train_timesteps)

            trans_start_images = seg_batch["images"].to(self._execution_device)

            # Sample noise to add to the images
            noise = torch.randn(trans_start_images.shape).to(trans_start_images.device)
            timesteps = torch.full(
                (trans_start_images.size(0),),
                trans_start_t, 
                device=trans_start_images.device
                ).long()
            image = self.scheduler.add_noise(trans_start_images, noise, timesteps)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            if translate:
                # if doing translation, start at chosen time step given partially-noised image
                # skip all earlier time steps (with higher t)
                if t >= trans_start_t:
                  continue

            # 1. predict noise model_output
            # first, concat segmentations to noise
            image = add_segmentations_to_noise(image, seg_batch, self.external_config, self.device)

            if self.external_config.class_conditional:
                if class_label_cfg is not None:
                    class_labels = torch.full([image.size(0)], class_label_cfg).long().to(self.device)
                    model_output_cond = self.unet(image, t, class_labels=class_labels).sample
                    if self.external_config.use_cfg_for_eval_conditioning:
                        # use classifier-free guidance for sampling from the given class
                        if self.external_config.cfg_maskguidance_condmodel_only:
                            image_emptymask = torch.cat((image[:, :img_channel_ct, :, :], torch.zeros_like(image[:, img_channel_ct:, :, :])), dim=1)
                            model_output_uncond = self.unet(image_emptymask, t, 
                                    class_labels=torch.zeros_like(class_labels).long()).sample
                        else:
                            model_output_uncond = self.unet(image, t, 
                                    class_labels=torch.zeros_like(class_labels).long()).sample

                        # use cfg equation
                        model_output = (1. + self.external_config.cfg_weight) * model_output_cond - self.external_config.cfg_weight * model_output_uncond
                    else:
                        model_output = model_output_cond
               
                else:
                    # or, just use basic network conditioning to sample from both classes
                    if self.external_config.class_conditional:
                        # if training conditionally, evaluate source domain samples
                        class_labels = torch.ones(image.size(0)).long().to(self.device)
                        model_output = self.unet(image, t, class_labels=class_labels).sample
            else:
                model_output = self.unet(image, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            # but first, we're only adding denoising the image channel (not seg channel),
            # so remove segs
            image = image[:, :img_channel_ct, :, :]
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        # if training conditionally, also evaluate for target domain images
        # if not using chosen class for CFG
        if self.external_config.class_conditional and class_label_cfg is None:
            image_target_domain = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                # first, concat segmentations to noise
                # no masks in target domain so just use blank masks
                image_target_domain = torch.cat((image_target_domain, torch.zeros_like(image_target_domain)), dim=1)

                if self.external_config.class_conditional:
                    # if training conditionally, also evaluate unconditional model and target domain (no masks)
                    class_labels = torch.cat([torch.full([image_target_domain.size(0) // 2], 2), torch.zeros(image_target_domain.size(0) // 2)]).long().to(self.device)
                    model_output = self.unet(image_target_domain, t, class_labels=class_labels).sample
                else:
                    model_output = self.unet(image_target_domain, t).sample

                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                # but first, we're only adding denoising the image channel (not seg channel),
                # so remove segs
                image_target_domain = image_target_domain[:, :img_channel_ct, :, :]
                image_target_domain = self.scheduler.step(
                    model_output, t, image_target_domain, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
                ).prev_sample

            image = torch.cat((image, image_target_domain), dim=0)
            # will output source domain images first, then target domain images

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


# In eval.py, add this new pipeline class:

class PCBDiffusionPipeline(DiffusionPipeline):
    """Pipeline for PCB defect generation using paired normal/defect images"""
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, external_config):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.external_config = external_config

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        master_image: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Start from random noise
        if self.device.type == "mps":
            x_t = torch.randn((batch_size, 3, self.unet.config.sample_size, self.unet.config.sample_size),
                             generator=generator)
            x_t = x_t.to(self.device)
        else:
            x_t = torch.randn((batch_size, 3, self.unet.config.sample_size, self.unet.config.sample_size),
                             generator=generator,
                             device=self.device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for t in self.progress_bar(self.scheduler.timesteps):
            # Concatenate master image with current noisy image
            model_input = torch.cat([x_t, master_image], dim=1)
            
            # Get model prediction
            model_output = self.unet(model_input, t).sample
            
            # Update sample with scheduler
            x_t = self.scheduler.step(model_output, t, x_t, generator=generator).prev_sample

        # Convert to images
        x_0 = (x_t / 2 + 0.5).clamp(0, 1)
        x_0 = x_0.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            x_0 = self.numpy_to_pil(x_0)

        if not return_dict:
            return (x_0,)

        return ImagePipelineOutput(images=x_0)
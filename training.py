"""
training utils
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from datetime import timedelta

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import diffusers

from eval import evaluate, SegGuidedDDPMPipeline, SegGuidedDDIMPipeline, PCBDiffusionPipeline

@dataclass
class TrainingConfig:
    model_type: str = "DDPM"
    image_size: int = 256  # the generated image resolution
    train_batch_size: int = 32
    eval_batch_size: int = 8  # how many images to sample during evaluation
    num_epochs: int = 200
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 20
    save_model_epochs: int = 30
    mixed_precision: str = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = None

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0

    # custom options
    segmentation_guided: bool = False
    segmentation_channel_mode: str = "single"
    num_segmentation_classes: int = None # INCLUDING background
    use_ablated_segmentations: bool = False
    dataset: str = "breast_mri"
    resume_epoch: int = None

    # EXPERIMENTAL/UNTESTED: classifier-free class guidance and image translation
    class_conditional: bool = False
    cfg_p_uncond: float = 0.2 # p_uncond in classifier-free guidance paper
    cfg_weight: float = 0.3 # w in the paper
    trans_noise_level: float = 0.5 # ratio of time step t to noise trans_start_images to total T before denoising in translation. e.g. value of 0.5 means t = 500 for default T = 1000.
    use_cfg_for_eval_conditioning: bool = True  # whether to use classifier-free guidance for or just naive class conditioning for main sampling loop
    cfg_maskguidance_condmodel_only: bool = True  # if using mask guidance AND cfg, only give mask to conditional network
    # ^ this is because giving mask to both uncond and cond model make class guidance not work 
    # (see "Classifier-free guidance resolution weighting." in ControlNet paper)


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader, lr_scheduler, device='cuda'):
    global_step = 0
    
    # logging
    run_name = '{}-{}-{}'.format(config.model_type.lower(), config.dataset, config.image_size)
    if config.segmentation_guided:
        run_name += "-segguided"
    writer = SummaryWriter(comment=run_name)

    # For loading segs to condition on:
    eval_dataloader = iter(eval_dataloader)

    start_epoch = 0
    if config.resume_epoch is not None:
        start_epoch = config.resume_epoch

    for epoch in range(start_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()

        for step, batch in enumerate(train_dataloader):
            # Get both master and defect images
            master_images = batch['master_images'].to(device)
            defect_images = batch['defect_images'].to(device)

            # Create valid mask (where defect_images differ from master_images)
            valid_mask = (defect_images != master_images).float()

            # Sample noise and timesteps
            noise = torch.randn(defect_images.shape).to(device)
            bs = defect_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

            # Add noise to the defect images
            noisy_defects = noise_scheduler.add_noise(defect_images, noise, timesteps)

            # Prepare model input by concatenating noisy defects with master images
            model_input = torch.cat([noisy_defects, master_images], dim=1)

            # Get model prediction
            noise_pred = model(model_input, timesteps, return_dict=False)[0]

            # Calculate loss with valid mask
            loss = F.mse_loss(noise_pred * valid_mask, noise * valid_mask)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            writer.add_scalar("loss", loss.detach().item(), global_step)
            
            progress_bar.set_postfix(**logs)
            global_step += 1

        # After each epoch you optionally sample some demo images and save the model
        if epoch == 0 or (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
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

            # Get some master images for conditioning
            eval_batch = next(eval_dataloader)
            master_images_eval = eval_batch['master_images'].to(device)

            # Generate samples
            evaluate(config, epoch, pipeline, master_images_eval)

        if (epoch + 1) % config.save_model_epochs == 0:
            pipeline.save_pretrained(config.output_dir)

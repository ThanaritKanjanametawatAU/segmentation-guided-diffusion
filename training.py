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
import json

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


def save_model_checkpoint(config, epoch, pipeline, optimizer, lr_scheduler, scaler=None):
    """Helper function to save model checkpoint and training state"""
    
    # Create output directory if it doesn't exist
    checkpoint_dir = os.path.join(config.output_dir, "unets", f"checkpoint_{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the model using pipeline's save_pretrained
    pipeline.save_pretrained(checkpoint_dir)
    
  
    # Save Training Config
    config_path = os.path.join(checkpoint_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=4)


def load_checkpoint(config, optimizer, lr_scheduler, scaler=None):
    """Helper function to load model checkpoint and training state"""
    
    checkpoint_dir = os.path.join(config.output_dir, 'unets', f'checkpoint_{config.resume_epoch}')
    unet_path = os.path.join(checkpoint_dir, 'unet')
    optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pt')
    training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')
    
    if not os.path.exists(unet_path):
        raise FileNotFoundError(f"No checkpoint found at {unet_path}")
        
    # Load optimizer and training state if they exist
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path)
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        lr_scheduler.load_state_dict(training_state['lr_scheduler_state_dict'])
        
        if scaler is not None and 'scaler_state_dict' in training_state:
            scaler.load_state_dict(training_state['scaler_state_dict'])
            
        return training_state['epoch']
    
    # If no training state found, just return the resume epoch
    return config.resume_epoch

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, eval_dataloader, lr_scheduler, device='cuda',):
    global_step = 0
    
    # for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision == 'fp16')
    
    # Logging setup
    run_name = f"{config.model_type.lower()}-{config.dataset}-{config.image_size}"
    if config.segmentation_guided:
        run_name += "-segguided"
    writer = SummaryWriter(comment=run_name)

    # Evaluation data handling
    eval_dataloader_original = eval_dataloader
    eval_iter = iter(eval_dataloader)


    # Load checkpoint if resuming training
    if config.resume_epoch is not None:
        start_epoch = load_checkpoint(config, optimizer, lr_scheduler, scaler) + 1
    else:
        start_epoch = 1


    # Training Loop 
    for epoch in range(start_epoch, config.num_epochs+1):
        print("training at epoch {} / {}".format(epoch, config.num_epochs) + "\n")
        model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)  



        # Each Steps
        for step, batch in enumerate(train_dataloader):
            master_images = batch['master_images'].to(device, non_blocking=True)
            defect_images = batch['defect_images'].to(device, non_blocking=True)
            
            # Calculate the difference mask
            difference_mask = (defect_images != master_images).float()
            

            # Sample noise and timesteps
            noise = torch.randn_like(defect_images, device=device)
            bs = defect_images.shape[0]

            # Default Max timesteps: 1000
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device)

            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=config.mixed_precision == 'fp16'):
                # Add noise to defect images
                noisy_defects = noise_scheduler.add_noise(defect_images, noise, timesteps)
                
                # Prepare model input (6 channels: 3 for noisy defects, 3 for master images)
                model_input = torch.cat([noisy_defects, master_images], dim=1)
                
                
                # Get model prediction
                noise_pred = model(model_input, timesteps, return_dict=False)[0]
                
                # Calculate loss
                loss = F.mse_loss(noise_pred * difference_mask, noise * difference_mask)
                loss = loss / config.gradient_accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights if gradient accumulation complete
            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Logging
                total_loss += loss.detach().item()
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step
                }
                writer.add_scalar("train/loss", loss.detach().item(), global_step)
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
                
                progress_bar.set_postfix(**logs)
                global_step += 1

            progress_bar.update(1)





        # Evaluation and checkpointing
        if epoch % config.save_image_epochs == 0 or epoch == config.num_epochs:
            print("evaluating at epoch {}".format(epoch) + "\n")
            model.eval()
            
            pipeline_class = PCBDiffusionPipeline
            pipeline = pipeline_class(
                unet=model.module,
                scheduler=noise_scheduler,
                external_config=config
            )

            # Get evaluation batch
            try:
                eval_batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_dataloader_original)
                eval_batch = next(eval_iter)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.mixed_precision == 'fp16'):
                master_images_eval = eval_batch['master_images'].to(device)
                evaluate(config, epoch, pipeline, master_images_eval)

            # Log average loss for epoch
            avg_loss = total_loss / len(train_dataloader)
            writer.add_scalar("train/epoch_loss", avg_loss, epoch)




        # Save checkpoints
        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            print("saving model checkpoint at epoch {}".format(epoch) + "\n")
            save_model_checkpoint(
                config=config,
                epoch=epoch,
                pipeline=pipeline,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler if config.mixed_precision == 'fp16' else None
            )
            

    writer.close()

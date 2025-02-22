import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from argparse import ArgumentParser
import datasets
from datasets import load_dataset
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

class PCBLatentInpainting:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize VAE for encoding pixel images to latent space
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Initialize text encoder
        # self.text_encoder = transformers.BertModel.from_pretrained(
        #     "bert-base-uncased",
        #     torch_dtype=torch.float16
        # ).to(self.device)

        self.text_encoder = transformers.CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Freeze VAE and text encoder weights
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Initialize UNet for latent space inpainting
        self.unet = UNet2DConditionModel(
            sample_size=args.latent_size,
            in_channels=9,  # 4 for latent image (4*64*64) + 4 for latent conditioning image (4*64*64) + 1 for mask
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=512,
        ).to(self.device)
        
        # DDPM scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear"
        )
        
        # Tokenizer
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        # Optimizer and gradient scaler for mixed precision
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01
        )
        self.scaler = GradScaler()


    def encode_text(self, text_input):
        with torch.no_grad():
            text_output = self.text_encoder(**text_input)[0]
        return text_output


    def encode_images(self, images):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents


    def train_step(self, batch):
        # Autocast for mixed precision
        with autocast(enabled=True):
            # Get text embeddings
            text_embeddings = self.encode_text({
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
            })

            # Encode images to latent space
            master_latents = self.encode_images(batch['master_images'].to(self.device))
            defect_latents = self.encode_images(batch['defect_images'].to(self.device))
            
            # Process mask - downsample to match latent dimensions
            mask = F.interpolate(
                batch['masks'].to(self.device),
                size=(self.args.latent_size, self.args.latent_size),
                mode='nearest'
            )
            
            # Sample noise and timesteps
            noise = torch.randn_like(defect_latents)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (defect_latents.shape[0],), device=self.device
            ).long()

            # Add noise to defect latents
            noisy_latents = self.noise_scheduler.add_noise(defect_latents, noise, timesteps)
            
            # Concatenate model inputs
            model_input = torch.cat([noisy_latents, master_latents, mask], dim=1)

            # Get model prediction
            noise_pred = self.unet(
                model_input,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample

            # Calculate masked loss
            loss = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)

        return loss

    def train(self, train_dataloader):
        num_update_steps_per_epoch = len(train_dataloader)
        total_steps = self.args.num_epochs * num_update_steps_per_epoch
        
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=int(total_steps * 0.05),
            num_training_steps=total_steps
        )

        progress_bar = tqdm(range(total_steps))
        
        for epoch in range(self.args.num_epochs):
            for batch in train_dataloader:
                self.optimizer.zero_grad()
                
                loss = self.train_step(batch)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                lr_scheduler.step()
                
                progress_bar.update(1)
                progress_bar.set_description(f"Loss: {loss.item():.4f}")

            # Save checkpoint
            if (epoch + 1) % self.args.save_epochs == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save UNet
        self.unet.save_pretrained(checkpoint_dir / "unet")
        
        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt"
        )

def create_data_loader(args):
    """Load and preprocess the PCB dataset"""
    tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    dataset = load_dataset(args.dataset)["train"]
    dataset = dataset.filter(lambda x: x["description"] != "nothing")

    # Write summary to file
    output_path = Path(args.output_dir) / "dataset_summary.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("Dataset Summary:\n")
        f.write(f"Total number of examples: {len(dataset)}\n")
        unique_descriptions = dataset.unique("description")
        for desc in unique_descriptions:
            count = sum(1 for d in dataset["description"] if d == desc)
            f.write(f"{desc}: {count}\n")

    # Convert columns to images
    dataset = dataset.cast_column("master_image", datasets.Image())
    dataset = dataset.cast_column("defect_image", datasets.Image())
    dataset = dataset.cast_column("mask", datasets.Image())

    # Create separate transforms for images and masks
    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    def preprocess(examples):
        # Process images with image transform
        master_images = [image_transform(image.convert("RGB")) for image in examples["master_image"]]
        defect_images = [image_transform(image.convert("RGB")) for image in examples["defect_image"]] 
        
        # Process masks with mask transform
        masks = [mask_transform(image.convert("L")) for image in examples["mask"]]

        # Tokenize descriptions
        text_inputs = examples["description"]
        encoded = tokenizer(
            text_inputs,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "master_images": torch.stack(master_images),
            "defect_images": torch.stack(defect_images),
            "masks": torch.stack(masks),
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
        }

    dataset.set_transform(preprocess)
    return DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

def main(args):
    # Create model and trainer
    model = PCBLatentInpainting(args)
    
    # Create data loader
    train_dataloader = create_data_loader(args)
    
    # Train model
    model.train(train_dataloader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Thanarit/PCB-v3")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="pcb_latent_inpainting")
    args = parser.parse_args()
    
    main(args)
import torch
from torch import nn
import transformers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from datasets import load_dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from argparse import ArgumentParser
from pathlib import Path
from torchvision.utils import save_image
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont

class PCBLatentInpaintingInference:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        
        # Initialize models with FP16 for faster inference
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.text_encoder = transformers.CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Load UNet from checkpoint
        self.unet = UNet2DConditionModel.from_pretrained(
            self.checkpoint_path / "unet",
            torch_dtype=torch.float16
        ).to(self.device)

        # Initialize tokenizer
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Set models to eval mode
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()
        
        # Initialize scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear"
        )

    @torch.no_grad()
    def generate(self, prompt, master_image, mask, num_inference_steps=50):
        """
        Generate defective PCB image based on prompt, master image and mask.
        
        Args:
            prompt (str): Text description of the defect
            master_image (torch.Tensor): [1, 3, H, W] master PCB image
            mask (torch.Tensor): [1, 1, H, W] mask tensor
            num_inference_steps (int): Number of denoising steps
        """
        with autocast(enabled=True):
            # Encode text
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            text_embeddings = self.text_encoder(**text_input)[0]
            
            # Encode master image to latent
            master_latents = self.vae.encode(master_image.to(self.device)).latent_dist.sample()
            master_latents = master_latents * self.vae.config.scaling_factor
            
            # Process mask - downsample to match latent dimensions
            mask = transforms.functional.resize(
                mask, 
                size=[master_latents.shape[2], master_latents.shape[3]],  # [H, W] of latents
                interpolation=transforms.InterpolationMode.NEAREST
            ).to(device=self.device, dtype=torch.float16)
            
            # Start from random noise in masked regions
            latents = torch.randn_like(master_latents)
            inv_mask = 1 - mask
            
            # Initialize with original content in unmasked regions
            latents = (latents * mask) + (master_latents * inv_mask)
            
            # Set timesteps
            self.noise_scheduler.set_timesteps(num_inference_steps)
            
            # Denoising loop
            for t in self.noise_scheduler.timesteps:
                # Prepare input
                model_input = torch.cat([
                    latents,
                    master_latents, 
                    mask
                ], dim=1)
                
                # Get model prediction
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # Scheduler step
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents
                ).prev_sample
                
                # Reapply mask to preserve original content
                latents = (latents * mask) + (master_latents * inv_mask)
            
            # Decode latents to image
            images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            
            # Denormalize
            images = (images / 2 + 0.5).clamp(0, 1)
            
            return images
        
# Create comparison grid with proper formatting and labels
def create_comparison_image(original, generated, pair_id, prompt):
    # Set up dimensions
    img_size = 512
    spacing = 20  # Spacing between images
    label_height = 120  # Height for labels
    title_height = 80  # Height for title
    total_height = img_size + label_height + title_height
    total_width = img_size * 2 + spacing
    
    # Create white background
    background = torch.ones((3, total_height, total_width))
    
    # Place images
    # Original on left
    background[:, title_height:title_height+img_size, :img_size] = original.squeeze(0)
    # Generated on right
    background[:, title_height:title_height+img_size, -img_size:] = generated.squeeze(0)
    
    # Convert to PIL for adding text
    pil_img = transforms.ToPILImage()(background)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        # Increased font sizes
        font = ImageFont.truetype("arial.ttf", 24) 
        title_font = ImageFont.truetype("arial.ttf", 28) 
    except:
        # Fallback to default with size scaling
        font = ImageFont.load_default(size=24)
        title_font = ImageFont.load_default(size=28)

    # Adjust text positions for larger fonts
    # Add title
    title = f"Pair {pair_id}"
    title_w = draw.textlength(title, font=title_font)
    draw.text(
        ((total_width - title_w) // 2, 10),  # Was 5
        title,
        fill="black",
        font=title_font
    )

    # Add prompt
    prompt_display = f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}"
    prompt_w = draw.textlength(prompt_display, font=font)
    draw.text(
        ((total_width - prompt_w) // 2, 40),  
        prompt_display,
        fill="black",
        font=font
    )

    # Add labels below images
    # Original image label
    orig_label = "Original Defect"
    orig_w = draw.textlength(orig_label, font=font)
    draw.text(
        ((img_size - orig_w) // 2, total_height - label_height + 15),  # Was +10
        orig_label,
        fill="black",
        font=font
    )

    # Generated image label
    gen_label = "Generated Defect"
    gen_w = draw.textlength(gen_label, font=font)
    draw.text(
        (img_size + spacing + (img_size - gen_w) // 2, total_height - label_height + 15),  # Was +10
        gen_label,
        fill="black",
        font=font
    )

    return pil_img

def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="inference_outputs")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_examples", type=int, default=10)
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = PCBLatentInpaintingInference(args.checkpoint_path)
    
    # Load first example from dataset
    dataset = load_dataset("Thanarit/PCB-v3")["train"]
    
    # Process first N examples
    for idx in range(args.num_examples):
        # Filter out rows with description is "nothing"
        if dataset[idx]["description"] == "nothing":
            continue

        example = dataset[idx]
        pair_id = example.get("pair_id", f"{idx:04d}")
        
        # Create output directory for this pair
        pair_dir = os.path.join(args.output_dir, f"Pair_{pair_id}-{example['description']}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Separate transforms for image and mask
        transform_image = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        transform_mask = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        transform_save = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
        # Process images
        master_image = transform_image(example["master_image"].convert("RGB")).unsqueeze(0)
        mask = transform_mask(example["mask"].convert("L")).unsqueeze(0)
        prompt = example["description"]
        defect_image = transform_image(example["defect_image"].convert("RGB")).unsqueeze(0) #Original
        
        print(f"Generating with prompt: {prompt}")
        
        # Generate image
        generated_image = model.generate(
            prompt=prompt,
            master_image=master_image,
            mask=mask,
            num_inference_steps=args.num_inference_steps
        )

        master_image_save = transform_save(example["master_image"].convert("RGB")).unsqueeze(0)
        defect_image_save = transform_save(example["defect_image"].convert("RGB")).unsqueeze(0)
        
        # Save individual components
        save_path = lambda name: os.path.join(pair_dir, name)
        save_image(master_image_save, save_path("master_image.png"))
        save_image(defect_image_save, save_path("original_defect.png"))
        save_image(mask, save_path("mask.png"))
        save_image(generated_image, save_path("generated.png"))
        
        # Create and save comparison image
        comparison_img = create_comparison_image(
            defect_image_save,
            generated_image,
            pair_id,
            prompt
        )
        comparison_img.save(save_path(f"comparison.png"))
        
        # Save prompt
        # with open(save_path("prompt.txt"), "w") as f:
        #     f.write(prompt)
            
        print(f"Processed Pair {pair_id}\n")

    print(f"\nFinished processing {args.num_examples} examples in {args.output_dir}")

if __name__ == "__main__":
    main()
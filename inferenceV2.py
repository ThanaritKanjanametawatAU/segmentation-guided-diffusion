import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from diffusers import UNet2DModel, DiffusionPipeline, ImagePipelineOutput, DDPMScheduler
import os
from typing import Optional, Tuple, Union
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_comparison_image(master_image, defect_image, generated_image):
    """Create a side-by-side comparison image"""
    # Create figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    
    # Plot master image
    ax1.imshow(master_image)
    ax1.set_title('Master Image')
    ax1.axis('off')
    
    # Plot defect image
    ax2.imshow(defect_image)
    ax2.set_title('Actual Defect')
    ax2.axis('off')
    
    # Plot generated image
    ax3.imshow(generated_image)
    ax3.set_title('Generated Defect')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig


def batch_inference(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    num_inference_steps: int = 1000,
    target_size: int = 64,  # Changed default to match model
    device: str = 'cuda',
    batch_size: int = 4
):
    """
    Perform batch inference on a HuggingFace dataset
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "defects"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
    
    # Load model and get configuration
    model = UNet2DModel.from_pretrained(model_path).to(device)
    
    # Get model's expected input size from config
    model_size = model.config.sample_size
    print(f"Model expected input size: {model_size}")
    
    # Setup pipeline

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipeline = PCBDiffusionPipeline(
        unet=model,
        scheduler=scheduler,
        external_config=None
    ).to(device)
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="test")
    
    # Setup image transform using model's expected size
    transform = transforms.Compose([
        transforms.Resize((model_size, model_size)), # Use model's size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Process dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        
        # Process master images
        master_images = []
        for img in batch['master_image']:
            master_tensor = transform(img).unsqueeze(0)
            master_images.append(master_tensor)
        
        master_batch = torch.cat(master_images, dim=0).to(device)
        
        # Generate defect images
        generated_images = pipeline(
            batch_size=len(master_images),
            master_image=master_batch,
            num_inference_steps=num_inference_steps,
        ).images
        
        # Save individual images and comparisons
        for j, (master_img, defect_img, gen_img, pair_id) in enumerate(zip(batch['master_image'], 
                                                              batch['defect_image'],
                                                              generated_images,
                                                              batch['pair_id'])):
            # Save generated defect
            defect_path = os.path.join(output_dir, "defects", f"{pair_id}_defect.png")
            gen_img.save(defect_path)
            
            # Create and save comparison
            fig = create_comparison_image(master_img, defect_img, gen_img)
            comparison_path = os.path.join(output_dir, "comparisons", f"{pair_id}_comparison.png")
            fig.savefig(comparison_path)
            plt.close(fig)

class PCBDiffusionPipeline(DiffusionPipeline):
    """Pipeline for PCB defect generation using paired normal/defect images"""
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, external_config=None):
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler
        )
        self._external_config = external_config

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
        # Get model's expected size
        model_size = self.unet.config.sample_size
        
        # Start from random noise with correct size
        if self.device.type == "mps":
            x_t = torch.randn((batch_size, 3, model_size, model_size),
                             generator=generator)
            x_t = x_t.to(self.device)
        else:
            x_t = torch.randn((batch_size, 3, model_size, model_size),
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

if __name__ == "__main__":
    # Example usage
    model_path = "ddim-Thanarit_PCB-v2-64/unets/checkpoint_5000/unet"
    output_dir = "ddim-Thanarit_PCB-v2-64/inference_test_output"
    
    batch_inference(
        model_path=model_path,
        dataset_name="Thanarit/PCB-v2",
        output_dir=output_dir,
        num_inference_steps=1000,
        device='cuda',
        batch_size=128
    )
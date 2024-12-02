import torch
from PIL import Image
from torchvision import transforms
from diffusers import UNet2DModel, DiffusionPipeline, ImagePipelineOutput
import os
from typing import Optional, Tuple, Union

# Add these functions at the top of the file
def load_image(image_path, target_size=128):
    """Load and preprocess an image from a given path"""
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def load_model(model_path, device='cuda'):
    """Load UNet model from a given path"""
    model = UNet2DModel.from_pretrained(model_path)
    model = model.to(device)
    return model

# Add this function for simple inference
def simple_inference(
    model_path: str,
    image_path: str,
    output_path: str,
    num_inference_steps: int = 1000,
    target_size: int = 128,
    device: str = 'cuda'
):
    """
    Perform simple inference with a saved model on a single image
    
    Args:
        model_path: Path to the saved UNet model
        image_path: Path to the master image
        output_path: Path to save the generated image
        num_inference_steps: Number of denoising steps
        device: Device to run inference on ('cuda' or 'cpu')
    """
    # Load model
    model = load_model(model_path, device)
    
    # Load and preprocess image
    master_image = load_image(image_path, target_size)
    master_image = master_image.to(device)
    
    # Setup pipeline
    from diffusers import DDPMScheduler  # Add appropriate import
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    pipeline = PCBDiffusionPipeline(
        unet=model,
        scheduler=scheduler,
        external_config=None  # Not needed for inference
    )
    pipeline = pipeline.to(device)
    
    # Generate image
    output_image = pipeline(
        batch_size=1,
        master_image=master_image,
        num_inference_steps=num_inference_steps,
    ).images[0]
    
    # Save the generated image
    output_image.save(output_path)
    return output_image

class PCBDiffusionPipeline(DiffusionPipeline):
    """Pipeline for PCB defect generation using paired normal/defect images"""
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, external_config=None):
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler
        )
        # Store external_config as an attribute but don't register it as a module
        self._external_config = external_config

    @property
    def external_config(self):
        return self._external_config

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

def evaluate_sample_many(
    sample_size,
    config,
    model,
    noise_scheduler,
    eval_dataloader,
    device='cuda'
    ):
    
    # Get a batch of master images to condition on
    eval_batch = next(iter(eval_dataloader))
    master_images = eval_batch['master_images'].to(device)
    
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

    sample_dir = os.path.join(config.output_dir, f"samples_many_{sample_size}")
    os.makedirs(sample_dir, exist_ok=True)

    num_sampled = 0
    while num_sampled < sample_size:
        # Generate images conditioned on the master image
        images = pipeline(
            batch_size=min(config.eval_batch_size, sample_size - num_sampled),
            master_image=master_images[:1].repeat(config.eval_batch_size, 1, 1, 1),  # Repeat the first master image
        ).images

        # Save each image in the batch
        for i, img in enumerate(images):
            img_fname = f"{sample_dir}/{num_sampled + i:04d}.png"
            img.save(img_fname)

        num_sampled += len(images)
        print(f"sampled {num_sampled}/{sample_size}")


# Example usage
model_path = "ddim-Thanarit_PCB-128/unets/checkpoint_14/unet"
image_path = "Data/test/ts_1.jpg"
output_path = "Data/test_output/ts_1_output.png"

generated_image = simple_inference(
    model_path=model_path,
    image_path=image_path,
    output_path=output_path,
    num_inference_steps=1000,  # You can adjust this
    target_size=128,
    device='cuda'  
)
from PIL import Image
import torch
import torchvision.transforms as transforms

def make_grid(images, rows, cols):
    if isinstance(images[0], torch.Tensor):
        # Handle torch tensors
        # Denormalize if needed
        images = [(img.cpu() * 0.5 + 0.5).clamp(0, 1) for img in images]
        # Convert to PIL images
        to_pil = transforms.ToPILImage()
        images = [to_pil(img) for img in images]
    
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid
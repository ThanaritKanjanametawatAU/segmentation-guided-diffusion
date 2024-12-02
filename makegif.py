import os
from PIL import Image, ImageDraw, ImageFont
import glob
import re
import cv2
import numpy as np

def create_gif_and_mp4(source_folder, output_base, size=(640, 640), duration=100, loop=0):
    """
    Create a GIF and MP4 from all images in the source folder, styled like a plot.
    
    :param source_folder: Path to the folder containing images
    :param output_base: Base path for output files (without extension)
    :param size: Tuple of (width, height) for resizing images
    :param duration: Duration of each frame in milliseconds
    :param loop: Number of times to loop the GIF (0 means infinite)
    """
    # Get list of all image files in the source folder
    image_files = glob.glob(os.path.join(source_folder, "*.[pj][np][g]"))
    
    if not image_files:
        print(f"No image files found in {source_folder}")
        return

    def extract_number(filename):
        match = re.search(r'\d+', os.path.basename(filename))
        return int(match.group()) if match else 0

    image_files.sort(key=extract_number)

    # Calculate padding for plot-like appearance
    padding = 40
    plot_size = (size[0], size[1] + padding * 2)  # Add padding for title
    
    images = []
    frames_for_video = []
    
    try:
        # Try to load Arial font, fall back to default if not available
        font = ImageFont.truetype("arial.ttf", size[0]//20)
        title_font = ImageFont.truetype("arial.ttf", size[0]//15)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    for file in image_files:
        try:
            # Create a new white background image with padding
            plot_img = Image.new('RGB', plot_size, 'white')
            
            # Open and resize the sample image
            img = Image.open(file)
            img_resized = img.resize(size, Image.LANCZOS)
            
            # Paste the sample image onto the white background
            plot_img.paste(img_resized, (0, padding))
            
            draw = ImageDraw.Draw(plot_img)
            
            # Add title
            title = source_folder.split("/")[0]
            title_width = draw.textlength(title, font=title_font)
            title_position = ((plot_size[0] - title_width) // 2, 5)
            draw.text(title_position, title, font=title_font, fill='black')
            
            # Add epoch number
            epoch_num = extract_number(file)
            text = str(epoch_num)
            text_width = draw.textlength(text, font=font)
            # Position text in the middle below the title
            text_position = ((plot_size[0] - text_width) // 2, padding // 2)  # Center horizontally, below title
            draw.text(text_position, text, font=font, fill='black')
            
            images.append(plot_img)
            
            # Convert to numpy array for video
            frames_for_video.append(cv2.cvtColor(np.array(plot_img), cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not images:
        print("No valid images found to create GIF/MP4")
        return

    # Save GIF
    gif_path = f"{output_base}.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )
    print(f"GIF created successfully: {gif_path}")


if __name__ == "__main__":
    source_folder = "ddim-Thanarit_PCB-256/samples"
    output_base = "results/PCB-256/sample"
    
    create_gif_and_mp4(source_folder, output_base, size=(256, 256), duration=300, loop=0)


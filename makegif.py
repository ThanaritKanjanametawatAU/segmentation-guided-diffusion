import os
from PIL import Image
import glob
import re

def create_gif(source_folder, output_file, size=(640, 640), duration=100, loop=0):
    """
    Create a GIF from all images in the source folder, sorted by numbers in filenames.
    All images are resized to the specified size.
    
    :param source_folder: Path to the folder containing images
    :param output_file: Path and filename for the output GIF
    :param size: Tuple of (width, height) for resizing images
    :param duration: Duration of each frame in milliseconds
    :param loop: Number of times to loop the GIF (0 means infinite)
    """
    # Get list of all image files in the source folder
    image_files = glob.glob(os.path.join(source_folder, "*.[pj][np][g]"))
    
    if not image_files:
        print(f"No image files found in {source_folder}")
        return

    # Sort image files based on numbers in filenames
    def extract_number(filename):
        match = re.search(r'\d+', os.path.basename(filename))
        return int(match.group()) if match else 0

    image_files.sort(key=extract_number)


    # Open all images and resize them
    images = []
    for file in image_files:
        try:
            img = Image.open(file)
            img_resized = img.resize(size, Image.LANCZOS)
            images.append(img_resized)
        except Exception as e:
            print(f"Error opening or resizing {file}: {e}")

    if not images:
        print("No valid images found to create GIF")
        return

    # Save the first image as GIF and append the rest
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )
    print(f"GIF created successfully: {output_file}")

if __name__ == "__main__":
    source_folder = "ddim-printed_circuit_board-640/samples"
    output_file = "sample.gif"
    
    create_gif(source_folder, output_file, size=(640, 640), duration=300, loop=0)


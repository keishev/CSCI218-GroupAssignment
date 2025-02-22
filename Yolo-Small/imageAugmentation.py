import os
from PIL import Image

def rotate_image(input_path, output_path):
    """
    Rotate an image by 90 degrees clockwise and save it to the output path
    """
    with Image.open(input_path) as img:
        # Rotate 90 degrees clockwise (PIL uses counter-clockwise angles, so 270)
        rotated_img = img.rotate(270, expand=True)
        rotated_img.save(output_path)

# Set your directories
dirInitial = "Small-Dataset/train/down"  # Directory with original images
dirResult = "Small-Dataset/train/left"  # Directory to save rotated images

# Create output directory if it doesn't exist
os.makedirs(dirResult, exist_ok=True)

# Supported image formats
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')

print(f"Processing images in {dirInitial}...")
# Process all images in the directory
for filename in os.listdir(dirInitial):
    if filename.lower().endswith(image_extensions):
        input_path = os.path.join(dirInitial, filename)
        output_path = os.path.join(dirResult, filename)

        try:
            rotate_image(input_path, output_path)
            # print(f"Successfully processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
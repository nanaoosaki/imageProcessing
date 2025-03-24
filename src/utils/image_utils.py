"""
Image Utilities Module

This module provides utility functions for image processing,
including loading, saving, and basic transformations.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_image(image_path):
    """
    Load an image from a file path.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        tuple: (PIL Image, OpenCV Image) - Both representations of the loaded image.
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load with PIL
    pil_image = Image.open(image_path)
    
    # Load with OpenCV
    cv_image = cv2.imread(image_path)
    
    # Check if OpenCV read the image correctly
    if cv_image is None:
        # Try converting from PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return pil_image, cv_image


def save_image(image, output_path, format="PIL"):
    """
    Save an image to a file.
    
    Args:
        image: PIL Image or OpenCV Image to save.
        output_path (str): Path where the image will be saved.
        format (str): Format of the input image ('PIL' or 'CV').
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format.upper() == "PIL":
        # Save PIL Image
        image.save(output_path)
    elif format.upper() == "CV":
        # Save OpenCV Image
        cv2.imwrite(output_path, image)
    else:
        raise ValueError("Format must be 'PIL' or 'CV'")


def create_sample_image(output_path, width=800, height=600, text="Sample Text", font_size=30):
    """
    Create a sample image with text for testing.
    
    Args:
        output_path (str): Path where the image will be saved.
        width (int): Width of the image.
        height (int): Height of the image.
        text (str): Text to add to the image.
        font_size (int): Font size for the text.
    
    Returns:
        PIL.Image: The created image.
    """
    # Create blank image
    image = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    # Try to use a standard font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fall back to default font
        font = ImageFont.load_default()
    
    # Add some shapes
    draw.rectangle([(50, 50), (width-50, height-50)], outline=(200, 200, 200), width=2)
    draw.ellipse([(100, 100), (300, 200)], fill=(230, 230, 230), outline=(200, 0, 0))
    
    # Add text in multiple locations
    text_positions = [
        (width // 2, 100),
        (width // 4, height // 2),
        (width * 3 // 4, height // 2),
        (width // 2, height - 100)
    ]
    
    for pos in text_positions:
        # Get text size
        text_width = font_size * len(text) // 2
        
        # Draw text
        draw.text(
            (pos[0] - text_width // 2, pos[1] - font_size // 2),
            text,
            font=font,
            fill=(0, 0, 0)
        )
    
    # Save the image
    save_image(image, output_path)
    
    return image


if __name__ == "__main__":
    # Create a sample image if run directly
    sample_path = os.path.join("samples", "sample_text.png")
    create_sample_image(sample_path, text="This is a sample text for OCR testing.") 
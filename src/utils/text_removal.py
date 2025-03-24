"""
Text Removal Utilities Module

This module provides functions for removing text from images
using detected text regions.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Import our text detection utilities
from .text_detection import detect_text_tesseract, detect_text_easyocr


def remove_text_inpaint(image, regions, radius=10):
    """
    Remove text from an image using OpenCV's inpainting algorithms.
    
    Args:
        image: PIL Image or OpenCV Image.
        regions (list): List of detected text regions from text detection.
        radius (int): Radius of neighborhood for inpainting.
        
    Returns:
        numpy.ndarray: OpenCV image with text removed.
    """
    # If image is in PIL format, convert to OpenCV format
    if not isinstance(image, np.ndarray):
        # Convert to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create a copy of the image
    result = image.copy()
    
    # Create a mask for the text regions (white text on black background)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Fill the mask with the text regions
    for region in regions:
        box = region['box']
        # Add some padding around the text
        padding = 2
        x1, y1, x2, y2 = box
        cv2.rectangle(mask, (x1-padding, y1-padding), (x2+padding, y2+padding), 255, -1)
    
    # Use inpainting to remove the text
    # We'll try both algorithms and return the better one
    result_ns = cv2.inpaint(result, mask, radius, cv2.INPAINT_NS)  # Navier-Stokes
    result_telea = cv2.inpaint(result, mask, radius, cv2.INPAINT_TELEA)  # Alexandru Telea
    
    # For now, return the Navier-Stokes result (often provides smoother results)
    # In a more advanced version, we could evaluate both and choose the best
    return result_ns


def remove_text_content_aware(image, regions, sample_radius=5):
    """
    Remove text by using content-aware fill based on surrounding pixels.
    Works well for solid or simple gradient backgrounds.
    
    Args:
        image: PIL Image or OpenCV Image.
        regions (list): List of detected text regions from text detection.
        sample_radius (int): Radius to sample surrounding pixels.
        
    Returns:
        PIL.Image: Image with text removed.
    """
    # If image is in OpenCV format, convert to PIL
    if isinstance(image, np.ndarray):
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = image
    
    # Create a copy of the image
    result = pil_image.copy()
    draw = ImageDraw.Draw(result)
    
    width, height = pil_image.size
    
    # Process each text region
    for region in regions:
        box = region['box']
        x1, y1, x2, y2 = box
        
        # Add some padding
        padding = 2
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Sample colors from above and below the text region
        top_samples = []
        bottom_samples = []
        
        # Sample above
        for x in range(x1, x2, 5):
            for y_offset in range(1, sample_radius + 1):
                y = y1 - y_offset
                if 0 <= y < height:
                    top_samples.append(pil_image.getpixel((x, y)))
        
        # Sample below
        for x in range(x1, x2, 5):
            for y_offset in range(1, sample_radius + 1):
                y = y2 + y_offset
                if 0 <= y < height:
                    bottom_samples.append(pil_image.getpixel((x, y)))
        
        # If we have samples, interpolate between them to fill the text region
        if top_samples and bottom_samples:
            for y in range(y1, y2 + 1):
                # Calculate interpolation factor (0 at top, 1 at bottom)
                t = (y - y1) / (y2 - y1) if y2 > y1 else 0
                
                for x in range(x1, x2 + 1):
                    # Get average color from top and bottom samples
                    if top_samples and bottom_samples:
                        top_avg = np.mean(top_samples, axis=0)
                        bottom_avg = np.mean(bottom_samples, axis=0)
                        
                        # Linear interpolation between top and bottom
                        color = tuple(int((1-t) * top_avg[i] + t * bottom_avg[i]) for i in range(3))
                        
                        # Set the pixel color
                        draw.point((x, y), fill=color)
        else:
            # If no samples, use a simple surrounding average
            # Get all boundary pixels
            boundary_pixels = []
            
            # Top and bottom boundaries
            for x in range(x1, x2):
                if y1 > 0:
                    boundary_pixels.append(pil_image.getpixel((x, y1 - 1)))
                if y2 < height - 1:
                    boundary_pixels.append(pil_image.getpixel((x, y2 + 1)))
            
            # Left and right boundaries
            for y in range(y1, y2):
                if x1 > 0:
                    boundary_pixels.append(pil_image.getpixel((x1 - 1, y)))
                if x2 < width - 1:
                    boundary_pixels.append(pil_image.getpixel((x2 + 1, y)))
            
            if boundary_pixels:
                # Calculate average color
                avg_color = tuple(int(np.mean([p[i] for p in boundary_pixels])) for i in range(3))
                
                # Fill region with average color
                draw.rectangle([x1, y1, x2, y2], fill=avg_color)
    
    return result


def remove_text(image, method="inpaint", detection_method="easyocr", min_confidence=0.5, **kwargs):
    """
    Detect and remove text from an image using the specified method.
    
    Args:
        image: PIL Image or OpenCV Image.
        method (str): Text removal method - 'inpaint' or 'content_aware'.
        detection_method (str): Text detection method - 'tesseract' or 'easyocr'.
        min_confidence (float): Minimum confidence for text detection.
        **kwargs: Additional parameters for the removal method.
        
    Returns:
        PIL.Image or numpy.ndarray: Image with text removed.
    """
    # Detect text regions
    if detection_method.lower() == "tesseract":
        regions = detect_text_tesseract(image, min_confidence=min_confidence)
    else:  # Default to EasyOCR
        regions = detect_text_easyocr(image, min_confidence=min_confidence)
    
    print(f"Detected {len(regions)} text regions with {detection_method}")
    
    # Check if any text was detected
    if not regions:
        print("No text regions detected, returning original image")
        return image
    
    # Remove text using the specified method
    if method.lower() == "inpaint":
        radius = kwargs.get("radius", 10)
        return remove_text_inpaint(image, regions, radius=radius)
    elif method.lower() == "content_aware":
        sample_radius = kwargs.get("sample_radius", 5)
        return remove_text_content_aware(image, regions, sample_radius=sample_radius)
    else:
        raise ValueError(f"Unknown text removal method: {method}")


if __name__ == "__main__":
    import os
    from .image_utils import load_image, save_image
    from .file_io import INPUT_DIR, OUTPUT_DIR
    
    # Test on a sample image
    sample_path = os.path.join(INPUT_DIR, "sample_plain_text.png")
    
    if os.path.exists(sample_path):
        # Load the image
        pil_image, cv_image = load_image(sample_path)
        
        # Remove text using inpainting
        result_inpaint = remove_text(cv_image, method="inpaint")
        save_image(result_inpaint, os.path.join(OUTPUT_DIR, "result_inpaint.png"), format="CV")
        
        # Remove text using content-aware fill
        result_content_aware = remove_text(pil_image, method="content_aware")
        save_image(result_content_aware, os.path.join(OUTPUT_DIR, "result_content_aware.png"), format="PIL")
        
        print("Text removal complete. Check the output directory for results.")
    else:
        print(f"Sample image not found: {sample_path}")
        print("Run generate_samples.py first to create sample images.") 
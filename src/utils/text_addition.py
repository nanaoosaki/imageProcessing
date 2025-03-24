"""
Text Addition Utilities Module

This module provides functions for adding text to images
with various customization options.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def add_text_pil(
    image,
    text,
    position="center",
    font_path=None,
    font_size=30,
    color=(0, 0, 0),
    align="center",
    background=None,
    padding=5,
    opacity=1.0
):
    """
    Add text to an image using PIL.
    
    Args:
        image: PIL Image or OpenCV Image.
        text (str): Text to add to the image.
        position (tuple or str): Position (x, y) or predefined position ('center', 'top', 'bottom', etc.).
        font_path (str): Path to a .ttf font file.
        font_size (int): Font size.
        color (tuple): RGB color tuple.
        align (str): Text alignment ('left', 'center', 'right').
        background (tuple): RGBA background color for text. None for transparent.
        padding (int): Padding around text if background is used.
        opacity (float): Text opacity (0.0 to 1.0).
        
    Returns:
        PIL.Image: Image with added text.
    """
    # If image is in OpenCV format, convert to PIL
    if isinstance(image, np.ndarray):
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = image.copy()
    
    # Create a transparent overlay for the text
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Load font
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Try a standard font first
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
                print("Warning: Default font used as specified font was not found.")
    except Exception as e:
        print(f"Font error: {e}")
        font = ImageFont.load_default()
    
    # Calculate text size
    if hasattr(draw, 'textbbox'):
        # For newer PIL versions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        # For older PIL versions
        text_width, text_height = draw.textsize(text, font=font)
    
    # Determine position
    if isinstance(position, str):
        img_width, img_height = pil_image.size
        if position.lower() == "center":
            position = ((img_width - text_width) // 2, (img_height - text_height) // 2)
        elif position.lower() == "top":
            position = ((img_width - text_width) // 2, padding)
        elif position.lower() == "bottom":
            position = ((img_width - text_width) // 2, img_height - text_height - padding)
        elif position.lower() == "topleft":
            position = (padding, padding)
        elif position.lower() == "topright":
            position = (img_width - text_width - padding, padding)
        elif position.lower() == "bottomleft":
            position = (padding, img_height - text_height - padding)
        elif position.lower() == "bottomright":
            position = (img_width - text_width - padding, img_height - text_height - padding)
    
    # Adjust position based on alignment
    x, y = position
    if align.lower() == "center":
        x -= text_width // 2
    elif align.lower() == "right":
        x -= text_width
    
    # Draw text background if specified
    if background:
        bg_x = x - padding
        bg_y = y - padding
        bg_width = text_width + (padding * 2)
        bg_height = text_height + (padding * 2)
        
        # Ensure background has alpha value
        if len(background) == 3:
            background = (*background, 255)
        
        # Draw background rectangle
        draw.rectangle(
            [(bg_x, bg_y), (bg_x + bg_width, bg_y + bg_height)],
            fill=background
        )
    
    # Draw text
    # Ensure color has alpha value for opacity
    if len(color) == 3:
        color = (*color, int(255 * opacity))
    
    draw.text((x, y), text, font=font, fill=color)
    
    # Composite the overlay with the original image
    if pil_image.mode != 'RGBA':
        pil_image = pil_image.convert('RGBA')
    
    result = Image.alpha_composite(pil_image, overlay)
    
    # Convert back to original mode if needed
    if image.mode != 'RGBA':
        result = result.convert(image.mode)
    
    return result


def add_text_cv2(
    image,
    text,
    position="center",
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    color=(0, 0, 0),
    thickness=2,
    line_type=cv2.LINE_AA,
    background=None,
    padding=5,
    opacity=1.0
):
    """
    Add text to an image using OpenCV.
    
    Args:
        image: PIL Image or OpenCV Image.
        text (str): Text to add to the image.
        position (tuple or str): Position (x, y) or predefined position ('center', 'top', 'bottom', etc.).
        font_face (int): OpenCV font face.
        font_scale (float): Font scale factor.
        color (tuple): BGR color tuple.
        thickness (int): Thickness of the lines used to draw text.
        line_type (int): Line type (cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA).
        background (tuple): BGR background color for text. None for transparent.
        padding (int): Padding around text if background is used.
        opacity (float): Text opacity (0.0 to 1.0).
        
    Returns:
        numpy.ndarray: OpenCV image with added text.
    """
    # If image is in PIL format, convert to OpenCV format
    if not isinstance(image, np.ndarray):
        # Convert to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create a copy of the image
    result = image.copy()
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font_face, font_scale, thickness
    )
    
    # Determine position
    if isinstance(position, str):
        img_height, img_width = result.shape[:2]
        if position.lower() == "center":
            position = ((img_width - text_width) // 2, (img_height + text_height) // 2)
        elif position.lower() == "top":
            position = ((img_width - text_width) // 2, text_height + padding)
        elif position.lower() == "bottom":
            position = ((img_width - text_width) // 2, img_height - padding)
        elif position.lower() == "topleft":
            position = (padding, text_height + padding)
        elif position.lower() == "topright":
            position = (img_width - text_width - padding, text_height + padding)
        elif position.lower() == "bottomleft":
            position = (padding, img_height - padding)
        elif position.lower() == "bottomright":
            position = (img_width - text_width - padding, img_height - padding)
    
    x, y = position
    
    # Draw text background if specified
    if background:
        bg_x = x - padding
        bg_y = y - text_height - padding
        bg_width = text_width + (padding * 2)
        bg_height = text_height + (padding * 2) + baseline
        
        # Create overlay for opacity if needed
        if opacity < 1.0:
            overlay = result.copy()
            cv2.rectangle(
                overlay,
                (bg_x, bg_y),
                (bg_x + bg_width, bg_y + bg_height),
                background,
                -1
            )
            cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0, result)
        else:
            cv2.rectangle(
                result,
                (bg_x, bg_y),
                (bg_x + bg_width, bg_y + bg_height),
                background,
                -1
            )
    
    # Draw text
    if opacity < 1.0 and background is None:
        # For transparent text without background
        overlay = result.copy()
        cv2.putText(
            overlay, text, (x, y), font_face, font_scale, color, thickness, line_type
        )
        cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0, result)
    else:
        cv2.putText(
            result, text, (x, y), font_face, font_scale, color, thickness, line_type
        )
    
    return result


def add_text(
    image,
    text,
    library="pil",
    position="center",
    font=None,
    font_size=30,
    color=(0, 0, 0),
    background=None,
    **kwargs
):
    """
    Add text to an image using either PIL or OpenCV.
    
    Args:
        image: PIL Image or OpenCV Image.
        text (str): Text to add to the image.
        library (str): Library to use - 'pil' or 'cv2'.
        position (tuple or str): Position or predefined position.
        font: Font specification (path for PIL, face for OpenCV).
        font_size (int): Font size (or scale for OpenCV).
        color (tuple): RGB or BGR color tuple.
        background (tuple): Background color tuple.
        **kwargs: Additional parameters for the specific library.
        
    Returns:
        PIL.Image or numpy.ndarray: Image with added text.
    """
    if library.lower() == "pil":
        return add_text_pil(
            image,
            text,
            position=position,
            font_path=font,
            font_size=font_size,
            color=color,
            background=background,
            **kwargs
        )
    elif library.lower() == "cv2":
        font_face = font if font is not None else cv2.FONT_HERSHEY_SIMPLEX
        return add_text_cv2(
            image,
            text,
            position=position,
            font_face=font_face,
            font_scale=font_size / 30.0,  # Rough conversion from font size to scale
            color=color if len(color) == 3 else color[:3],  # Ensure BGR
            background=background,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown library: {library}")


if __name__ == "__main__":
    import os
    from .image_utils import load_image, save_image
    from .file_io import INPUT_DIR, OUTPUT_DIR
    
    # Test on a sample image
    sample_path = os.path.join(INPUT_DIR, "sample_plain_text.png")
    
    if os.path.exists(sample_path):
        # Load the image
        pil_image, cv_image = load_image(sample_path)
        
        # Add text using PIL
        pil_result = add_text(
            pil_image,
            "Added with PIL",
            library="pil",
            position="top",
            font_size=40,
            color=(255, 0, 0),
            background=(240, 240, 240, 200),
            padding=10,
            opacity=0.9
        )
        save_image(pil_result, os.path.join(OUTPUT_DIR, "text_added_pil.png"), format="PIL")
        
        # Add text using OpenCV
        cv_result = add_text(
            cv_image,
            "Added with OpenCV",
            library="cv2",
            position="bottom",
            font_size=40,
            color=(0, 0, 255),  # BGR in OpenCV
            background=(240, 240, 240),
            padding=10,
            opacity=0.9
        )
        save_image(cv_result, os.path.join(OUTPUT_DIR, "text_added_cv2.png"), format="CV")
        
        print("Text addition complete. Check the output directory for results.")
    else:
        print(f"Sample image not found: {sample_path}")
        print("Run generate_samples.py first to create sample images.") 
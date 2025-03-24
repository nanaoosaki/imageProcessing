"""
Comic Text Addition Script

This script adds text boxes with white backgrounds to specific areas in an image,
creating a comic or sketch-like appearance.
"""
import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our utilities
from src.utils.image_utils import load_image, save_image
from src.utils.text_addition import add_text_pil

def add_comic_text(input_path, output_path, text_configs):
    """
    Add multiple text boxes with white backgrounds to an image.
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the output image
        text_configs (list): List of dictionaries with text configuration:
            - text: Text to add
            - position: (x, y) position or predefined position
            - font_path: Path to font file
            - font_size: Font size
            - color: RGB color tuple
            - background: RGBA background color
            - padding: Padding around text
            - opacity: Text and background opacity
            - align: Text alignment
            - max_width: Maximum width for text wrapping
            - angle: Rotation angle in degrees
    """
    print(f"Loading image: {input_path}")
    pil_image, _ = load_image(input_path)
    
    # Create a copy of the image
    result = pil_image.copy()
    
    for i, config in enumerate(text_configs):
        print(f"Adding text box {i+1}: '{config['text'][:20]}...'")
        
        # Extract configuration with defaults
        text = config['text']
        position = config.get('position', 'center')
        font_path = config.get('font_path', None)
        font_size = config.get('font_size', 30)
        color = config.get('color', (0, 0, 0))
        background = config.get('background', (255, 255, 255, 200))
        padding = config.get('padding', 10)
        opacity = config.get('opacity', 1.0)
        align = config.get('align', 'center')
        max_width = config.get('max_width', None)
        angle = config.get('angle', 0)  # Rotation angle
        
        # Wrap text if max_width is specified
        if max_width:
            # Get font
            try:
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Create a temporary draw object to measure text
            temp_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
            
            # Split text into words
            words = text.split()
            wrapped_text = ''
            current_line = ''
            
            for word in words:
                # Measure current line + new word
                test_line = current_line + ' ' + word if current_line else word
                if hasattr(temp_draw, 'textbbox'):
                    # For newer PIL versions
                    bbox = temp_draw.textbbox((0, 0), test_line, font=font)
                    line_width = bbox[2] - bbox[0]
                else:
                    # For older PIL versions
                    line_width, _ = temp_draw.textsize(test_line, font=font)
                
                # If line would be too long, start a new line
                if line_width > max_width:
                    if current_line:
                        wrapped_text += current_line + '\n'
                        current_line = word
                    else:
                        # Word itself is too long, force it on its own line
                        wrapped_text += word + '\n'
                        current_line = ''
                else:
                    current_line = test_line
            
            # Add the last line
            if current_line:
                wrapped_text += current_line
            
            # Use wrapped text
            text = wrapped_text
        
        # If angle is specified, we need to create a rotated text image and paste it
        if angle != 0:
            # Create a transparent image for the text
            text_img = Image.new('RGBA', result.size, (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_img)
            
            # Add text with specified configuration but without background
            # We'll create a separate background layer
            x, y = position
            
            # Get font
            try:
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                elif font_path:
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Calculate text size
            if hasattr(text_draw, 'textbbox'):
                # For newer PIL versions
                bbox = text_draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # For older PIL versions
                text_width, text_height = text_draw.textsize(text, font=font)
            
            # Create a slightly larger box for the background
            bg_width = text_width + (padding * 2)
            bg_height = text_height + (padding * 2)
            
            # Create background image
            bg_img = Image.new('RGBA', (bg_width, bg_height), background)
            
            # Create text image
            txt_img = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            
            # Draw text centered in the box
            txt_draw.text(
                (padding, padding),
                text,
                font=font,
                fill=(*color, int(255 * opacity))
            )
            
            # Composite text over background
            combined = Image.alpha_composite(bg_img, txt_img)
            
            # Rotate combined image
            rotated = combined.rotate(angle, expand=True, resample=Image.BICUBIC)
            
            # Calculate new position after rotation
            new_x = x - rotated.width // 2
            new_y = y - rotated.height // 2
            
            # Create a transparent layer for the result image
            temp_layer = Image.new('RGBA', result.size, (0, 0, 0, 0))
            
            # Paste rotated image onto the layer
            temp_layer.paste(rotated, (new_x, new_y), rotated)
            
            # Convert result to RGBA if needed
            if result.mode != 'RGBA':
                result = result.convert('RGBA')
            
            # Composite the layer with the result
            result = Image.alpha_composite(result, temp_layer)
        else:
            # Add text with specified configuration
            result = add_text_pil(
                result,
                text,
                position=position,
                font_path=font_path,
                font_size=font_size,
                color=color,
                align=align,
                background=background,
                padding=padding,
                opacity=opacity
            )
    
    # Save the result
    save_image(result, output_path, format="PIL")
    print(f"Text added successfully. Saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add comic-style text boxes to an image")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to save output image")
    parser.add_argument("--sketch", action="store_true", help="Use sketch-like styling")
    parser.add_argument("--config", help="Path to text configuration file")
    parser.add_argument("--preview", action="store_true", help="Generate a visualization of the text positions before adding text")
    
    args = parser.parse_args()
    
    # Define default text configurations based on the specific task
    default_text_configs = [
        {
            'text': "WE HAVE A NEW DASHBOARD WITH ALL OUR KPI TRACKED!",
            'position': (400, 30),  # Top of the image
            'font_size': 20,
            'color': (20, 20, 20),  # Black text
            'background': (255, 255, 255, 240),  # White background with some transparency
            'padding': 10,
            'align': 'center',
            'max_width': 500,  # Wrap text if it exceeds this width
            'angle': 0.5  # Slight tilt for sketch feel
        },
        {
            'text': "DID THAT KPI JUST SKYROCKETED TO THE ROOF?",
            'position': (230, 180),  # Left speech bubble (person standing)
            'font_size': 16,
            'color': (20, 20, 20),
            'background': (255, 255, 255, 240),
            'padding': 8,
            'align': 'center',
            'max_width': 220,
            'angle': -1  # Slight tilt in the opposite direction
        },
        {
            'text': "I'LL NEED TO CREATE A NEW DASHBOARD TO EXPLAIN THAT!",
            'position': (425, 45),  # Right speech bubble (person sitting)
            'font_size': 14,
            'color': (20, 20, 20),
            'background': (255, 255, 255, 240),
            'padding': 6,
            'align': 'center',
            'max_width': 180,
            'angle': 1.5  # More pronounced tilt
        }
    ]
    
    # Load text configurations from file if specified
    text_configs = default_text_configs
    if args.config:
        try:
            # Try to import the configuration file as a module
            import importlib.util
            import sys
            
            # Strip .py extension if provided
            config_path = args.config
            if config_path.endswith('.py'):
                config_path = config_path[:-3]
                
            # Replace path separators with dots for module name
            module_name = config_path.replace('/', '.').replace('\\', '.')
            
            # Remove ./ prefix if present
            if module_name.startswith('./'):
                module_name = module_name[2:]
            
            # Try different import methods
            try:
                # Method 1: Direct import
                config_module = __import__(module_name)
                if hasattr(config_module, 'text_configs'):
                    text_configs = config_module.text_configs
                elif hasattr(config_module, 'region_configs'):
                    text_configs = config_module.region_configs
            except ImportError:
                # Method 2: Using importlib.util
                spec = importlib.util.spec_from_file_location(module_name, args.config + '.py')
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                
                if hasattr(config_module, 'text_configs'):
                    text_configs = config_module.text_configs
                elif hasattr(config_module, 'region_configs'):
                    text_configs = config_module.region_configs
            
            print(f"Loaded {len(text_configs)} text configurations from {args.config}")
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            print("Using default text configurations instead.")
    
    # Preview text positions if requested
    if args.preview:
        try:
            import cv2
            
            # Load the image
            _, cv_image = load_image(args.input)
            
            # Create a copy of the image
            preview_image = cv_image.copy()
            
            # Draw position markers for each text config
            for i, config in enumerate(text_configs):
                position = config.get('position', (0, 0))
                x, y = position
                
                # Calculate text size (approximate)
                font_size = config.get('font_size', 16)
                text = config.get('text', f"Text {i+1}")
                max_width = config.get('max_width', 200)
                
                # Calculate text box dimensions (approximate)
                text_width = min(len(text) * font_size * 0.6, max_width)
                text_height = font_size * (1 + text.count('\n') + text_width // max_width)
                
                # Draw position marker
                cv2.circle(preview_image, (x, y), 5, (0, 0, 255), -1)
                
                # Draw text box outline
                padding = config.get('padding', 8)
                half_width = text_width // 2
                half_height = text_height // 2
                cv2.rectangle(
                    preview_image,
                    (x - half_width - padding, y - half_height - padding),
                    (x + half_width + padding, y + half_height + padding),
                    (0, 255, 255),
                    2
                )
                
                # Draw text config number
                cv2.putText(
                    preview_image,
                    f"Text {i+1}",
                    (x - half_width, y - half_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
            
            # Save preview image
            preview_path = args.output.replace('.png', '_preview.png')
            cv2.imwrite(preview_path, preview_image)
            print(f"Saved text position preview to: {preview_path}")
            
            # Show preview image
            cv2.imshow('Text Positions Preview', preview_image)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error generating preview: {e}")
            print("Continuing without preview...")
    
    # If sketch style is specified, adjust the text configuration
    if args.sketch:
        # Use a more handwritten-looking font and adjust styling for sketch look
        comic_fonts = [
            "Comic Sans MS",
            "Arial Rounded MT Bold",
            "Marker Felt",
            "Segoe Print",
            "Kristen ITC"
        ]
        
        # Try to find a sketch-like font
        font_found = False
        for font in comic_fonts:
            try:
                # Test if font is available
                ImageFont.truetype(font, 12)
                # If we get here, font exists
                font_found = True
                break
            except:
                continue
        
        # Apply found font to all configs
        if font_found:
            for config in text_configs:
                config['font_path'] = font
                # Make text slightly bolder
                config['color'] = (40, 40, 40)  # Darker black
    
    # Print debug info
    print("Text configurations:")
    for i, config in enumerate(text_configs):
        print(f"  Box {i+1}: Position {config['position']}, Font size {config['font_size']}")
    
    # Add the comic text
    add_comic_text(args.input, args.output, text_configs) 
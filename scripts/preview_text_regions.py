"""
Preview Text Regions Script

This script identifies potential text regions in an image and visualizes them
with numbered boxes for easy reference.
"""
import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our utilities
from src.utils.image_utils import load_image, save_image

# Define some common text regions for comic-style images
DEFAULT_REGIONS = [
    # General regions
    {'name': 'Top', 'box': (50, 10, 750, 70)},
    {'name': 'Middle', 'box': (50, 200, 750, 300)},
    {'name': 'Bottom', 'box': (50, 450, 750, 550)},
    
    # Speech bubble regions
    {'name': 'Left Speech', 'box': (100, 150, 350, 250)},
    {'name': 'Right Speech', 'box': (450, 150, 700, 250)},
    {'name': 'Bottom Left', 'box': (100, 350, 350, 450)},
    {'name': 'Bottom Right', 'box': (450, 350, 700, 450)},
]

def find_speech_bubbles(image, min_area=1000, max_area=100000):
    """
    Find potential speech bubble regions using contour detection.
    
    Args:
        image: OpenCV image
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider
        
    Returns:
        List of potential speech bubble regions
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth out the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    speech_bubbles = []
    for i, contour in enumerate(contours):
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Skip if too small or too large
        if area < min_area or area > max_area:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip if too wide or too tall
        if w > image.shape[1] * 0.9 or h > image.shape[0] * 0.9:
            continue
        
        # Add some padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Add to list
        speech_bubbles.append({
            'name': f'Auto Bubble {i+1}',
            'box': (x1, y1, x2, y2)
        })
    
    return speech_bubbles

def visualize_text_regions(image, regions, save_path=None):
    """
    Visualize text regions with numbered boxes.
    
    Args:
        image: OpenCV image
        regions: List of region dictionaries
        save_path: Path to save the visualization
        
    Returns:
        OpenCV image with visualized regions
    """
    # Create a copy of the image
    result = image.copy()
    
    # Colors for different region types
    colors = {
        'default': (0, 255, 0),    # Green for default regions
        'auto': (0, 0, 255),       # Blue for auto-detected regions
        'custom': (255, 0, 255)    # Purple for custom regions
    }
    
    # Draw boxes and labels
    for i, region in enumerate(regions):
        # Get the box coordinates
        x1, y1, x2, y2 = region['box']
        
        # Determine color based on name
        if region['name'].startswith('Auto'):
            color = colors['auto']
        elif region['name'].startswith('Custom'):
            color = colors['custom']
        else:
            color = colors['default']
        
        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with region number
        label = f"{i+1}: {region['name']}"
        cv2.putText(result, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2, cv2.LINE_AA)
    
    # Save the result if requested
    if save_path:
        save_image(result, save_path, format="CV")
        print(f"Saved visualization to: {save_path}")
    
    return result

def create_custom_regions_from_speech_bubbles(image):
    """
    Create custom regions for each speech bubble in the image.
    
    Args:
        image: OpenCV image
        
    Returns:
        List of custom region dictionaries
    """
    # Find speech bubbles
    speech_bubbles = find_speech_bubbles(image)
    
    # Convert to custom regions
    custom_regions = []
    for i, bubble in enumerate(speech_bubbles):
        custom_regions.append({
            'name': f'Speech Bubble {i+1}',
            'box': bubble['box']
        })
    
    return custom_regions

def preview_text_regions(input_path, output_path=None, include_auto=True, include_default=True):
    """
    Preview text regions in an image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save the visualization
        include_auto: Whether to include auto-detected regions
        include_default: Whether to include default regions
    """
    # Load the image
    _, cv_image = load_image(input_path)
    
    # Define regions list
    regions = []
    
    # Add default regions if requested
    if include_default:
        regions.extend(DEFAULT_REGIONS)
    
    # Add auto-detected regions if requested
    if include_auto:
        auto_regions = create_custom_regions_from_speech_bubbles(cv_image)
        regions.extend(auto_regions)
    
    # Create output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join("output", f"{base_name}_text_regions.png")
    
    # Visualize regions
    result = visualize_text_regions(cv_image, regions, output_path)
    
    # Print region information
    print("\nPotential Text Regions:")
    print("======================")
    for i, region in enumerate(regions):
        x1, y1, x2, y2 = region['box']
        print(f"Region {i+1}: {region['name']} - Position: ({x1}, {y1}), Size: {x2-x1}x{y2-y1}")
    
    # Return the visualization and regions
    return result, regions

def save_region_configurations(regions, output_path="region_config.py"):
    """
    Save region configurations to a Python script.
    
    Args:
        regions: List of region dictionaries
        output_path: Path to save the script
    """
    with open(output_path, 'w') as f:
        f.write('"""\nRegion Configuration\n\nGenerated by preview_text_regions.py\n"""\n\n')
        f.write('# Text region configurations for add_comic_text.py\n')
        f.write('region_configs = [\n')
        
        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region['box']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            
            f.write('    {\n')
            f.write(f'        \'text\': "Text for {region["name"]}",\n')
            f.write(f'        \'position\': ({center_x}, {center_y}),  # Center of region {i+1}\n')
            f.write(f'        \'font_size\': {max(12, min(24, width // 15))},\n')
            f.write( '        \'color\': (20, 20, 20),\n')
            f.write( '        \'background\': (255, 255, 255, 240),\n')
            f.write( '        \'padding\': 8,\n')
            f.write( '        \'align\': \'center\',\n')
            f.write(f'        \'max_width\': {max(100, width - 40)},\n')
            f.write( '        \'angle\': 0.5,\n')
            f.write('    },\n')
        
        f.write(']\n')
    
    print(f"\nSaved region configurations to: {output_path}")
    print("You can customize the text and other properties in this file.")
    print("Then use it with add_comic_text.py by importing the configuration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview text regions in an image")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("--output", help="Path to save the visualization")
    parser.add_argument("--no-auto", action="store_true", help="Don't include auto-detected regions")
    parser.add_argument("--no-default", action="store_true", help="Don't include default regions")
    parser.add_argument("--generate-config", action="store_true", help="Generate region config file")
    
    args = parser.parse_args()
    
    # Preview text regions
    result, regions = preview_text_regions(
        args.input, 
        args.output, 
        not args.no_auto, 
        not args.no_default
    )
    
    # Generate region configurations if requested
    if args.generate_config:
        save_region_configurations(regions)
    
    # Display the result
    cv2.imshow('Text Regions', result)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
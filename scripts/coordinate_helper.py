"""
Coordinate Helper Script

This script displays an image with coordinate grid overlay and allows
clicking to get precise coordinates for text box placement.
"""
import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our utilities
from src.utils.image_utils import load_image, save_image

# Global variables to store click coordinates
click_coordinates = []

def draw_coordinate_grid(image, grid_spacing=50, line_thickness=1):
    """
    Draw a coordinate grid on an image.
    
    Args:
        image: OpenCV image
        grid_spacing: Spacing between grid lines
        line_thickness: Thickness of grid lines
        
    Returns:
        OpenCV image with grid overlay
    """
    height, width = image.shape[:2]
    result = image.copy()
    
    # Draw vertical lines
    for x in range(0, width, grid_spacing):
        cv2.line(result, (x, 0), (x, height), (100, 100, 100), line_thickness)
        # Add x-coordinate labels
        cv2.putText(result, str(x), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw horizontal lines
    for y in range(0, height, grid_spacing):
        cv2.line(result, (0, y), (width, y), (100, 100, 100), line_thickness)
        # Add y-coordinate labels
        cv2.putText(result, str(y), (5, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return result

def click_event(event, x, y, flags, param):
    """Handle mouse click events."""
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the coordinates
        click_coordinates.append((x, y))
        
        # Get the image from param
        image = param
        
        # Draw a circle at the clicked point
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        
        # Display the coordinates near the point
        text = f"({x}, {y})"
        cv2.putText(image, text, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Show the updated image
        cv2.imshow('Coordinate Helper', image)
        
        # Print coordinates to console for easy copy-paste
        print(f"Selected coordinates: ({x}, {y})")

def show_coordinate_grid(input_path, grid_spacing=50, save_output=False):
    """
    Display an image with coordinate grid and capture mouse clicks.
    
    Args:
        input_path: Path to the input image
        grid_spacing: Spacing between grid lines
        save_output: Whether to save the grid image
    """
    # Load the image
    _, cv_image = load_image(input_path)
    
    # Draw the coordinate grid
    grid_image = draw_coordinate_grid(cv_image, grid_spacing)
    
    # Create a window and set the mouse callback
    cv2.namedWindow('Coordinate Helper')
    cv2.setMouseCallback('Coordinate Helper', click_event, grid_image)
    
    # Show instructions
    print("\nCoordinate Helper")
    print("=================")
    print("Click on the image to get coordinates for text box placement.")
    print("Press 'ESC' to exit.\n")
    
    # Display the image
    cv2.imshow('Coordinate Helper', grid_image)
    
    # Save the grid image if requested
    if save_output:
        output_path = os.path.splitext(input_path)[0] + "_grid.png"
        save_image(grid_image, output_path, format="CV")
        print(f"Saved grid image to: {output_path}")
    
    # Wait for key press and handle it
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC key
            break
    
    # Clean up
    cv2.destroyAllWindows()
    
    # Print summary of all clicked positions
    if click_coordinates:
        print("\nAll selected coordinates:")
        for i, coords in enumerate(click_coordinates):
            print(f"Position {i+1}: {coords}")
    
    return click_coordinates

def generate_text_config_from_clicks(coordinates, text_list):
    """
    Generate text configurations from clicked coordinates and text list.
    
    Args:
        coordinates: List of (x, y) coordinates
        text_list: List of text strings
        
    Returns:
        List of text configurations for add_comic_text.py
    """
    text_configs = []
    
    # Use the minimum of coordinates and text list lengths
    num_configs = min(len(coordinates), len(text_list))
    
    for i in range(num_configs):
        x, y = coordinates[i]
        text = text_list[i]
        
        config = {
            'text': text,
            'position': (x, y),
            'font_size': 16,  # Default size, can be adjusted
            'background': (255, 255, 255, 230),
            'padding': 8,
            'align': 'center',
            'max_width': 200,  # Default width, can be adjusted
            'angle': 0.5  # Slight tilt for sketch feel
        }
        
        text_configs.append(config)
    
    return text_configs

def save_text_config_script(text_configs, output_path="text_config.py"):
    """
    Save the text configurations to a Python script.
    
    Args:
        text_configs: List of text configuration dictionaries
        output_path: Path to save the script
    """
    with open(output_path, 'w') as f:
        f.write('"""\nText Configuration\n\nGenerated by coordinate_helper.py\n"""\n\n')
        f.write('# Text configurations for add_comic_text.py\n')
        f.write('text_configs = [\n')
        
        for config in text_configs:
            f.write('    {\n')
            for key, value in config.items():
                if isinstance(value, str):
                    f.write(f'        \'{key}\': "{value}",\n')
                else:
                    f.write(f'        \'{key}\': {value},\n')
            f.write('    },\n')
        
        f.write(']\n')
    
    print(f"\nSaved text configurations to: {output_path}")
    print("You can now use these configurations with add_comic_text.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display an image with coordinate grid and capture clicks")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("--grid-spacing", type=int, default=50, help="Spacing between grid lines (default: 50)")
    parser.add_argument("--save-grid", action="store_true", help="Save the grid image")
    parser.add_argument("--generate-config", action="store_true", help="Generate text config file from clicks")
    parser.add_argument("--text", nargs='+', help="Text strings for each click position (use with --generate-config)")
    
    args = parser.parse_args()
    
    # Show the coordinate grid and get the clicked coordinates
    coordinates = show_coordinate_grid(args.input, args.grid_spacing, args.save_grid)
    
    # Generate text configurations if requested
    if args.generate_config and args.text:
        text_configs = generate_text_config_from_clicks(coordinates, args.text)
        
        # Save the configurations to a Python script
        save_text_config_script(text_configs) 
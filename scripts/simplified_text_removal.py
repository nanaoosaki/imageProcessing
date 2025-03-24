"""
Simplified Text Removal Script with User Confirmation

This script provides a two-step process for text removal without dependencies on pytesseract:
1. Generate a visualization of detected text regions
2. Remove text from specified regions based on user input
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import only the necessary utilities
from src.utils.image_utils import load_image, save_image

# Skip importing pytesseract-related functions
# We'll implement a simplified version of what we need directly in this script

def detect_text_easyocr(image, min_confidence=0.1, detailed_mode=False):
    """
    Detect text in an image using EasyOCR.
    
    Args:
        image: OpenCV Image.
        min_confidence (float): Minimum confidence threshold (0-1).
        detailed_mode (bool): If True, use enhanced detection for single characters.
        
    Returns:
        list: List of dictionaries with text regions.
    """
    # Import EasyOCR here to handle any import errors gracefully
    try:
        import easyocr
    except ImportError:
        print("EasyOCR is not installed. Please install it with: pip install easyocr")
        sys.exit(1)
    
    # Use a much lower confidence threshold in detailed mode
    if detailed_mode:
        min_confidence = min(min_confidence, 0.05)
        print(f"Using detailed detection mode with lower confidence threshold: {min_confidence}")
        
        # Apply additional preprocessing for better single character detection
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Dilate to connect components 
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours - this can help detect isolated characters
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        print("EasyOCR initialized. Reading text in detailed mode...")
        
        # Process standard EasyOCR results
        easyocr_results = reader.readtext(image)
        
        # Process results
        results = []
        for detection in easyocr_results:
            box, text, confidence = detection
            
            if confidence < min_confidence:
                continue
                
            # Convert box to x1, y1, x2, y2 format
            x1, y1 = map(int, box[0])
            x3, y3 = map(int, box[2])
            
            results.append({
                'text': text,
                'box': (x1, y1, x3, y3),  # (x1, y1, x2, y2) format
                'confidence': confidence
            })
        
        # Add additional regions based on contours for potential single characters
        for cnt in contours:
            # Filter out very small or very large contours
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            
            # Skip contours that are likely not characters
            if area < 50 or area > 2000 or w < 5 or h < 10 or w > 100 or h > 100:
                continue
                
            # Check if this contour overlaps with existing detections
            overlap = False
            for result in results:
                box = result['box']
                # Check if center of contour is within an existing box
                center_x = x + w // 2
                center_y = y + h // 2
                if (box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]):
                    overlap = True
                    break
            
            if not overlap:
                # Add padding to the contour box
                padding = 2
                results.append({
                    'text': '?',  # Unknown character
                    'box': (x-padding, y-padding, x+w+padding, y+h+padding),
                    'confidence': 0.1,  # Low confidence for contour-based detections
                    'contour_based': True
                })
        
        return results
    else:
        # Standard detection mode - original code
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        print("EasyOCR initialized. Reading text...")
        
        # Get text data
        easyocr_results = reader.readtext(image)
        
        # Process results
        results = []
        for detection in easyocr_results:
            box, text, confidence = detection
            
            if confidence < min_confidence:
                continue
                
            # Convert box to x1, y1, x2, y2 format
            x1, y1 = map(int, box[0])
            x3, y3 = map(int, box[2])
            
            results.append({
                'text': text,
                'box': (x1, y1, x3, y3),  # (x1, y1, x2, y2) format
                'confidence': confidence
            })
        
        return results

def highlight_text_regions(image, regions, color=(0, 255, 0), thickness=2):
    """
    Draw rectangles around detected text regions.
    
    Args:
        image: OpenCV Image.
        regions (list): List of detected text regions.
        color (tuple): BGR color for the bounding boxes.
        thickness (int): Thickness of the bounding box lines.
        
    Returns:
        numpy.ndarray: OpenCV image with highlighted text regions.
    """
    # Create a copy of the image
    output = image.copy()
    
    # Draw rectangles
    for region in regions:
        box = region['box']
        cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), color, thickness)
    
    return output

def remove_text_inpaint(image, regions, radius=10):
    """
    Remove text from an image using OpenCV's inpainting.
    
    Args:
        image: OpenCV Image.
        regions (list): List of detected text regions.
        radius (int): Radius of neighborhood for inpainting.
        
    Returns:
        numpy.ndarray: OpenCV image with text removed.
    """
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
    result_ns = cv2.inpaint(result, mask, radius, cv2.INPAINT_NS)
    
    return result_ns

def detect_and_visualize(input_path, detailed_mode=False):
    """
    Detect text regions and create a visualization with region numbers.
    
    Args:
        input_path (str): Path to input image
        detailed_mode (bool): If True, use enhanced detection for single characters
    
    Returns:
        tuple: (all_regions, output_path)
    """
    # Define output path for highlighted image
    output_path = os.path.join("output", os.path.basename(input_path).replace(".", "_regions."))
    
    print(f"Processing image: {input_path}")
    
    # Load the image
    pil_image, cv_image = load_image(input_path)
    
    # Get image dimensions
    height, width = cv_image.shape[:2]
    
    # Detect text with low confidence threshold
    min_confidence = 0.1
    print(f"Detecting text with EasyOCR using {'detailed mode and ' if detailed_mode else ''}low confidence threshold: {min_confidence}")
    auto_regions = detect_text_easyocr(cv_image, min_confidence=min_confidence, detailed_mode=detailed_mode)
    
    print(f"\nAutomatically detected {len(auto_regions)} text regions:")
    for i, region in enumerate(auto_regions):
        contour_info = " (contour-based)" if region.get('contour_based', False) else ""
        print(f"  Region {i+1}: '{region['text']}' (Confidence: {region['confidence']:.2f}){contour_info}")
    
    # Add manual regions for areas that might be missed
    print("\nAdding manual regions for areas that might be missed...")
    manual_regions = [
        # Top text - full width to ensure everything is captured
        {'text': 'TOP TEXT AREA', 'box': (20, 10, width-20, 60), 'confidence': 1.0},
        
        # Middle text area
        {'text': 'MIDDLE TEXT AREA', 'box': (0, 220, 250, 320), 'confidence': 1.0},
        
        # Bottom speech bubble
        {'text': 'SPEECH BUBBLE', 'box': (280, 300, 420, 370), 'confidence': 1.0},
    ]
    
    # Combine all regions
    all_regions = auto_regions + manual_regions
    
    # Create a visualization with numbered regions
    highlighted_image = cv_image.copy()
    
    # Highlight auto regions in green with numbers
    for i, region in enumerate(auto_regions):
        box = region['box']
        # Use different color for contour-based detections
        color = (0, 200, 0)  # Regular green
        if region.get('contour_based', False):
            color = (0, 255, 255)  # Yellow for contour-based
            
        # Draw rectangle
        cv2.rectangle(highlighted_image, (box[0], box[1]), (box[2], box[3]), color, 2)
        # Add region number
        cv2.putText(
            highlighted_image, 
            f"{i+1}", 
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            color, 
            2
        )
    
    # Highlight manual regions in blue with numbers
    for i, region in enumerate(manual_regions):
        box = region['box']
        # Draw rectangle
        cv2.rectangle(highlighted_image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        # Add region number
        cv2.putText(
            highlighted_image, 
            f"{len(auto_regions) + i + 1}", 
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 0, 0), 
            2
        )
    
    # Save the highlighted image
    save_image(highlighted_image, output_path, format="CV")
    print(f"\nSaved highlighted regions to: {output_path}")
    print(f"Green regions (1-{len(auto_regions)}): Automatically detected text")
    if detailed_mode:
        print(f"Yellow regions: Potential single characters detected via contour analysis")
    print(f"Blue regions ({len(auto_regions)+1}-{len(all_regions)}): Manually added areas")
    
    print("\nTo remove specific regions, run:")
    print(f"python scripts/simplified_text_removal.py remove {input_path} 1 2 3")
    print("(Replace 1 2 3 with the region numbers you want to remove)")
    
    return all_regions, output_path

def remove_regions(input_path, region_numbers, detailed_mode=False):
    """
    Remove text from specified regions.
    
    Args:
        input_path (str): Path to input image
        region_numbers (list): List of region numbers to remove
        detailed_mode (bool): If True, use enhanced detection for single characters
    """
    # Define output path
    output_path = os.path.join("output", os.path.basename(input_path).replace(".", "_text_removed."))
    
    # First detect all regions
    all_regions, _ = detect_and_visualize(input_path, detailed_mode=detailed_mode)
    
    # Convert region numbers to indices (adjusting for 1-based numbering)
    region_indices = [int(num) - 1 for num in region_numbers]
    
    # Filter for valid indices
    valid_indices = [i for i in region_indices if 0 <= i < len(all_regions)]
    
    if not valid_indices:
        print("Error: No valid region numbers provided.")
        return
    
    # Get regions to remove
    regions_to_remove = [all_regions[i] for i in valid_indices]
    
    # Load the image
    _, cv_image = load_image(input_path)
    
    # Create visualization of selected regions
    result_image = cv_image.copy()
    for i, region in enumerate(regions_to_remove):
        box = region['box']
        # Draw rectangle
        cv2.rectangle(result_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    
    # Save the selected regions image
    selected_path = output_path.replace("_text_removed.", "_selected_regions.")
    save_image(result_image, selected_path, format="CV")
    print(f"Saved selected regions to: {selected_path}")
    
    # Remove text
    print(f"Removing text from {len(regions_to_remove)} regions...")
    result = remove_text_inpaint(cv_image, regions_to_remove, radius=15)
    
    # Save result
    save_image(result, output_path, format="CV")
    print(f"Text removal complete. Result saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  To detect text regions: python scripts/simplified_text_removal.py detect input/image.png [--detailed]")
        print("  To remove text regions: python scripts/simplified_text_removal.py remove input/image.png 1 2 3 ... [--detailed]")
        sys.exit(1)
    
    command = sys.argv[1]
    input_path = sys.argv[2]
    
    # Check if detailed mode is requested
    detailed_mode = "--detailed" in sys.argv
    
    if command == "detect":
        detect_and_visualize(input_path, detailed_mode=detailed_mode)
    elif command == "remove":
        # Extract only the region numbers from the arguments (exclude the --detailed flag)
        region_numbers = [arg for arg in sys.argv[3:] if not arg.startswith("--")]
        
        if not region_numbers:
            print("Error: Please specify region numbers to remove.")
            print("Example: python scripts/simplified_text_removal.py remove input/image.png 1 2 3 [--detailed]")
            sys.exit(1)
        
        remove_regions(input_path, region_numbers, detailed_mode=detailed_mode)
    else:
        print("Invalid command. Use 'detect' or 'remove'.") 
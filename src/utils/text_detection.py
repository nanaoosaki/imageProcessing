"""
Text Detection Utilities Module

This module provides functions for detecting text in images
using both Pytesseract and EasyOCR.
"""
import os
import cv2
import numpy as np
import pytesseract
import easyocr

# Initialize the EasyOCR reader (this may take a moment)
# Only initialize when needed to save memory
_easyocr_reader = None


def get_easyocr_reader(languages=['en']):
    """
    Get or initialize the EasyOCR reader.
    
    Args:
        languages (list): List of language codes to support.
        
    Returns:
        easyocr.Reader: Initialized EasyOCR reader.
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(languages)
    return _easyocr_reader


def detect_text_tesseract(image, min_confidence=0):
    """
    Detect text in an image using Pytesseract.
    
    Args:
        image: PIL Image or OpenCV Image.
        min_confidence (float): Minimum confidence threshold (0-100).
        
    Returns:
        list: List of dictionaries with text regions (text, box coordinates, confidence).
    """
    # If image is in OpenCV format, convert to RGB
    if isinstance(image, np.ndarray):
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = image
    
    # Get text data
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    
    # Process results
    results = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        # Skip empty text
        if int(data['conf'][i]) < min_confidence or not data['text'][i].strip():
            continue
            
        # Extract coordinates
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        
        results.append({
            'text': data['text'][i],
            'box': (x, y, x + w, y + h),  # (x1, y1, x2, y2) format
            'confidence': float(data['conf'][i])
        })
    
    return results


def detect_text_easyocr(image, min_confidence=0.5):
    """
    Detect text in an image using EasyOCR.
    
    Args:
        image: PIL Image or OpenCV Image.
        min_confidence (float): Minimum confidence threshold (0-1).
        
    Returns:
        list: List of dictionaries with text regions (text, box coordinates, confidence).
    """
    # If image is in PIL format, convert to OpenCV format
    if not isinstance(image, np.ndarray):
        # Convert to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Initialize EasyOCR reader
    reader = get_easyocr_reader()
    
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


def highlight_text_regions(image, regions, color=(255, 0, 0), thickness=2):
    """
    Draw rectangles around detected text regions.
    
    Args:
        image: PIL Image or OpenCV Image.
        regions (list): List of detected text regions.
        color (tuple): RGB color for the bounding boxes.
        thickness (int): Thickness of the bounding box lines.
        
    Returns:
        numpy.ndarray: OpenCV image with highlighted text regions.
    """
    # If image is in PIL format, convert to OpenCV format
    if not isinstance(image, np.ndarray):
        # Convert to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create a copy of the image
    output = image.copy()
    
    # Draw rectangles and text
    for idx, region in enumerate(regions):
        box = region['box']
        text = region['text']
        confidence = region.get('confidence', 0)
        
        # Draw rectangle
        cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), color, thickness)
        
        # Draw text and confidence
        label = f"{text} ({confidence:.2f})"
        cv2.putText(output, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output


if __name__ == "__main__":
    from image_utils import load_image, save_image
    import os
    
    # Test on a sample image
    sample_path = os.path.join("samples", "sample_text.png")
    
    if os.path.exists(sample_path):
        # Load the image
        pil_image, cv_image = load_image(sample_path)
        
        # Detect text using Tesseract
        tesseract_regions = detect_text_tesseract(pil_image)
        print(f"Tesseract detected {len(tesseract_regions)} text regions")
        
        # Highlight Tesseract results
        tesseract_result = highlight_text_regions(cv_image, tesseract_regions)
        save_image(tesseract_result, os.path.join("samples", "tesseract_detection.png"), format="CV")
        
        # Detect text using EasyOCR
        easyocr_regions = detect_text_easyocr(cv_image)
        print(f"EasyOCR detected {len(easyocr_regions)} text regions")
        
        # Highlight EasyOCR results
        easyocr_result = highlight_text_regions(cv_image, easyocr_regions, color=(0, 255, 0))
        save_image(easyocr_result, os.path.join("samples", "easyocr_detection.png"), format="CV")
    else:
        print(f"Sample image not found: {sample_path}")
        print("Run image_utils.py first to create a sample image.") 
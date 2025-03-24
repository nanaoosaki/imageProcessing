"""
Library Comparison Script for Image Processing
This script checks the availability of various image processing libraries
and prints basic information about their capabilities.
"""
import sys
import importlib
import pkg_resources

def check_library(lib_name):
    """Check if a library is installed and print its version."""
    try:
        lib = importlib.import_module(lib_name)
        try:
            version = pkg_resources.get_distribution(lib_name).version
        except:
            version = getattr(lib, '__version__', 'unknown')
        print(f"✓ {lib_name} is installed (version: {version})")
        return True
    except ImportError:
        print(f"✗ {lib_name} is not installed")
        return False

def main():
    """Main function to check libraries and their capabilities."""
    print("\n=== Image Processing Libraries Comparison ===\n")
    
    # Check common image processing libraries
    pillow_installed = check_library('PIL')
    cv2_installed = check_library('cv2')
    skimage_installed = check_library('skimage')
    
    # Check OCR libraries
    print("\n=== Text Detection/OCR Libraries ===\n")
    pytesseract_installed = check_library('pytesseract')
    easyocr_installed = check_library('easyocr')
    
    # Print capabilities summary
    print("\n=== Capabilities Summary ===\n")
    
    if pillow_installed:
        print("Pillow (PIL):")
        print("  ✓ Basic image processing (resize, crop, filter)")
        print("  ✓ Simple text addition with ImageDraw")
        print("  ✗ No built-in text detection/OCR")
        print("  ✓ Lightweight and easy to use")
        print("  ✗ Limited advanced processing capabilities")
    
    if cv2_installed:
        print("\nOpenCV:")
        print("  ✓ Comprehensive image processing")
        print("  ✓ Text addition with various options")
        print("  ✓ Basic text detection (with proper methods)")
        print("  ✓ Powerful for complex image operations")
        print("  ✗ Steeper learning curve than Pillow")
    
    if skimage_installed:
        print("\nscikit-image:")
        print("  ✓ Scientific/research-oriented image processing")
        print("  ✓ Good for filters and transformations")
        print("  ✗ Limited text handling capabilities")
        print("  ✓ Strong integration with scientific Python stack")
    
    if pytesseract_installed:
        print("\nPytesseract:")
        print("  ✓ Text detection and recognition (OCR)")
        print("  ✓ Works with Tesseract OCR engine")
        print("  ✓ Good for extracting text content")
    
    if easyocr_installed:
        print("\nEasyOCR:")
        print("  ✓ Modern deep learning-based OCR")
        print("  ✓ Multi-language support")
        print("  ✓ Good accuracy for complex text scenarios")
    
    # Installation suggestions
    print("\n=== Installation Suggestions ===\n")
    
    if not pillow_installed:
        print("Install Pillow: pip install Pillow")
    
    if not cv2_installed:
        print("Install OpenCV: pip install opencv-python")
    
    if not skimage_installed:
        print("Install scikit-image: pip install scikit-image")
    
    if not pytesseract_installed:
        print("Install pytesseract: pip install pytesseract")
        print("Note: Also requires Tesseract OCR engine installed on your system")
    
    if not easyocr_installed:
        print("Install EasyOCR: pip install easyocr")

if __name__ == "__main__":
    main() 
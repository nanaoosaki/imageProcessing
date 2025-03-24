"""
Image Processor Demo

This script demonstrates the capabilities of the image processing application
by running several operations on sample images.
"""
import os
import sys
import time

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our utilities
from src.utils.image_utils import load_image, save_image
from src.utils.text_detection import detect_text_tesseract, detect_text_easyocr, highlight_text_regions
from src.utils.text_removal import remove_text
from src.utils.text_addition import add_text
from src.utils.file_io import INPUT_DIR, OUTPUT_DIR, SAMPLES_DIR
from scripts.generate_samples import generate_text_samples


def run_demo():
    """Run a demonstration of all image processing capabilities."""
    print("\n" + "="*50)
    print("Image Processor Demonstration")
    print("="*50 + "\n")
    
    # Step 1: Check if sample images exist, generate if not
    if not os.path.exists(os.path.join(INPUT_DIR, "sample_plain_text.png")):
        print("Generating sample images...")
        generate_text_samples()
    else:
        print("Using existing sample images in input/ directory")
    
    # Get input files
    input_files = [
        os.path.join(INPUT_DIR, "sample_plain_text.png"),
        os.path.join(INPUT_DIR, "sample_mixed_text.png"),
        os.path.join(INPUT_DIR, "sample_complex_background.png")
    ]
    
    # Step 2: Text detection with EasyOCR and highlight
    print("\n" + "-"*50)
    print("Demonstration 1: Text Detection with EasyOCR")
    print("-"*50)
    
    sample_path = input_files[0]  # Use the plain text sample
    print(f"Processing: {os.path.basename(sample_path)}")
    
    # Load the image
    pil_image, cv_image = load_image(sample_path)
    
    # Detect text using EasyOCR
    print("Detecting text with EasyOCR...")
    easyocr_regions = detect_text_easyocr(cv_image, min_confidence=0.5)
    
    print(f"Detected {len(easyocr_regions)} text regions:")
    for i, region in enumerate(easyocr_regions):
        print(f"  Text {i+1}: '{region['text']}' (Confidence: {region['confidence']:.2f})")
    
    # Highlight the detected text
    print("Highlighting detected text...")
    highlighted_image = highlight_text_regions(cv_image, easyocr_regions, color=(0, 255, 0))
    highlighted_output = os.path.join(OUTPUT_DIR, "demo_detection_easyocr.png")
    save_image(highlighted_image, highlighted_output, format="CV")
    print(f"Saved result to: {highlighted_output}")
    
    # Step 3: Text removal with inpainting
    print("\n" + "-"*50)
    print("Demonstration 2: Text Removal with Inpainting")
    print("-"*50)
    
    sample_path = input_files[1]  # Use the mixed text sample
    print(f"Processing: {os.path.basename(sample_path)}")
    
    # Load the image
    pil_image, cv_image = load_image(sample_path)
    
    # Remove text using inpainting
    print("Removing text with inpainting...")
    inpainted_image = remove_text(
        cv_image,
        method="inpaint",
        detection_method="easyocr",
        min_confidence=0.5,
        radius=10
    )
    
    inpainted_output = os.path.join(OUTPUT_DIR, "demo_removal_inpaint.png")
    save_image(inpainted_image, inpainted_output, format="CV")
    print(f"Saved result to: {inpainted_output}")
    
    # Step 4: Text removal with content-aware fill
    print("\n" + "-"*50)
    print("Demonstration 3: Text Removal with Content-Aware Fill")
    print("-"*50)
    
    # Use the same image as before
    print(f"Processing: {os.path.basename(sample_path)}")
    
    # Remove text using content-aware fill
    print("Removing text with content-aware fill...")
    content_aware_image = remove_text(
        pil_image,
        method="content_aware",
        detection_method="easyocr",
        min_confidence=0.5,
        sample_radius=8
    )
    
    content_aware_output = os.path.join(OUTPUT_DIR, "demo_removal_content_aware.png")
    save_image(content_aware_image, content_aware_output, format="PIL")
    print(f"Saved result to: {content_aware_output}")
    
    # Step 5: Text addition with PIL
    print("\n" + "-"*50)
    print("Demonstration 4: Text Addition with PIL")
    print("-"*50)
    
    sample_path = input_files[2]  # Use the complex background sample
    print(f"Processing: {os.path.basename(sample_path)}")
    
    # Load the image
    pil_image, cv_image = load_image(sample_path)
    
    # Add text with PIL
    print("Adding text with PIL...")
    pil_text_image = add_text(
        pil_image,
        "Text Added with PIL",
        library="pil",
        position="top",
        font_size=40,
        color=(255, 0, 0),
        background=(240, 240, 240, 200),
        padding=10,
        opacity=0.9
    )
    
    pil_text_output = os.path.join(OUTPUT_DIR, "demo_text_addition_pil.png")
    save_image(pil_text_image, pil_text_output, format="PIL")
    print(f"Saved result to: {pil_text_output}")
    
    # Step 6: Text addition with OpenCV
    print("\n" + "-"*50)
    print("Demonstration 5: Text Addition with OpenCV")
    print("-"*50)
    
    # Use the same image as before
    print(f"Processing: {os.path.basename(sample_path)}")
    
    # Add text with OpenCV
    print("Adding text with OpenCV...")
    cv_text_image = add_text(
        cv_image,
        "Text Added with OpenCV",
        library="cv2",
        position="bottom",
        font_size=40,
        color=(0, 0, 255),  # BGR in OpenCV
        background=(240, 240, 240),
        padding=10,
        opacity=0.9
    )
    
    cv_text_output = os.path.join(OUTPUT_DIR, "demo_text_addition_cv2.png")
    save_image(cv_text_image, cv_text_output, format="CV")
    print(f"Saved result to: {cv_text_output}")
    
    # Final summary
    print("\n" + "="*50)
    print("Demonstration Complete!")
    print("="*50)
    print("\nAll processed images have been saved to the output/ directory:")
    print(f"  - {os.path.basename(highlighted_output)}")
    print(f"  - {os.path.basename(inpainted_output)}")
    print(f"  - {os.path.basename(content_aware_output)}")
    print(f"  - {os.path.basename(pil_text_output)}")
    print(f"  - {os.path.basename(cv_text_output)}")
    print("\nThank you for using the Image Processor!")


if __name__ == "__main__":
    start_time = time.time()
    run_demo()
    end_time = time.time()
    print(f"\nDemo completed in {end_time - start_time:.2f} seconds") 
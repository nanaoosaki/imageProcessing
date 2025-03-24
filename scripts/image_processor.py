"""
Image Processor Application

A tool for processing images with text detection, removal, and addition capabilities.
"""
import os
import sys
import argparse
from datetime import datetime
from PIL import Image

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our utility modules
from src.utils.image_utils import load_image, save_image
from src.utils.text_detection import detect_text_tesseract, detect_text_easyocr, highlight_text_regions
from src.utils.text_removal import remove_text
from src.utils.text_addition import add_text
from src.utils.file_io import INPUT_DIR, OUTPUT_DIR, SAMPLES_DIR, list_input_files, get_output_path, get_timestamped_output_path


def process_image(
    input_path,
    output_path=None,
    operation=None,
    detection_method="easyocr",
    removal_method="inpaint",
    min_confidence=0.5,
    text_to_add=None,
    font=None,
    font_size=30,
    position="center",
    text_color=(0, 0, 0),
    background_color=None,
    **kwargs
):
    """
    Process an image with the specified operation.
    
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path where the output image will be saved.
        operation (str): Operation to perform - 'detect', 'remove', 'add', or 'highlight'.
        detection_method (str): Text detection method - 'tesseract' or 'easyocr'.
        removal_method (str): Text removal method - 'inpaint' or 'content_aware'.
        min_confidence (float): Minimum confidence for text detection.
        text_to_add (str): Text to add to the image (for 'add' operation).
        font: Font specification (path for PIL, face for OpenCV).
        font_size (int): Font size.
        position (tuple or str): Position or predefined position for text.
        text_color (tuple): Color for added text.
        background_color (tuple): Background color for added text.
        **kwargs: Additional parameters for the specific operations.
    
    Returns:
        str: Path to the output image.
    """
    # Create default output path if not provided
    if output_path is None:
        if operation:
            prefix = operation
            output_path = get_timestamped_output_path(input_path, prefix)
        else:
            output_path = get_output_path(input_path)
    
    print(f"Processing image: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Load the image
    try:
        pil_image, cv_image = load_image(input_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Determine which operation to perform
    if operation == "detect" or operation == "highlight":
        # Detect text
        if detection_method.lower() == "tesseract":
            regions = detect_text_tesseract(pil_image, min_confidence=min_confidence)
            print(f"Tesseract detected {len(regions)} text regions")
        else:  # Default to EasyOCR
            regions = detect_text_easyocr(cv_image, min_confidence=min_confidence)
            print(f"EasyOCR detected {len(regions)} text regions")
        
        # Print detected text
        for i, region in enumerate(regions):
            print(f"  Text {i+1}: '{region['text']}' (Confidence: {region['confidence']:.2f})")
        
        if operation == "highlight":
            # Highlight detected text regions
            result = highlight_text_regions(cv_image, regions)
            save_image(result, output_path, format="CV")
        else:
            # Just return the original image for 'detect' operation
            save_image(pil_image, output_path, format="PIL")
    
    elif operation == "remove":
        # Remove text
        result = remove_text(
            cv_image if detection_method.lower() == "easyocr" else pil_image,
            method=removal_method,
            detection_method=detection_method,
            min_confidence=min_confidence,
            **kwargs
        )
        
        # Save the result
        if removal_method.lower() == "inpaint":
            save_image(result, output_path, format="CV")
        else:
            save_image(result, output_path, format="PIL")
    
    elif operation == "add":
        # Add text
        if not text_to_add:
            print("Error: No text provided to add")
            return None
        
        # Determine which library to use (PIL by default)
        library = kwargs.get("library", "pil")
        
        # Add text to the image
        result = add_text(
            pil_image if library.lower() == "pil" else cv_image,
            text_to_add,
            library=library,
            position=position,
            font=font,
            font_size=font_size,
            color=text_color,
            background=background_color,
            **kwargs
        )
        
        # Save the result
        save_image(result, output_path, format=library.upper())
    
    else:
        print(f"Unknown operation: {operation}")
        return None
    
    print(f"Processing complete. Result saved to: {output_path}")
    return output_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Image Processor for text detection, removal, and addition")
    
    # Input and output paths
    parser.add_argument("--input", type=str, help="Path to input image or directory")
    parser.add_argument("--output", type=str, help="Path for output image")
    
    # Operation selection
    parser.add_argument(
        "--operation", type=str, choices=["detect", "highlight", "remove", "add"],
        help="Operation to perform: detect, highlight, remove, or add text"
    )
    
    # Text detection options
    parser.add_argument(
        "--detection", type=str, default="easyocr", choices=["tesseract", "easyocr"],
        help="Text detection method (default: easyocr)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum confidence threshold for text detection (0-1 for easyocr, 0-100 for tesseract)"
    )
    
    # Text removal options
    parser.add_argument(
        "--removal", type=str, default="inpaint", choices=["inpaint", "content_aware"],
        help="Text removal method (default: inpaint)"
    )
    parser.add_argument(
        "--radius", type=int, default=10,
        help="Radius for inpainting (for 'inpaint' method)"
    )
    parser.add_argument(
        "--sample-radius", type=int, default=5,
        help="Sample radius for content-aware fill (for 'content_aware' method)"
    )
    
    # Text addition options
    parser.add_argument("--text", type=str, help="Text to add to the image")
    parser.add_argument(
        "--position", type=str, default="center",
        choices=["center", "top", "bottom", "topleft", "topright", "bottomleft", "bottomright"],
        help="Position for added text (default: center)"
    )
    parser.add_argument("--font", type=str, help="Path to font file (for PIL) or font face (for OpenCV)")
    parser.add_argument("--font-size", type=int, default=30, help="Font size (default: 30)")
    parser.add_argument(
        "--color", type=str, default="0,0,0",
        help="Text color in R,G,B format (default: '0,0,0' for black)"
    )
    parser.add_argument(
        "--background", type=str,
        help="Background color in R,G,B format (default: None for transparent)"
    )
    parser.add_argument(
        "--opacity", type=float, default=1.0,
        help="Text opacity from 0.0 to 1.0 (default: 1.0)"
    )
    parser.add_argument(
        "--padding", type=int, default=5,
        help="Padding around text for background (default: 5)"
    )
    parser.add_argument(
        "--library", type=str, default="pil", choices=["pil", "cv2"],
        help="Library to use for text addition (default: pil)"
    )
    
    # List available input files
    parser.add_argument(
        "--list", action="store_true",
        help="List available input files and exit"
    )
    
    # Generate sample images
    parser.add_argument(
        "--generate-samples", action="store_true",
        help="Generate sample images and exit"
    )
    
    return parser.parse_args()


def main():
    """Main application function."""
    args = parse_arguments()
    
    # List available input files if requested
    if args.list:
        print("Available input files:")
        input_files = list_input_files()
        for i, file_path in enumerate(input_files):
            print(f"  {i+1}. {os.path.basename(file_path)}")
        return
    
    # Generate sample images if requested
    if args.generate_samples:
        print("Generating sample images...")
        # Import the sample generator function
        from scripts.generate_samples import generate_text_samples
        generate_text_samples()
        return
    
    # Ensure input path is provided
    if not args.input:
        print("Error: Input path is required")
        return
    
    # Check if input path is a directory or file
    if os.path.isdir(args.input):
        # Process all files in directory
        input_files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                      if os.path.isfile(os.path.join(args.input, f)) and
                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    else:
        # Process single file
        input_files = [args.input]
    
    # Validate we have files to process
    if not input_files:
        print("Error: No valid input files found")
        return
    
    # Prepare additional kwargs for process_image
    kwargs = {}
    
    # Text removal options
    if args.operation == "remove":
        kwargs["radius"] = args.radius
        kwargs["sample_radius"] = args.sample_radius
    
    # Text addition options
    if args.operation == "add":
        if not args.text:
            print("Error: --text is required for 'add' operation")
            return
        
        # Parse color
        if args.color:
            try:
                kwargs["text_color"] = tuple(map(int, args.color.split(',')))
            except ValueError:
                print("Error: Color must be in R,G,B format (e.g., '255,0,0')")
                return
        
        # Parse background color
        if args.background:
            try:
                kwargs["background_color"] = tuple(map(int, args.background.split(',')))
            except ValueError:
                print("Error: Background color must be in R,G,B format (e.g., '240,240,240')")
                return
        
        kwargs["opacity"] = args.opacity
        kwargs["padding"] = args.padding
        kwargs["library"] = args.library
    
    # Process each input file
    for input_path in input_files:
        # Determine output path
        if args.output and len(input_files) == 1:
            # Use provided output path for single file
            output_path = args.output
        else:
            # Generate output path based on input filename
            if args.operation:
                output_path = get_output_path(input_path, f"_{args.operation}")
            else:
                output_path = get_output_path(input_path)
        
        # Process the image
        process_image(
            input_path,
            output_path,
            operation=args.operation,
            detection_method=args.detection,
            removal_method=args.removal,
            min_confidence=args.confidence,
            text_to_add=args.text,
            font=args.font,
            font_size=args.font_size,
            position=args.position,
            **kwargs
        )


if __name__ == "__main__":
    main() 
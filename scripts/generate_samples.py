"""
Sample Generator Script

This script generates sample images with text for testing
the text detection and removal capabilities of our application.
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Add src directory to path to import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.image_utils import create_sample_image
from src.utils.file_io import INPUT_DIR, SAMPLES_DIR

def generate_text_samples():
    """
    Generate a variety of sample images with text for testing.
    Places images in both the input and samples directories.
    """
    print("Generating sample images with text...")
    
    # Sample 1: Simple text on plain background
    sample1_path = os.path.join(INPUT_DIR, "sample_plain_text.png")
    sample1 = create_sample_image(
        sample1_path,
        width=800,
        height=600,
        text="This is sample text for removal",
        font_size=30
    )
    print(f"  - Created {sample1_path}")
    
    # Sample 2: Text with different sizes and positions
    sample2_path = os.path.join(INPUT_DIR, "sample_mixed_text.png")
    
    # Create a blank image
    img = Image.new('RGB', (800, 600), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Try to use a standard font
    try:
        small_font = ImageFont.truetype("arial.ttf", 20)
        medium_font = ImageFont.truetype("arial.ttf", 30)
        large_font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        # Fall back to default font
        small_font = ImageFont.load_default()
        medium_font = ImageFont.load_default()
        large_font = ImageFont.load_default()
    
    # Add background elements
    draw.rectangle([(50, 50), (750, 550)], outline=(200, 200, 200), width=2)
    draw.ellipse([(150, 150), (350, 250)], fill=(230, 230, 230), outline=(200, 0, 0))
    draw.polygon([(500, 150), (600, 200), (550, 300)], fill=(230, 230, 230), outline=(0, 200, 0))
    
    # Add text samples in different sizes and positions
    text_samples = [
        {"text": "Small Text", "position": (200, 100), "font": small_font, "color": (0, 0, 0)},
        {"text": "Medium Text", "position": (400, 200), "font": medium_font, "color": (50, 50, 150)},
        {"text": "Large Text", "position": (300, 300), "font": large_font, "color": (150, 50, 50)},
        {"text": "Another Line", "position": (500, 400), "font": medium_font, "color": (50, 150, 50)},
        {"text": "Final Text", "position": (200, 500), "font": small_font, "color": (100, 100, 100)}
    ]
    
    for text_sample in text_samples:
        draw.text(
            text_sample["position"],
            text_sample["text"],
            font=text_sample["font"],
            fill=text_sample["color"]
        )
    
    # Save the image
    img.save(sample2_path)
    print(f"  - Created {sample2_path}")
    
    # Sample 3: Text on complex background (gradients and patterns)
    sample3_path = os.path.join(INPUT_DIR, "sample_complex_background.png")
    
    # Create gradient background
    img = Image.new('RGB', (800, 600), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Create a simple gradient background
    for y in range(600):
        r = int(255 * (1 - y / 600))
        g = int(200 * (y / 600))
        b = int(100 + 100 * (y / 600))
        for x in range(800):
            draw.point((x, y), fill=(r, g, b))
    
    # Add some pattern elements
    for i in range(0, 800, 50):
        draw.line([(i, 0), (i, 600)], fill=(255, 255, 255, 128), width=1)
    for i in range(0, 600, 50):
        draw.line([(0, i), (800, i)], fill=(255, 255, 255, 128), width=1)
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text((250, 250), "Text on complex background", font=font, fill=(255, 255, 255))
    draw.text((200, 350), "This should be harder to remove", font=font, fill=(240, 240, 240))
    
    # Save the image
    img.save(sample3_path)
    print(f"  - Created {sample3_path}")
    
    # Copy samples to samples directory as well
    for src_path, filename in [
        (sample1_path, "sample_plain_text.png"),
        (sample2_path, "sample_mixed_text.png"),
        (sample3_path, "sample_complex_background.png")
    ]:
        dst_path = os.path.join(SAMPLES_DIR, filename)
        img = Image.open(src_path)
        img.save(dst_path)
        print(f"  - Copied to {dst_path}")
    
    print("Sample generation complete.")

if __name__ == "__main__":
    generate_text_samples() 
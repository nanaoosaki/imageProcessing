# Image Processor

A Python application for image processing with text detection, removal, and addition capabilities.

## Features

- **Text Detection**: Detect text in images using either Tesseract OCR or EasyOCR
- **Text Removal**: Remove detected text from images using inpainting or content-aware fill
- **Text Addition**: Add customizable text to images with various options for font, size, color, position, etc.
- **Batch Processing**: Process multiple images at once

## Installation

### Prerequisites

- Python 3.6+
- pip (Python package installer)

### Dependencies

The application requires the following Python libraries:
- Pillow (PIL)
- OpenCV (cv2)
- pytesseract (requires Tesseract OCR engine installation)
- EasyOCR
- NumPy

You can install them using the following command:

```bash
pip install pillow opencv-python pytesseract easyocr numpy
```

**Note**: Tesseract OCR requires separate installation:
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- macOS: `brew install tesseract`
- Linux: `apt-get install tesseract-ocr`

## Usage

### Command Line Interface

The application provides a command-line interface for all operations:

```bash
python scripts/image_processor.py --input [INPUT_PATH] --operation [OPERATION] [OPTIONS]
```

#### Basic Operations

1. **Generate Sample Images**:
```bash
python scripts/generate_samples.py
```

2. **List Available Input Files**:
```bash
python scripts/image_processor.py --list
```

3. **Detect Text**:
```bash
python scripts/image_processor.py --input input/image.png --operation detect --detection easyocr
```

4. **Highlight Detected Text**:
```bash
python scripts/image_processor.py --input input/image.png --operation highlight --detection easyocr
```

5. **Remove Text**:
```bash
python scripts/image_processor.py --input input/image.png --operation remove --removal inpaint
```

6. **Add Text**:
```bash
python scripts/image_processor.py --input input/image.png --operation add --text "Hello, World!" --position center
```

#### Advanced Options

- **Text Detection Options**:
  - `--detection [tesseract|easyocr]`: Choose the detection method
  - `--confidence [0.0-1.0]`: Set minimum confidence threshold

- **Text Removal Options**:
  - `--removal [inpaint|content_aware]`: Choose the removal method
  - `--radius [1-20]`: Set radius for inpainting
  - `--sample-radius [1-10]`: Set sample radius for content-aware fill

- **Text Addition Options**:
  - `--position [center|top|bottom|topleft|topright|bottomleft|bottomright]`: Set text position
  - `--font [path/to/font.ttf]`: Specify font file
  - `--font-size [size]`: Set font size
  - `--color "R,G,B"`: Set text color (e.g., "255,0,0" for red)
  - `--background "R,G,B"`: Set background color
  - `--opacity [0.0-1.0]`: Set text opacity
  - `--padding [pixels]`: Set padding around text
  - `--library [pil|cv2]`: Choose rendering library

### Examples

1. **Detect and Highlight Text**:
```bash
python scripts/image_processor.py --input input/sample.png --operation highlight --detection easyocr
```

2. **Remove Text with Inpainting**:
```bash
python scripts/image_processor.py --input input/sample.png --operation remove --removal inpaint --radius 15
```

3. **Add Red Text at Top with Background**:
```bash
python scripts/image_processor.py --input input/sample.png --operation add --text "Sample Title" --position top --color "255,0,0" --background "240,240,240" --font-size 40
```

### User-Guided Text Removal (Recommended Workflow)

Our improved text removal workflow gives you more control over which text regions to remove:

1. **Detect and Visualize Text Regions**:
```bash
python scripts/simplified_text_removal.py detect input/image.png
```
This will create a visualization with numbered boxes in `output/image_regions.png`.

2. **Selectively Remove Text Regions**:
```bash
python scripts/simplified_text_removal.py remove input/image.png 1 3 5 7
```
Replace the numbers with the specific region IDs you want to remove.

## Directory Structure

- `input/`: Place input images here
- `output/`: Processed images will be saved here
  - `archive/`: Contains previous iterations and temporary files
- `scripts/`: Contains all executable scripts
  - `simplified_text_removal.py`: User-guided text removal script (main script)
  - `demo.py`: Demonstration script for testing all features
  - `image_processor.py`: Main application with command-line interface
  - `generate_samples.py`: Sample image generator
  - `compare_libs.py`: Utility script for comparing library performance
- `src/utils/`: Contains utility modules
  - `image_utils.py`: Basic image operations
  - `text_detection.py`: Text detection utilities
  - `text_removal.py`: Text removal utilities
  - `text_addition.py`: Text addition utilities
  - `file_io.py`: File input/output utilities

## License

This project is available under the MIT License.

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)
- [Pillow](https://python-pillow.org/)

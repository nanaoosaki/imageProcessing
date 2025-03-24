"""
File I/O Utilities Module

This module provides utility functions for file input/output operations
related to our image processing application.
"""
import os
import glob
from datetime import datetime

# Define global constants for input and output paths
INPUT_DIR = os.path.join(os.getcwd(), "input")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
SAMPLES_DIR = os.path.join(os.getcwd(), "samples")

# Ensure directories exist
for directory in [INPUT_DIR, OUTPUT_DIR, SAMPLES_DIR]:
    os.makedirs(directory, exist_ok=True)


def list_input_files(extension=None):
    """
    List all files in the input directory.
    
    Args:
        extension (str, optional): Filter files by extension (e.g., '.png', '.jpg').
        
    Returns:
        list: List of file paths.
    """
    if extension:
        if not extension.startswith("."):
            extension = "." + extension
        pattern = os.path.join(INPUT_DIR, f"*{extension}")
        return glob.glob(pattern)
    else:
        return [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) 
                if os.path.isfile(os.path.join(INPUT_DIR, f))]


def get_output_path(input_path, suffix="_processed"):
    """
    Generate an output file path based on an input file path.
    
    Args:
        input_path (str): Path to the input file.
        suffix (str): Suffix to add to the filename.
        
    Returns:
        str: Path for the output file.
    """
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}{suffix}{ext}"
    return os.path.join(OUTPUT_DIR, output_filename)


def get_timestamped_output_path(input_path, prefix=""):
    """
    Generate an output file path with a timestamp.
    
    Args:
        input_path (str): Path to the input file.
        prefix (str): Prefix to add to the filename.
        
    Returns:
        str: Path for the output file.
    """
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{prefix}_{name}_{timestamp}{ext}" if prefix else f"{name}_{timestamp}{ext}"
    return os.path.join(OUTPUT_DIR, output_filename)


if __name__ == "__main__":
    # Print directory information
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Samples directory: {SAMPLES_DIR}")
    
    # Create a sample input folder structure
    print("\nCreating sample folder structure...")
    for directory in [INPUT_DIR, OUTPUT_DIR, SAMPLES_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"  - {directory} {'exists' if os.path.exists(directory) else 'could not be created'}")
        
    print("\nApplication is ready to process images.") 
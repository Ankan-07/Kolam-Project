import os
import json
import argparse
import cv2
import matplotlib.pyplot as plt

# Import our custom modules from the 'src' package
from src.analyzer import DesignAnalyzer, MultiImageAnalyzer
from src.generator import ArtisticKolamGenerator
from src.converter import PNGtoSVGConverter

# --- Define folder paths for organization ---
INPUT_DIR = "input_kolams"
ANALYSIS_OUTPUT_DIR = "output_analysis"
ARTWORK_OUTPUT_DIR = "output_artworks"

def analyze_kolams():
    """
    Finds all images in the input directory, runs a collective analysis,
    and saves the results to a JSON file.
    """
    print("--- Starting Kolam Analysis ---")
    
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print(f"Error: The '{INPUT_DIR}' directory is missing or empty.")
        print("Please add some Kolam images to it before running the analysis.")
        return

    image_paths = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))]
    
    if not image_paths:
        print(f"No valid images found in '{INPUT_DIR}'.")
        return

    print(f"Found {len(image_paths)} images to analyze.")

    base_analyzer = DesignAnalyzer()
    multi_analyzer = MultiImageAnalyzer(base_analyzer)
    collective_results = multi_analyzer.analyze_multiple_designs(image_paths)

    if 'error' in collective_results:
        print(f"Analysis failed: {collective_results['error']}")
        return

    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, "collective_analysis_results.json")
    with open(output_path, 'w') as f:
        json.dump(collective_results, f, indent=4)
        
    print(f"Analysis complete. Results saved to '{output_path}'")

def generate_artwork():
    """
    Loads the analysis results and generates a new, artistic Kolam design
    in black and white.
    """
    print("\n--- Starting Artistic Kolam Generation ---")
    
    os.makedirs(ARTWORK_OUTPUT_DIR, exist_ok=True)
    analysis_file = os.path.join(ANALYSIS_OUTPUT_DIR, "collective_analysis_results.json")

    if not os.path.exists(analysis_file):
        print(f"Error: Analysis file not found at '{analysis_file}'.")
        print("Please run the 'analyze' command first to generate the design principles.")
        return

    generator = ArtisticKolamGenerator(analysis_file)
    artistic_kolam = generator.generate()

    output_path = os.path.join(ARTWORK_OUTPUT_DIR, "artistic_kolam_final.png")
    cv2.imwrite(output_path, artistic_kolam)
    print(f"Artwork generation complete. Saved to '{output_path}'")
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(artistic_kolam, cv2.COLOR_BGR2RGB))
    plt.title("Generated Artistic Kolam", fontsize=16)
    plt.axis('off')
    plt.show()

def convert_to_svg():
    """
    Converts the final generated PNG artwork into a scalable SVG file.
    """
    png_input_path = os.path.join(ARTWORK_OUTPUT_DIR, "artistic_kolam_final.png")
    svg_output_path = os.path.join(ARTWORK_OUTPUT_DIR, "artistic_kolam_final.svg")

    if not os.path.exists(png_input_path):
        print(f"❌ Error: PNG file not found at '{png_input_path}'.")
        print("Please run the 'generate' command first to create the artwork.")
        return

    try:
        converter = PNGtoSVGConverter(png_input_path)
        converter.convert(svg_output_path)
    except Exception as e:
        print(f"❌ An error occurred during SVG conversion: {e}")

if __name__ == "__main__":
    # This creates the command-line interface for your project
    parser = argparse.ArgumentParser(description="A complete pipeline for Kolam analysis and artistic generation.")
    parser.add_argument('action', choices=['analyze', 'generate', 'convert'], help="Choose an action: 'analyze' input images, 'generate' a new black and white artwork, or 'convert' the artwork to SVG.")
    
    args = parser.parse_args()

    # Execute the chosen action
    if args.action == 'analyze':
        analyze_kolams()
    elif args.action == 'generate':
        generate_artwork()
    elif args.action == 'convert':
        convert_to_svg()


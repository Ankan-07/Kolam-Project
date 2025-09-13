import gradio as gr
import os
import cv2
import numpy as np
from src.analyzer import DesignAnalyzer, MultiImageAnalyzer
from src.generator import ArtisticKolamGenerator

INPUT_DIR = "input_kolams"
ANALYSIS_OUTPUT_DIR = "output_analysis"
ARTWORK_OUTPUT_DIR = "output_artworks"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTWORK_OUTPUT_DIR, exist_ok=True)

def analyze_and_generate(images):
    if len(images) < 3:
        return None, "Please upload at least 3 Kolam images."
    # images is a list of file paths
    image_paths = []
    for idx, img_path in enumerate(images):
        # Copy the file to input_kolams with a new name
        ext = os.path.splitext(img_path)[1]
        dest_path = os.path.join(INPUT_DIR, f"uploaded_{idx}{ext}")
        with open(img_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())
        image_paths.append(dest_path)
    # Analyze
    base_analyzer = DesignAnalyzer()
    multi_analyzer = MultiImageAnalyzer(base_analyzer)
    collective_results = multi_analyzer.analyze_multiple_designs(image_paths)
    # Save analysis
    analysis_file = os.path.join(ANALYSIS_OUTPUT_DIR, "collective_analysis_results.json")
    import json
    with open(analysis_file, 'w') as f:
        json.dump(collective_results, f, indent=4)
    # Generate new kolam
    generator = ArtisticKolamGenerator(analysis_file)
    artistic_kolam = generator.generate()
    output_path = os.path.join(ARTWORK_OUTPUT_DIR, "artistic_kolam_final.png")
    cv2.imwrite(output_path, artistic_kolam)
    return output_path, "Artistic Kolam generated successfully!"

def check_if_kolam(image):
    # Dummy check: Use DesignAnalyzer to check if image is a kolam
    # (Replace with your own logic if needed)
    # image is a numpy array (from gr.Image)
    temp_path = os.path.join(INPUT_DIR, "check_kolam.png")
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    analyzer = DesignAnalyzer()
    result = analyzer.is_kolam(temp_path) if hasattr(analyzer, 'is_kolam') else None
    if result is None:
        return "Kolam check function not implemented."
    return "This is a Kolam!" if result else "This is NOT a Kolam."

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="pink")) as demo:
    gr.Markdown("# Kolam Analyzer & Generator\nUpload at least 3 Kolam images to analyze and generate a new one.")
    with gr.Row():
        image_input = gr.File(label="Upload Kolam Images", file_count="multiple", type="filepath", file_types=[".png", ".jpg", ".jpeg"])
        analyze_btn = gr.Button("Analyze & Generate Kolam")
    output_image = gr.Image(label="Generated Artistic Kolam")
    status = gr.Textbox(label="Status Message")
    analyze_btn.click(fn=analyze_and_generate, inputs=image_input, outputs=[output_image, status])
    gr.Markdown("---")
    gr.Markdown("## Check if Image is a Kolam")
    check_image = gr.Image(label="Upload Image to Check")
    check_btn = gr.Button("Check if Kolam")
    check_result = gr.Textbox(label="Kolam Check Result")
    check_btn.click(fn=check_if_kolam, inputs=check_image, outputs=check_result)

demo.launch(share=True)

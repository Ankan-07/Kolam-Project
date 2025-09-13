import gradio as gr
import os
import cv2
import numpy as np
from src.analyzer import DesignAnalyzer, MultiImageAnalyzer
from src.generator import ArtisticKolamGenerator

INPUT_DIR = "input_kolams"
ANALYSIS_OUTPUT_DIR = "output_analysis"
ARTWORK_OUTPUT_DIR = "output_artworks"
GENERATED_KOLAM_PATH = os.path.join(ARTWORK_OUTPUT_DIR, "artistic_kolam_final.png")

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

def is_symmetric(img, threshold=0.85):
    # Check vertical and horizontal symmetry
    height, width = img.shape[:2]
    left = img[:, :width//2]
    right = cv2.flip(img[:, width//2:], 1)
    right = cv2.resize(right, (left.shape[1], left.shape[0]))
    
    # Compare left and right halves
    similarity = cv2.matchTemplate(left, right, cv2.TM_CCOEFF_NORMED)[0][0]
    return similarity > threshold

def has_dot_pattern(img, min_dots=5):
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Detect dots using blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 100
    params.filterByCircularity = True
    params.minCircularity = 0.7
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    
    return len(keypoints) >= min_dots

def check_if_kolam():
    if not os.path.exists(GENERATED_KOLAM_PATH):
        return "No generated Kolam found. Please generate a Kolam first."
    
    # Read the image
    img = cv2.imread(GENERATED_KOLAM_PATH)
    if img is None:
        return "Error reading the generated Kolam image."
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Check kolam characteristics
    checks = {
        "Symmetry": is_symmetric(binary),
        "Dot Pattern": has_dot_pattern(binary)
    }
    
    # Calculate confidence score
    confidence = sum(1 for check in checks.values() if check) / len(checks)
    
    if confidence >= 0.5:
        details = "\n".join([f"- {k}: {'✓' if v else '✗'}" for k, v in checks.items()])
        return f"This appears to be a Kolam! (Confidence: {confidence*100:.1f}%)\n\nDetails:\n{details}"
    else:
        return f"This doesn't appear to be a valid Kolam. (Confidence: {confidence*100:.1f}%)"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="pink")) as demo:
    gr.Markdown("# Kolam Analyzer & Generator\nUpload at least 3 Kolam images to analyze and generate a new one.")
    with gr.Row():
        image_input = gr.File(label="Upload Kolam Images", file_count="multiple", type="filepath", file_types=[".png", ".jpg", ".jpeg"])
        analyze_btn = gr.Button("Analyze & Generate Kolam")
    output_image = gr.Image(label="Generated Artistic Kolam")
    status = gr.Textbox(label="Status Message")
    analyze_btn.click(fn=analyze_and_generate, inputs=image_input, outputs=[output_image, status])
    gr.Markdown("---")
    gr.Markdown("## Check if Generated Kolam is Valid")
    check_btn = gr.Button("Check Generated Kolam")
    check_result = gr.Textbox(label="Kolam Check Result")
    check_btn.click(fn=check_if_kolam, inputs=None, outputs=check_result)

demo.launch(share=True)

import cv2
import numpy as np

class PNGtoSVGConverter:
    """
    Converts a rasterized Kolam PNG artwork into a scalable SVG format
    by tracing its distinct colored regions and lines.
    """

    def __init__(self, image_path: str):
        """
        Initializes the converter with the path to the input PNG image.
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image from '{image_path}'")
        self.height, self.width, _ = self.image.shape

    def convert(self, output_svg_path: str):
        """
        Orchestrates the conversion process and saves the final SVG file.
        """
        print("\n--- Starting PNG to SVG Conversion ---")
        print("   - Tracing colored regions and lines...")
        regions = self._extract_colored_regions()
        print("   - Assembling SVG file...")
        svg_content = self._generate_svg_content(regions)

        with open(output_svg_path, 'w') as f:
            f.write(svg_content)
        
        print(f"✅ SVG conversion complete. File saved to '{output_svg_path}'")

    def _extract_colored_regions(self) -> dict:
        """
        Finds all unique colors in the image and traces the contours of each
        colored area.
        """
        regions = {}
        pixels = self.image.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)

        for color_bgr in unique_colors:
            if np.mean(color_bgr) > 230: # Ignore background
                continue

            mask = cv2.inRange(self.image, color_bgr, color_bgr)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            hex_color = "#{:02x}{:02x}{:02x}".format(color_bgr[2], color_bgr[1], color_bgr[0])
            regions[hex_color] = contours
            
        return regions

    def _contour_to_svg_path(self, contour: np.ndarray) -> str:
        """Converts an OpenCV contour into an SVG path data string."""
        # Handle different contour shapes safely
        if contour.shape[0] == 0:
            return ""
            
        # Only squeeze if the contour has the right dimensions
        if len(contour.shape) > 2:
            points = contour.squeeze()
        else:
            points = contour
            
        if len(points) == 0:
            return ""
        
        # Ensure points have the right shape for indexing
        if len(points.shape) == 1 and points.shape[0] >= 2:
            # Single point case
            path_data = f"M {points[0]},{points[1]} "
            path_data += " Z"
        else:
            # Multiple points case
            path_data = f"M {points[0][0]},{points[0][1]} "
            path_data += " ".join([f"L {p[0]},{p[1]}" for p in points[1:]])
            path_data += " Z"
            
        return path_data

    def _generate_svg_content(self, regions: dict) -> str:
        """Assembles the final SVG file as a single string."""
        svg_header = (
            f'<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">\n'
            '  <rect width="100%" height="100%" fill="white"/>\n'
        )
        svg_paths = ""
        for color, contours in regions.items():
            for contour in contours:
                path_data = self._contour_to_svg_path(contour)
                svg_paths += f'  <path d="{path_data}" fill="{color}" stroke="{color}" stroke-width="0.5"/>\n'

        return svg_header + svg_paths + "</svg>"


import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import random
import json
import math
from typing import List, Tuple, Dict, Any, Optional

class ArtisticKolamGenerator:
    """
    Generates a new, artistic, and structurally complex Kolam design
    from a set of design principles, creating a black and white piece of art
    that is highly specific to the input analysis.
    """

    def __init__(self, analysis_path: str):
        """Initializes by loading the design principles from the analysis JSON file."""
        try:
            with open(analysis_path, 'r') as f:
                analysis_data = json.load(f)
            self.principles = analysis_data['collective_principles']
            print("✅ Design principles loaded successfully.")
        except (FileNotFoundError, KeyError):
            print(f"❌ Warning: '{analysis_path}' not found. Using default principles.")
            self.principles = self._get_default_principles()
        self.height, self.width = 1000, 1000

    def generate(self) -> np.ndarray:
        """
        The main method to generate a new, artistic Kolam from scratch.
        Always generates a black and white Kolam for a traditional aesthetic.
        """
        print("\n🎨 Generating new artistic Kolam...")
        
        # Create a white background
        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        # Generate the basic structure based on detailed analysis
        dots, spacing = self._generate_grid()
        structure_mask = self._generate_new_structure(dots, spacing)
        
        # Draw the structure in black
        final_canvas = self._draw_artistic_lines(canvas, structure_mask, line_color=(0, 0, 0))
        
        # Add a simple black border
        border_thickness = int(min(self.width, self.height) * 0.01)
        final_canvas = cv2.rectangle(final_canvas, 
                                   (border_thickness, border_thickness),
                                   (self.width - border_thickness, self.height - border_thickness), 
                                   (0, 0, 0), 
                                   border_thickness)
        
        print("✨ Generation complete!")
        return final_canvas

    def _generate_grid(self) -> tuple:
        """Creates the dot grid based on principles from the JSON file."""
        grid_size = self.principles.get('average_grid_size', [7, 7])
        rows, cols = max(5, grid_size[0]), max(5, grid_size[1])
        margin = int(self.width * 0.2)
        spacing = (self.width - 2 * margin) / (max(rows, cols) - 1)
        
        dots = []
        for r in range(rows):
            for c in range(cols):
                dots.append((margin + c * spacing, margin + r * spacing))
        return np.array(dots), spacing

    def _generate_new_structure(self, dots: np.ndarray, spacing: float) -> np.ndarray:
        """
        This is the creative core. It generates a new, complex pattern based on the detailed analysis.
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        center = (self.width // 2, self.height // 2)
        
        quadrant_dots = [d for d in dots if d[0] <= center[0] + spacing/2 and d[1] <= center[1] + spacing/2]
        
        style = self.principles.get('dominant_style', 'Sikku')
        primary_shapes = self.principles.get('primary_shapes', [])
        complexity = self.principles.get('average_complexity', 35)
        
        path_points = []
        if style == 'Sikku' and self.principles.get('continuity_preference', 0) > 0.5:
            print("   - Generating a 'Sikku' (continuous line) structure...")
            path_points = self._create_weaving_path(quadrant_dots, int(spacing), complexity)
        else:
            print("   - Generating a 'Pulli' (geometric) structure...")
            path_points = self._create_geometric_path_from_shapes(quadrant_dots, int(spacing), primary_shapes)

        if complexity > 40:
            secondary_paths = self._create_secondary_elements(quadrant_dots, int(spacing), primary_shapes, complexity)
            for path in secondary_paths:
                if len(path) > 2:
                    self._draw_smooth_path(mask, path, is_closed=True)

        if len(path_points) > 2:
            self._draw_smooth_path(mask, path_points, is_closed=False)
        
        mask = self._apply_symmetry(mask, center)
        return mask

    def _draw_smooth_path(self, mask, path_points, is_closed=False):
        x_coords = [p[0] for p in path_points]
        y_coords = [p[1] for p in path_points]
        tck, u = splprep([x_coords, y_coords], s=1000, k=min(3, len(path_points)-1))
        u_new = np.linspace(u.min(), u.max(), len(path_points) * 10)
        x_new, y_new = splev(u_new, tck)
        smooth_path = np.c_[x_new, y_new].astype(np.int32)
        cv2.polylines(mask, [smooth_path], isClosed=is_closed, color=255, thickness=2, lineType=cv2.LINE_AA)

    def _create_weaving_path(self, dots: list, spacing: int, complexity: float) -> list:
        """Creates a continuous, winding path that loops around dots for Sikku style."""
        if not dots: return []
        
        sorted_dots = sorted(dots, key=lambda p: (p[0] + p[1], p[0] - p[1]))
        num_dots = min(len(sorted_dots), max(5, int(complexity / 4)))
        selected_dots = sorted_dots[:num_dots]
        
        path = [selected_dots[0]]
        remaining_dots = set(map(tuple, selected_dots[1:]))
        
        for _ in range(len(selected_dots) * int(complexity/10)):
            if not remaining_dots: break
            current = path[-1]
            
            nearest = min(remaining_dots, key=lambda p: np.linalg.norm(np.array(p) - current))
            remaining_dots.remove(nearest)
            
            midpoint = ((current[0] + nearest[0]) / 2, (current[1] + nearest[1]) / 2)
            curve_intensity = random.uniform(0.5, 0.9) * (complexity / 20)
            
            control_point = (midpoint[0] + (current[1] - nearest[1]) * curve_intensity, 
                             midpoint[1] + (nearest[0] - current[0]) * curve_intensity)
            
            path.extend([control_point, nearest])
        
        return path

    def _create_geometric_path_from_shapes(self, dots: list, spacing: int, shapes: list) -> list:
        """Creates a geometric path based on the dominant shapes from the analysis."""
        if not dots: return []
        
        path = []
        if 'circle' in shapes:
            print("     - Incorporating circular elements.")
            path.extend(self._create_circular_path(dots, spacing))
        if 'rectangle' in shapes or 'triangle' in shapes:
            print("     - Incorporating linear elements.")
            path.extend(self._create_linear_path(dots))
        
        if not path:
            return self._create_linear_path(dots)
            
        return path

    def _create_circular_path(self, dots: list, spacing: int) -> list:
        if len(dots) < 3: return []
        center_dot = random.choice(dots)
        radius = spacing * random.uniform(0.8, 1.5)
        return [(center_dot[0] + radius * math.cos(a), center_dot[1] + radius * math.sin(a)) for a in np.linspace(0, 2 * np.pi, 15)]

    def _create_linear_path(self, dots: list) -> list:
        if len(dots) < 2: return []
        return random.sample(dots, min(len(dots), 4))

    def _create_secondary_elements(self, dots: list, spacing: int, shapes: list, complexity: float) -> List[List[Tuple[float, float]]]:
        """Creates additional geometric elements based on shapes and complexity."""
        if len(dots) < 4: return []
        
        secondary_paths = []
        num_elements = int(complexity / 20)

        for _ in range(num_elements):
            shape_choice = random.choice(shapes) if shapes else 'circle'
            
            if shape_choice == 'circle':
                path = self._create_circular_path(dots, spacing)
                secondary_paths.append(path)
            elif shape_choice == 'rectangle':
                dot1, dot2 = random.sample(dots, 2)
                path = [dot1, (dot1[0], dot2[1]), dot2, (dot2[0], dot1[1])]
                secondary_paths.append(path)
            elif shape_choice == 'triangle':
                path = random.sample(dots, 3)
                secondary_paths.append(path)

        return secondary_paths

    def _apply_symmetry(self, mask: np.ndarray, center: tuple) -> np.ndarray:
        """Applies symmetry based on the principles from the analysis."""
        symmetries = self.principles.get('common_symmetries', {})
        
        if 'rotational' in symmetries:
            for angle in symmetries['rotational']:
                if not isinstance(angle, (int, float)):
                    continue
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(mask, M, (self.width, self.height))
                mask = cv2.bitwise_or(mask, rotated)
        
        if 'reflective' in symmetries:
            if 'horizontal' in symmetries['reflective']:
                mask = cv2.bitwise_or(mask, cv2.flip(mask, 1))
            if 'vertical' in symmetries['reflective']:
                mask = cv2.bitwise_or(mask, cv2.flip(mask, 0))
        
        return mask

    def _draw_artistic_lines(self, canvas: np.ndarray, structure_mask: np.ndarray, line_color: tuple):
        """Draws artistic lines for the Kolam pattern."""
        contours, _ = cv2.findContours(structure_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return canvas
        
        cv2.drawContours(canvas, contours, -1, line_color, thickness=2, lineType=cv2.LINE_AA)
        return canvas

    def _get_default_principles(self) -> dict:
        """Provides fallback principles."""
        return {
            "average_grid_size": [7, 7], 
            "common_symmetries": {'rotational': [90], 'reflective': ['horizontal']},
            "dominant_style": "Pulli", 
            "continuity_preference": 0.2,
            "primary_shapes": ['circle'],
            "average_complexity": 30
        }
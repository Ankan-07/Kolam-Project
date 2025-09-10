import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import random
import json

class ArtisticKolamGenerator:
    """
    Generates a completely new, artistic, and structurally complex Kolam design
    from a set of design principles, creating a ready-to-use piece of art.
    """
    PALETTES = {
        "earthy": [(217, 185, 155), (166, 124, 82), (115, 76, 40)],
        "festival": [(255, 107, 107), (255, 169, 107), (52, 152, 219)],
        "rangoli": [(231, 84, 128), (255, 165, 0), (50, 205, 50), (65, 105, 225)],
        "peacock": [(0, 128, 128), (0, 77, 77), (72, 209, 204), (102, 205, 170)]
    }

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

    def generate(self, palette_name: str = "peacock") -> np.ndarray:
        """
        The main method to generate a new, artistic Kolam from scratch.
        """
        print("\n🎨 Generating new artistic Kolam...")
        dots, spacing = self._generate_grid()
        structure_mask = self._generate_new_structure(dots, spacing)
        canvas = self._create_textured_background()
        filled_canvas, regions = self._fill_regions_with_gradients(canvas, structure_mask, self.PALETTES[palette_name])
        embellished_canvas = self._add_internal_embellishments(filled_canvas, regions, self.PALETTES[palette_name])
        final_canvas = self._draw_artistic_lines(embellished_canvas, structure_mask)
        self._draw_ornate_border(final_canvas, self.PALETTES[palette_name])
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
        This is the creative core. It generates a new, complex pattern based on style.
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        center = (self.width // 2, self.height // 2)
        
        quadrant_dots = [d for d in dots if d[0] <= center[0] + spacing/2 and d[1] <= center[1] + spacing/2]
        
        style = self.principles.get('dominant_style', 'Sikku')
        
        if style == 'Sikku' and self.principles.get('continuity_preference', 0) > 0.6:
            print("   - Generating a new 'Sikku' (continuous line) structure...")
            path_points = self._create_weaving_path(quadrant_dots, int(spacing))
        else:
            print("   - Generating a new 'Pulli' (geometric) structure...")
            path_points = self._create_interconnected_geometric_path(quadrant_dots, int(spacing))

        if len(path_points) > 2:
            x_coords = [p[0] for p in path_points]
            y_coords = [p[1] for p in path_points]
            tck, u = splprep([x_coords, y_coords], s=1000, k=min(3, len(path_points)-1))
            u_new = np.linspace(u.min(), u.max(), len(path_points) * 10)
            x_new, y_new = splev(u_new, tck)
            smooth_path = np.c_[x_new, y_new].astype(np.int32)
            cv2.polylines(mask, [smooth_path], isClosed=False, color=255, thickness=2, lineType=cv2.LINE_AA)
        
        mask = self._apply_symmetry(mask, center)
        return mask

    def _create_weaving_path(self, dots: list, spacing: int) -> list:
        """Creates a single, continuous, winding path that loops around dots."""
        if not dots: return []
        path = [dots[0]]
        remaining_dots = set(map(tuple, dots[1:]))
        for _ in range(len(dots) * 2):
            if not remaining_dots: break
            current = path[-1]
            nearest = min(remaining_dots, key=lambda p: np.linalg.norm(np.array(p) - current))
            remaining_dots.remove(nearest)
            midpoint = ((current[0] + nearest[0]) / 2, (current[1] + nearest[1]) / 2)
            control_point = (midpoint[0] + (current[1] - nearest[1]) * 0.6, midpoint[1] + (nearest[0] - current[0]) * 0.6)
            path.extend([control_point, nearest])
        return path

    def _create_interconnected_geometric_path(self, dots: list, spacing: int) -> list:
        """Creates an interconnected path that forms the base of a geometric Kolam."""
        if len(dots) < 4: return []
        corner_dot = min(dots, key=lambda d: d[0] + d[1])
        edge_dot_x = min(dots, key=lambda d: abs(d[0] - (self.width/2)) + abs(d[1] - corner_dot[1]))
        edge_dot_y = min(dots, key=lambda d: abs(d[1] - (self.height/2)) + abs(d[0] - corner_dot[0]))
        path = [corner_dot]
        mid_arc = ((corner_dot[0] + edge_dot_x[0])/2 + spacing*0.5, (corner_dot[1] + edge_dot_y[1])/2 + spacing*0.5)
        path.extend([mid_arc, edge_dot_x])
        path.extend([(edge_dot_x[0], edge_dot_y[1]), edge_dot_y])
        center_dot = min(dots, key=lambda d: abs(d[0] - self.width/2) + abs(d[1] - self.height/2))
        path.append(center_dot)
        return path

    def _apply_symmetry(self, mask: np.ndarray, center: tuple) -> np.ndarray:
        """Applies 4-fold rotational and reflectional symmetry."""
        for angle in [90, 180, 270]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(mask, M, (self.width, self.height))
            mask = cv2.bitwise_or(mask, rotated)
        mask = cv2.bitwise_or(mask, cv2.flip(mask, 1))
        mask = cv2.bitwise_or(mask, cv2.flip(mask, 0))
        return mask

    def _create_textured_background(self) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), (245, 240, 230), dtype=np.uint8)
        noise = np.random.randint(-10, 10, canvas.shape, dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(canvas, (5, 5), 0)

    def _fill_regions_with_gradients(self, canvas: np.ndarray, binary_mask: np.ndarray, palette: list):
        if not palette: return canvas, []
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None: return canvas, []
        valid_regions = []
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                color1, color2 = random.sample(palette, 2)
                mask = np.zeros((self.height, self.width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                gradient = np.zeros_like(canvas)
                for row in range(self.height):
                    alpha = row / self.height
                    blended_color = tuple(int(c1 * (1-alpha) + c2 * alpha) for c1, c2 in zip(color1, color2))
                    gradient[row, :] = blended_color
                canvas[mask == 255] = gradient[mask == 255]
                valid_regions.append(contour)
        return canvas, valid_regions

    def _add_internal_embellishments(self, canvas: np.ndarray, regions: list, palette: list):
        for region in regions:
            M = cv2.moments(region)
            if M["m00"] == 0: continue
            cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
            motif_size = int(np.sqrt(cv2.contourArea(region)) * 0.1)
            if motif_size < 3: continue
            highlight_color_np = np.clip(np.array(random.choice(palette)) * 1.4, 0, 255)
            highlight_color = tuple(map(int, highlight_color_np))
            cv2.circle(canvas, (cX, cY), motif_size, highlight_color, -1, cv2.LINE_AA)
        return canvas

    def _draw_artistic_lines(self, canvas: np.ndarray, structure_mask: np.ndarray):
        line_color = (255, 250, 245); shadow_color = (40, 30, 20)
        contours, _ = cv2.findContours(structure_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shadow_canvas = np.zeros_like(canvas)
        cv2.polylines(shadow_canvas, contours, isClosed=False, color=shadow_color, thickness=12, lineType=cv2.LINE_AA)
        shadow_canvas = cv2.GaussianBlur(shadow_canvas, (21, 21), 0)
        canvas = cv2.addWeighted(canvas, 1, shadow_canvas, 0.3, 0)
        cv2.polylines(canvas, contours, isClosed=False, color=line_color, thickness=5, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, contours, isClosed=False, color=(50, 40, 30), thickness=2, lineType=cv2.LINE_AA)
        return canvas

    def _draw_ornate_border(self, canvas: np.ndarray, palette: list):
        margin = 15; border_color1 = palette[0]; border_color2 = palette[1]
        cv2.rectangle(canvas, (margin, margin), (self.width - margin, self.height - margin), border_color1, 10)
        cv2.rectangle(canvas, (margin + 12, margin + 12), (self.width - margin - 12, self.height - margin - 12), border_color2, 2)
        for x in [margin, self.width - margin]:
            for y in [margin, self.height - margin]:
                cv2.circle(canvas, (x, y), 12, border_color1, -1); cv2.circle(canvas, (x, y), 6, border_color2, -1)
    
    def _get_default_principles(self) -> dict:
        """Provides fallback principles."""
        return {"average_grid_size": [7, 7], "common_symmetries": [90], "dominant_style": "Pulli", "continuity_preference": 0.2}


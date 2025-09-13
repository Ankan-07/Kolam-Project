import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import random
import json
import math
from typing import List, Tuple, Dict, Any, Optional

class ArtisticKolamGenerator:
    """
    Generates a completely new, artistic, and structurally complex Kolam design
    from a set of design principles, creating a ready-to-use piece of art.
    
    Enhanced with traditional Tamil Kolam patterns, intricate designs, and
    culturally authentic elements for more visually appealing outputs.
    """
    PALETTES = {
        "earthy": [(217, 185, 155), (166, 124, 82), (115, 76, 40), (94, 65, 47)],
        "festival": [(255, 107, 107), (255, 169, 107), (52, 152, 219), (155, 89, 182)],
        "rangoli": [(231, 84, 128), (255, 165, 0), (50, 205, 50), (65, 105, 225), (138, 43, 226)],
        "peacock": [(0, 128, 128), (0, 77, 77), (72, 209, 204), (102, 205, 170), (25, 25, 112)],
        "traditional": [(153, 51, 0), (204, 0, 0), (255, 153, 51), (102, 51, 0), (51, 25, 0)],
        "pongal": [(255, 204, 102), (204, 102, 0), (153, 76, 0), (255, 255, 204), (102, 51, 0)]
    }
    
    # Traditional Tamil Kolam patterns
    TRADITIONAL_PATTERNS = {
        "lotus": {
            "points": [(0, 0), (1, 0), (0.5, 0.866), (-0.5, 0.866), (-1, 0), (-0.5, -0.866), (0.5, -0.866)],
            "connections": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)]
        },
        "star": {
            "points": [(0, 0), (0, 1), (0.588, 0.809), (0.951, 0.309), (0.951, -0.309), (0.588, -0.809), (0, -1), (-0.588, -0.809), (-0.951, -0.309), (-0.951, 0.309), (-0.588, 0.809)],
            "connections": [(0, 1), (0, 3), (0, 5), (0, 7), (0, 9), (1, 4), (1, 7), (2, 5), (2, 8), (3, 6), (3, 9), (4, 7), (4, 10), (5, 8), (6, 9)]
        },
        "chikku": {
            "points": [(0, 0), (1, 1), (2, 0), (1, -1), (0, -2), (-1, -1), (-2, 0), (-1, 1), (0, 2)],
            "connections": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0)]
        },
        "kambi": {
            "points": [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
            "connections": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0)]
        }
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

    def generate(self) -> np.ndarray:
        """
        The main method to generate a new, artistic Kolam from scratch.
        Always generates a black and white Kolam for traditional aesthetic.
        """
        print("\n🎨 Generating new artistic Kolam...")
        
        # Generate the basic structure
        dots, spacing = self._generate_grid()
        structure_mask = self._generate_new_structure(dots, spacing)
        
        # Create a white background
        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        
        # Draw the structure in black
        final_canvas = self._draw_artistic_lines(canvas, structure_mask, line_color=(0, 0, 0))
        
        # Add a simple black border
        border_thickness = int(min(self.width, self.height) * 0.01)  # 1% of size
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
        This is the creative core. It generates a new, complex pattern based on style.
        Enhanced with traditional Tamil Kolam patterns and more intricate designs.
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        center = (self.width // 2, self.height // 2)
        
        quadrant_dots = [d for d in dots if d[0] <= center[0] + spacing/2 and d[1] <= center[1] + spacing/2]
        
        style = self.principles.get('dominant_style', 'Sikku')
        primary_shapes = self.principles.get('primary_shapes', ['lotus', 'stars'])
        complexity = self.principles.get('average_complexity', 35)
        
        # Randomly select pattern generation approach with equal probability
        generation_choice = random.random()
        
        if generation_choice < 0.33:  # Traditional pattern with variations
            print("   - Generating a traditional pattern based on", primary_shapes)
            pattern_name = random.choice(primary_shapes) if primary_shapes else 'lotus'
            # Default to lotus if the specified pattern doesn't exist
            pattern = self.TRADITIONAL_PATTERNS.get(pattern_name, self.TRADITIONAL_PATTERNS['lotus'])
            mask = self._create_traditional_pattern(pattern, center, spacing, mask)
        elif style == 'Sikku' and self.principles.get('continuity_preference', 0) > 0.3:
            print("   - Generating an enhanced 'Sikku' (continuous line) structure...")
            path_points = self._create_enhanced_weaving_path(quadrant_dots, int(spacing), complexity)
        else:
            print("   - Generating an enhanced 'Pulli' (geometric) structure...")
            path_points = self._create_interconnected_geometric_path(quadrant_dots, int(spacing))
            
            # Add additional geometric elements for more complexity
            if complexity > 30:
                secondary_paths = self._create_secondary_geometric_elements(quadrant_dots, int(spacing))
                for path in secondary_paths:
                    if len(path) > 2:
                        x_coords = [p[0] for p in path]
                        y_coords = [p[1] for p in path]
                        tck, u = splprep([x_coords, y_coords], s=1000, k=min(3, len(path)-1))
                        u_new = np.linspace(u.min(), u.max(), len(path) * 10)
                        x_new, y_new = splev(u_new, tck)
                        smooth_path = np.c_[x_new, y_new].astype(np.int32)
                        cv2.polylines(mask, [smooth_path], isClosed=True, color=255, thickness=2, lineType=cv2.LINE_AA)

        # Only process path_points if they exist (not used in traditional patterns)
        if 'path_points' in locals() and len(path_points) > 2:
            x_coords = [p[0] for p in path_points]
            y_coords = [p[1] for p in path_points]
            tck, u = splprep([x_coords, y_coords], s=1000, k=min(3, len(path_points)-1))
            u_new = np.linspace(u.min(), u.max(), len(path_points) * 10)
            x_new, y_new = splev(u_new, tck)
            smooth_path = np.c_[x_new, y_new].astype(np.int32)
            cv2.polylines(mask, [smooth_path], isClosed=False, color=255, thickness=2, lineType=cv2.LINE_AA)
        
        mask = self._apply_symmetry(mask, center)
        return mask
        
    def _create_traditional_pattern(self, pattern: Dict, center: Tuple[int, int], spacing: float, mask: np.ndarray) -> np.ndarray:
        """
        Creates a traditional Tamil Kolam pattern based on the specified pattern template with variations.
        """
        # Add random variation to scale
        scale_variation = random.uniform(0.8, 1.5)
        scale_factor = spacing * 2 * scale_variation
        
        # Add slight rotation for variety
        rotation_angle = random.uniform(-30, 30)
        points = []
        
        # Scale, rotate, and position the pattern points
        for point in pattern["points"]:
            # Apply rotation
            x_rot = point[0] * math.cos(math.radians(rotation_angle)) - point[1] * math.sin(math.radians(rotation_angle))
            y_rot = point[0] * math.sin(math.radians(rotation_angle)) + point[1] * math.cos(math.radians(rotation_angle))
            
            # Scale and position
            x = center[0] + x_rot * scale_factor
            y = center[1] + y_rot * scale_factor
            points.append((int(x), int(y)))
        
        # Draw the connections between points
        for connection in pattern["connections"]:
            start_idx, end_idx = connection
            start_point = points[start_idx]
            end_point = points[end_idx]
            
            # Create multiple control points for more varied curves
            num_control_points = random.randint(1, 3)
            control_points = []
            
            for i in range(num_control_points):
                base_x = start_point[0] + (end_point[0] - start_point[0]) * ((i + 1) / (num_control_points + 1))
                base_y = start_point[1] + (end_point[1] - start_point[1]) * ((i + 1) / (num_control_points + 1))
                
                # Add varied randomness to control points
                offset_x = random.uniform(-0.5, 0.5) * spacing
                offset_y = random.uniform(-0.5, 0.5) * spacing
                
                control_points.append((int(base_x + offset_x), int(base_y + offset_y)))
            
            # Use the middle control point for the bezier curve
            middle_idx = len(control_points) // 2
            control_point = control_points[middle_idx]
            
            # Create a smooth curve using Bezier interpolation
            curve_points = self._bezier_curve([start_point, control_point, end_point], 20)
            curve_points = np.array(curve_points, dtype=np.int32)
            
            cv2.polylines(mask, [curve_points], isClosed=False, color=255, thickness=2, lineType=cv2.LINE_AA)
        
        return mask
    
    def _bezier_curve(self, points: List[Tuple[int, int]], num_points: int) -> List[Tuple[int, int]]:
        """
        Generate a Bezier curve from control points.
        """
        curve_points = []
        for t in np.linspace(0, 1, num_points):
            x = (1-t)**2 * points[0][0] + 2*(1-t)*t * points[1][0] + t**2 * points[2][0]
            y = (1-t)**2 * points[0][1] + 2*(1-t)*t * points[1][1] + t**2 * points[2][1]
            curve_points.append((int(x), int(y)))
        return curve_points
    
    def _create_enhanced_weaving_path(self, dots: list, spacing: int, complexity: float) -> list:
        """
        Creates a more intricate, continuous, winding path that loops around dots.
        Enhanced with varying curve intensities and additional control points for complexity.
        Avoids generating symmetrical four-corner patterns.
        """
        if not dots: return []
        
        # Randomize the pattern type
        pattern_type = random.choice(['spiral', 'meandering', 'asymmetric', 'nested'])
        
        # Different sorting strategies for different pattern types
        if pattern_type == 'spiral':
            center = (self.width // 2, self.height // 2)
            sorted_dots = sorted(dots, key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
        elif pattern_type == 'meandering':
            sorted_dots = sorted(dots, key=lambda p: math.sin(p[0] / spacing) + p[1])
        elif pattern_type == 'asymmetric':
            sorted_dots = sorted(dots, key=lambda p: p[0] + random.uniform(0.8, 1.2) * p[1])
        else:  # nested
            sorted_dots = sorted(dots, key=lambda p: abs(math.sin(p[0] / spacing) * math.cos(p[1] / spacing)))
        
        # Select a subset of dots based on complexity
        num_dots = min(len(sorted_dots), max(5, int(complexity / 5)))
        selected_dots = sorted_dots[:num_dots]
        
        path = [selected_dots[0]]
        remaining_dots = set(map(tuple, selected_dots[1:]))
        
        # Create a more complex path with additional control points
        for _ in range(len(selected_dots) * 3):  # More iterations for complexity
            if not remaining_dots: break
            current = path[-1]
            
            # Find the nearest dot with some randomness for variety
            if random.random() < 0.3:  # Sometimes pick a random dot instead of nearest
                nearest = random.choice(list(remaining_dots))
            else:
                nearest = min(remaining_dots, key=lambda p: np.linalg.norm(np.array(p) - current))
                
            remaining_dots.remove(nearest)
            
            # Create multiple control points for more intricate curves
            midpoint = ((current[0] + nearest[0]) / 2, (current[1] + nearest[1]) / 2)
            
            # Vary the curve intensity based on complexity
            curve_intensity = random.uniform(0.4, 0.8) * (complexity / 30)
            
            # Create two control points for S-shaped curves
            control_point1 = (midpoint[0] + (current[1] - nearest[1]) * curve_intensity, 
                             midpoint[1] + (nearest[0] - current[0]) * curve_intensity)
            
            control_point2 = (midpoint[0] - (current[1] - nearest[1]) * curve_intensity * 0.5, 
                             midpoint[1] - (nearest[0] - current[0]) * curve_intensity * 0.5)
            
            # Add the control points and the destination to the path
            path.extend([control_point1, control_point2, nearest])
        
        return path
    
    def _create_secondary_geometric_elements(self, dots: list, spacing: int) -> List[List[Tuple[float, float]]]:
        """
        Creates additional geometric elements with more varied and asymmetric patterns.
        """
        if len(dots) < 5: return []
        
        secondary_paths = []
        pattern_style = random.choice(['waves', 'curves', 'triangular', 'organic'])
        
        # Create dynamic elements based on pattern style
        filtered_dots = [d for d in dots if spacing < d[0] < self.width-spacing and 
                                          spacing < d[1] < self.height-spacing]
        
        # Remove any previous references to circular patterns
        
        if pattern_style == 'waves':
            # Create flowing wave patterns
            for i in range(random.randint(3, 7)):
                points = []
                amplitude = random.uniform(0.5, 1.5) * spacing
                frequency = random.uniform(0.1, 0.3)
                phase = random.uniform(0, 2 * math.pi)
                for t in np.linspace(0, 2*math.pi, 20):
                    x = self.width/2 + amplitude * math.cos(t)
                    y = self.height/2 + amplitude * math.sin(t + phase) * math.sin(frequency * t)
                    points.append((x, y))
                secondary_paths.append(points)
                
        elif pattern_style == 'curves':
            # Create organic curved patterns
            for _ in range(random.randint(2, 5)):
                points = []
                start_point = random.choice(filtered_dots)
                for t in np.linspace(0, 2*math.pi, 15):
                    radius = spacing * (1 + 0.5 * math.sin(3*t))
                    x = start_point[0] + radius * math.cos(t)
                    y = start_point[1] + radius * math.sin(t)
                    points.append((x, y))
                secondary_paths.append(points)
                
        elif pattern_style == 'triangular':
            # Create angular geometric patterns
            for _ in range(random.randint(3, 6)):
                if len(filtered_dots) < 3: break
                points = []
                vertices = random.sample(filtered_dots, 3)
                for i in range(4):  # Include closing point
                    points.append(vertices[i % 3])
                secondary_paths.append(points)
                
        else:  # organic
            # Create natural, flowing patterns
            if filtered_dots:
                start = random.choice(filtered_dots)
                points = [start]
                for _ in range(random.randint(8, 15)):
                    angle = random.uniform(0, 2*math.pi)
                    dist = random.uniform(0.3, 1.0) * spacing
                    x = points[-1][0] + dist * math.cos(angle)
                    y = points[-1][1] + dist * math.sin(angle)
                    points.append((x, y))
                secondary_paths.append(points)
        

        
        # Create star-like patterns
        if len(dots) > 10:
            corner_dot = min(dots, key=lambda d: d[0] + d[1])
            points = []
            for i in range(5):
                angle = 2 * np.pi * i / 5
                x = corner_dot[0] + spacing * 1.5 * math.cos(angle)
                y = corner_dot[1] + spacing * 1.5 * math.sin(angle)
                points.append((x, y))
                
                # Add inner points for star shape
                inner_angle = 2 * np.pi * (i + 0.5) / 5
                inner_x = corner_dot[0] + spacing * 0.6 * math.cos(inner_angle)
                inner_y = corner_dot[1] + spacing * 0.6 * math.sin(inner_angle)
                points.append((inner_x, inner_y))
            
            # Connect alternating points
            connected_points = []
            for i in range(10):
                connected_points.append(points[i])
                connected_points.append(points[(i+2) % 10])
            
            secondary_paths.append(connected_points)
        
        return secondary_paths

    def _create_weaving_path(self, dots: list, spacing: int) -> list:
        """Creates a single, continuous, winding path that loops around dots.
        This is the original implementation, kept for backward compatibility.
        For enhanced paths, use _create_enhanced_weaving_path instead.
        """
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
        """Applies symmetry based on the principles from the analysis."""
        # Get symmetry preferences from principles
        symmetries = self.principles.get('common_symmetries', [180])
        
        # Apply rotational symmetry based on the specified angles
        for angle in symmetries:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(mask, M, (self.width, self.height))
            mask = cv2.bitwise_or(mask, rotated)
        
        # Apply reflectional symmetry if needed
        if 180 in symmetries:  # Traditional Kolam often has horizontal and vertical reflection
            mask = cv2.bitwise_or(mask, cv2.flip(mask, 1))  # Horizontal reflection
            mask = cv2.bitwise_or(mask, cv2.flip(mask, 0))  # Vertical reflection
        
        return mask

    def _create_textured_background(self) -> np.ndarray:
        """Creates a textured background for the Kolam. Optimized for performance."""
        # Use vectorized operations for better performance
        canvas = np.full((self.height, self.width, 3), (245, 240, 230), dtype=np.uint8)
        
        # Generate noise more efficiently by using a smaller random array and resizing
        # This reduces memory usage while maintaining visual quality
        small_noise_shape = (self.height // 4, self.width // 4, 3)
        small_noise = np.random.randint(-10, 10, small_noise_shape, dtype=np.int16)
        noise = cv2.resize(small_noise, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        
        # Apply noise to canvas using vectorized operations
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply blur for a smoother texture
        return cv2.GaussianBlur(canvas, (5, 5), 0)

    def _fill_regions_with_gradients(self, canvas: np.ndarray, binary_mask: np.ndarray, palette: list):
        """Fills regions with color gradients. Optimized for better performance."""
        if not palette: return canvas, []
        
        # Find contours in the binary mask
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None: return canvas, []
        
        valid_regions = []
        
        # Pre-compute the gradient once instead of for each contour
        gradient = np.zeros_like(canvas)
        row_indices = np.arange(self.height).reshape(-1, 1)
        alphas = row_indices / self.height
        
        for i, contour in enumerate(contours):
            # Only process child contours (inner regions)
            if hierarchy[0][i][3] != -1:
                # Select random colors from palette
                color1, color2 = random.sample(palette, 2)
                
                # Create mask for this contour
                mask = np.zeros((self.height, self.width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                # Find the region of interest to avoid processing the entire image
                x, y, w, h = cv2.boundingRect(contour)
                roi_mask = mask[y:y+h, x:x+w]
                
                # Only compute gradient for the ROI
                roi_gradient = np.zeros((h, w, 3), dtype=np.uint8)
                roi_alphas = alphas[y:y+h]
                
                # Vectorized gradient computation for better performance
                for c in range(3):  # For each color channel
                    roi_gradient[:, :, c] = np.uint8((1 - roi_alphas) * color1[c] + roi_alphas * color2[c])
                
                # Apply gradient only to masked area within ROI
                roi_canvas = canvas[y:y+h, x:x+w]
                roi_canvas[roi_mask == 255] = roi_gradient[roi_mask == 255]
                canvas[y:y+h, x:x+w] = roi_canvas
                
                valid_regions.append(contour)
                
        return canvas, valid_regions

    def _add_internal_embellishments(self, canvas: np.ndarray, regions: list, palette: list):
        """Adds more intricate embellishments to the Kolam design."""
        complexity = self.principles.get('average_complexity', 35)
        
        for region in regions:
            M = cv2.moments(region)
            if M["m00"] == 0: continue
            cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
            
            # Calculate region size for proportional embellishments
            region_area = cv2.contourArea(region)
            motif_size = int(np.sqrt(region_area) * 0.15)  # Slightly larger motifs
            
            if motif_size < 3: continue
            
            # Randomly choose embellishment type based on complexity
            embellishment_type = random.choices(
                ['circle', 'flower', 'dot_pattern', 'star'], 
                weights=[0.2, 0.3, 0.3, 0.2],
                k=1
            )[0]
            
            highlight_color_np = np.clip(np.array(random.choice(palette)) * 1.4, 0, 255)
            highlight_color = tuple(map(int, highlight_color_np))
            
            if embellishment_type == 'circle':
                cv2.circle(canvas, (cX, cY), motif_size, highlight_color, -1, cv2.LINE_AA)
                # Add inner circle for depth
                inner_color = tuple(map(int, np.clip(np.array(highlight_color) * 1.2, 0, 255)))
                cv2.circle(canvas, (cX, cY), int(motif_size * 0.6), inner_color, -1, cv2.LINE_AA)
                
            elif embellishment_type == 'flower' and motif_size > 5:
                # Create a flower-like pattern
                petal_count = random.randint(5, 8)
                for i in range(petal_count):
                    angle = 2 * np.pi * i / petal_count
                    petal_x = cX + int(motif_size * 0.8 * np.cos(angle))
                    petal_y = cY + int(motif_size * 0.8 * np.sin(angle))
                    cv2.circle(canvas, (petal_x, petal_y), int(motif_size * 0.5), highlight_color, -1, cv2.LINE_AA)
                # Add center
                cv2.circle(canvas, (cX, cY), int(motif_size * 0.4), tuple(map(int, np.array(highlight_color) * 0.8)), -1, cv2.LINE_AA)
                
            elif embellishment_type == 'dot_pattern' and motif_size > 4:
                # Create a pattern of dots
                dot_size = max(2, int(motif_size * 0.25))
                dot_count = min(7, max(3, int(complexity / 10)))
                
                for i in range(dot_count):
                    angle = 2 * np.pi * i / dot_count
                    dot_x = cX + int(motif_size * np.cos(angle))
                    dot_y = cY + int(motif_size * np.sin(angle))
                    cv2.circle(canvas, (dot_x, dot_y), dot_size, highlight_color, -1, cv2.LINE_AA)
                # Center dot
                cv2.circle(canvas, (cX, cY), dot_size, highlight_color, -1, cv2.LINE_AA)
                
            elif embellishment_type == 'star' and motif_size > 6:
                # Create a star pattern
                points = 5
                outer_radius = motif_size
                inner_radius = motif_size * 0.4
                
                star_points = []
                for i in range(points * 2):
                    angle = np.pi * i / points
                    radius = outer_radius if i % 2 == 0 else inner_radius
                    x = cX + int(radius * np.cos(angle))
                    y = cY + int(radius * np.sin(angle))
                    star_points.append((x, y))
                
                star_points = np.array(star_points, dtype=np.int32)
                cv2.fillPoly(canvas, [star_points], highlight_color, lineType=cv2.LINE_AA)
        
        return canvas

    def _draw_artistic_lines(self, canvas: np.ndarray, structure_mask: np.ndarray, line_color: tuple = (40, 30, 20)):
        """
        Draws more intricate and artistic lines for the Kolam pattern.
        
        Args:
            canvas: The canvas to draw on
            structure_mask: The binary mask defining the structure
            line_color: The color to use for drawing lines (default: dark brown)
        """
        # Extract contours from the structure mask
        contours, hierarchy = cv2.findContours(structure_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return canvas
        
        if line_color == (0, 0, 0):
            # For black and white mode, just draw the lines directly
            cv2.polylines(canvas, contours, isClosed=False, color=line_color, thickness=2, lineType=cv2.LINE_AA)
        else:
            # Create a shadow effect for depth
            shadow_canvas = np.zeros_like(canvas)
            
            # Draw thicker shadow lines
            cv2.polylines(shadow_canvas, contours, isClosed=False, color=line_color, thickness=12, lineType=cv2.LINE_AA)
            shadow_canvas = cv2.GaussianBlur(shadow_canvas, (21, 21), 0)
            canvas = cv2.addWeighted(canvas, 1, shadow_canvas, 0.3, 0)
        
        # Draw the main lines with a textured effect
        line_color = (255, 250, 245)
        outline_color = (50, 40, 30)
        
        # Process each contour for more artistic rendering
        for contour in contours:
            # Simplify the contour slightly for smoother lines
            epsilon = 0.002 * cv2.arcLength(contour, False)
            approx_contour = cv2.approxPolyDP(contour, epsilon, False)
            
            # Draw the main line with varying thickness for a hand-drawn look
            points = approx_contour.reshape(-1, 2)
            if len(points) < 2: continue
            
            # Create a more natural, hand-drawn look with varying thickness
            for i in range(len(points) - 1):
                pt1 = tuple(points[i])
                pt2 = tuple(points[i + 1])
                
                # Vary thickness slightly for a more natural look
                thickness = random.randint(4, 6)
                cv2.line(canvas, pt1, pt2, line_color, thickness, cv2.LINE_AA)
                
                # Add a thinner outline for definition
                cv2.line(canvas, pt1, pt2, outline_color, 2, cv2.LINE_AA)
        
        # Add rice flour texture effect (small white dots along the lines)
        for contour in contours:
            points = contour.reshape(-1, 2)
            for i in range(0, len(points), 5):  # Add dots every few pixels
                pt = tuple(points[i])
                # Small random offset for natural look
                offset_x = random.randint(-2, 2)
                offset_y = random.randint(-2, 2)
                dot_pt = (pt[0] + offset_x, pt[1] + offset_y)
                cv2.circle(canvas, dot_pt, 1, (255, 255, 255), -1, cv2.LINE_AA)
        
        return canvas

    def _draw_ornate_border(self, canvas: np.ndarray, palette: list):
        """Draws a more ornate and culturally authentic border for the Kolam."""
        margin = 15
        border_color1 = palette[0]
        border_color2 = palette[1]
        accent_color = palette[2] if len(palette) > 2 else border_color1
        
        # Draw the main border rectangle
        cv2.rectangle(canvas, (margin, margin), (self.width - margin, self.height - margin), border_color1, 10)
        cv2.rectangle(canvas, (margin + 12, margin + 12), (self.width - margin - 12, self.height - margin - 12), border_color2, 2)
        
        # Add corner decorations
        for x in [margin, self.width - margin]:
            for y in [margin, self.height - margin]:
                # Larger outer circle
                cv2.circle(canvas, (x, y), 15, border_color1, -1)
                # Middle circle
                cv2.circle(canvas, (x, y), 10, accent_color, -1)
                # Inner circle
                cv2.circle(canvas, (x, y), 5, border_color2, -1)
        
        # Add traditional kolam patterns along the border
        # Top and bottom borders
        for x in range(margin + 50, self.width - margin, 50):
            # Top border
            self._draw_border_motif(canvas, (x, margin), palette)
            # Bottom border
            self._draw_border_motif(canvas, (x, self.height - margin), palette)
        
        # Left and right borders
        for y in range(margin + 50, self.height - margin, 50):
            # Left border
            self._draw_border_motif(canvas, (margin, y), palette)
            # Right border
            self._draw_border_motif(canvas, (self.width - margin, y), palette)
    
    def _draw_border_motif(self, canvas: np.ndarray, position: Tuple[int, int], palette: list):
        """Draws a small traditional motif for the border decoration."""
        x, y = position
        motif_size = 8
        motif_type = random.choice(['dot_flower', 'small_star', 'diamond'])
        
        if motif_type == 'dot_flower':
            # Draw a small flower-like pattern with dots
            for i in range(6):  # 6 petals
                angle = 2 * np.pi * i / 6
                petal_x = x + int(motif_size * np.cos(angle))
                petal_y = y + int(motif_size * np.sin(angle))
                cv2.circle(canvas, (petal_x, petal_y), 3, palette[0], -1, cv2.LINE_AA)
            # Center dot
            cv2.circle(canvas, (x, y), 4, palette[1], -1, cv2.LINE_AA)
            
        elif motif_type == 'small_star':
            # Draw a small star
            points = 5
            star_points = []
            for i in range(points * 2):
                angle = np.pi * i / points
                radius = motif_size if i % 2 == 0 else motif_size * 0.4
                star_x = x + int(radius * np.cos(angle))
                star_y = y + int(radius * np.sin(angle))
                star_points.append((star_x, star_y))
            star_points = np.array(star_points, dtype=np.int32)
            cv2.fillPoly(canvas, [star_points], palette[1], lineType=cv2.LINE_AA)
            
        elif motif_type == 'diamond':
            # Draw a diamond shape
            diamond_points = [
                (x, y - motif_size),
                (x + motif_size, y),
                (x, y + motif_size),
                (x - motif_size, y)
            ]
            diamond_points = np.array(diamond_points, dtype=np.int32)
            cv2.fillPoly(canvas, [diamond_points], palette[2] if len(palette) > 2 else palette[0], lineType=cv2.LINE_AA)
            # Add a smaller inner diamond
            inner_size = motif_size * 0.6
            inner_diamond = [
                (x, y - inner_size),
                (x + inner_size, y),
                (x, y + inner_size),
                (x - inner_size, y)
            ]
            inner_diamond = np.array(inner_diamond, dtype=np.int32)
            cv2.fillPoly(canvas, [inner_diamond], palette[1], lineType=cv2.LINE_AA)
    
    def _get_default_principles(self) -> dict:
        """Provides fallback principles."""
        return {"average_grid_size": [7, 7], "common_symmetries": [90], "dominant_style": "Pulli", "continuity_preference": 0.2}


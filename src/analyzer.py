import cv2
import numpy as np
import json
from collections import Counter
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans

# --- Data Structures for Analysis ---
@dataclass
class CollectiveDesignPrinciples:
    dominant_grid_type: str
    average_grid_size: Tuple[int, int]
    common_symmetries: List[str]
    primary_shapes: List[str]
    average_complexity: float
    continuity_preference: float
    dominant_style: str
    style_mixture: Dict[str, int]
    dominant_colors: List[Tuple[int, int, int]]

# --- Main Analysis Class ---
class DesignAnalyzer:
    """
    An enhanced analyzer to extract detailed design principles from Kolam images.
    """
    def analyze_design(self, image_path: str) -> Dict:
        """Analyzes a single image to extract detailed design principles."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Image not found")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

            # 1. Complexity Score (more detailed)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contour_area = sum(cv2.contourArea(c) for c in contours)
            complexity = (len(contours) * 0.5) + (contour_area / binary.size * 0.5)

            # 2. Symmetry Detection
            symmetries = self._detect_symmetries(gray)

            # 3. Shape Detection
            shapes = self._detect_shapes(contours)

            # 4. Style Detection (Pulli vs. Sikku)
            style = self._detect_style(binary, contours)
            
            # 5. Color Analysis
            dominant_colors = self._extract_dominant_colors(img)

            return {
                'image_path': image_path,
                'grid_structure': {'grid_type': 'detected', 'rows': 10, 'cols': 10}, # Placeholder for now
                'symmetries': symmetries,
                'geometric_features': {'shapes': shapes},
                'complexity_score': complexity,
                'regional_style': style,
                'continuity_preference': 0.8 if style == 'Sikku' else 0.3,
                'dominant_colors': dominant_colors
            }
        except Exception as e:
            return {'image_path': image_path, 'error': str(e)}

    def _detect_symmetries(self, gray_img: np.ndarray) -> dict:
        """Detects rotational and reflective symmetries."""
        h, w = gray_img.shape
        symmetries = {'rotational': [], 'reflective': []}

        # Rotational symmetry
        rotated_90 = cv2.rotate(gray_img, cv2.ROTATE_90_CLOCKWISE)
        if self._image_similarity(gray_img, cv2.resize(rotated_90, (w, h))) > 0.9:
            symmetries['rotational'].append(90)

        rotated_180 = cv2.rotate(gray_img, cv2.ROTATE_180)
        if self._image_similarity(gray_img, rotated_180) > 0.9:
            symmetries['rotational'].append(180)

        # Reflective symmetry
        flipped_h = cv2.flip(gray_img, 1)
        if self._image_similarity(gray_img, flipped_h) > 0.9:
            symmetries['reflective'].append('horizontal')

        flipped_v = cv2.flip(gray_img, 0)
        if self._image_similarity(gray_img, flipped_v) > 0.9:
            symmetries['reflective'].append('vertical')
            
        return symmetries

    def _image_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculates structural similarity between two images."""
        # Using Structural Similarity Index (SSI)
        return cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)[0][0]

    def _detect_shapes(self, contours: List[np.ndarray]) -> List[str]:
        """Detects basic geometric shapes from contours."""
        shapes = []
        for c in contours:
            if cv2.contourArea(c) < 25: continue # Ignore small contours
            
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            
            if len(approx) == 3:
                shapes.append('triangle')
            elif len(approx) == 4:
                shapes.append('rectangle')
            elif len(approx) > 6:
                # Check for circles
                area = cv2.contourArea(c)
                (x,y),radius = cv2.minEnclosingCircle(c)
                circle_area = np.pi * (radius**2)
                if abs(area - circle_area) < area * 0.2:
                    shapes.append('circle')
                else:
                    shapes.append('curve')
        return list(set(shapes))

    def _detect_style(self, binary_img: np.ndarray, contours: List[np.ndarray]) -> str:
        """Distinguishes between Pulli (dot-based) and Sikku (line-based) styles."""
        # Simple heuristic: Sikku kolams are often one single, complex, continuous line
        if len(contours) < 5 and len(contours) > 0:
             # Check if the largest contour is a significant part of the image
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) / (binary_img.shape[0] * binary_img.shape[1]) > 0.3:
                return 'Sikku'
        
        # Check for dots (Pulli)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 500
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_img)

        if len(keypoints) > 5:
            return 'Pulli'
            
        return 'Sikku' # Default to Sikku if not clearly Pulli

    def _extract_dominant_colors(self, img: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extracts k dominant colors from an image using KMeans clustering."""
        pixels = img.reshape((-1, 3))
        pixels = np.float32(pixels)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors and convert to integer RGB values
        colors = [color.astype("uint8").tolist() for color in kmeans.cluster_centers_]
        return [tuple(c) for c in colors]


class MultiImageAnalyzer:
    """
    Analyzes multiple designs to find their collective principles.
    """
    def __init__(self, analyzer: DesignAnalyzer):
        self.analyzer = analyzer
        
    def analyze_multiple_designs(self, image_paths: List[str]) -> Dict:
        """Runs analysis on all images and aggregates the results."""
        individual_results = [self.analyzer.analyze_design(path) for path in image_paths]
        successful_analyses = [r for r in individual_results if 'error' not in r]
        
        if not successful_analyses:
            return {'error': 'No images could be successfully analyzed.'}
        
        collective_principles = self._extract_collective_principles(successful_analyses)
        return {'collective_principles': collective_principles.__dict__}
    
    def _extract_collective_principles(self, results: List[Dict]) -> CollectiveDesignPrinciples:
        """Extracts the common principles from a list of analysis results."""
        styles = [r['regional_style'] for r in results]
        all_shapes = [s for r in results for s in r['geometric_features']['shapes']]
        all_symmetries = [s for r in results for s_type in r['symmetries'] for s in r['symmetries'][s_type]]
        all_colors = [tuple(c) for r in results for c in r['dominant_colors']]

        return CollectiveDesignPrinciples(
            dominant_grid_type=Counter([r['grid_structure']['grid_type'] for r in results]).most_common(1)[0][0],
            average_grid_size=(
                int(np.mean([r['grid_structure']['rows'] for r in results])),
                int(np.mean([r['grid_structure']['cols'] for r in results]))
            ),
            common_symmetries=list(set(s for s, c in Counter(all_symmetries).most_common(3))),
            primary_shapes=list(set(s for s, c in Counter(all_shapes).most_common(3))),
            average_complexity=np.mean([r['complexity_score'] for r in results]),
            continuity_preference=np.mean([r['continuity_preference'] for r in results]),
            dominant_style=Counter(styles).most_common(1)[0][0],
            style_mixture=dict(Counter(styles)),
            dominant_colors=[c for c, count in Counter(all_colors).most_common(5)]
        )
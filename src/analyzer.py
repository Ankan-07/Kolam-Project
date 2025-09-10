import cv2
import numpy as np
import json
from collections import Counter
from typing import Dict, List, Tuple
from dataclasses import dataclass

# --- Data Structures for Analysis ---
@dataclass
class CollectiveDesignPrinciples:
    dominant_grid_type: str
    average_grid_size: Tuple[int, int]
    common_symmetries: List[int]
    primary_shapes: List[str]
    average_complexity: float
    continuity_preference: float
    dominant_style: str
    style_mixture: Dict[str, int]

# --- Main Analysis Class ---
class DesignAnalyzer:
    """
    A simplified analyzer to extract mock design principles from images.
    This serves as the first step in the pipeline.
    """
    def analyze_design(self, image_path: str) -> Dict:
        """Analyzes a single image to extract mock data."""
        try:
            # In a real scenario, this would involve complex image processing.
            # Here, we generate consistent mock data for demonstration.
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Image not found")
            
            # Mock analysis based on simple image properties
            is_colorful = len(img.shape) > 2 and img.shape[2] == 3
            complexity = np.mean(cv2.Canny(img, 100, 200)) / 100.0 * 50 + 25
            
            return {
                'image_path': image_path,
                'grid_structure': {'grid_type': 'rectangular', 'rows': 7, 'cols': 7},
                'symmetries': {'rotational': [90, 180] if complexity > 50 else [180]},
                'geometric_features': {'shapes': ['curves', 'lines'] if not is_colorful else ['lotus', 'stars']},
                'complexity_score': complexity,
                'regional_style': 'Pulli' if complexity > 60 else 'Sikku',
                'continuity_preference': 0.8 if 'Sikku' in locals() and locals()['regional_style'] == 'Sikku' else 0.3
            }
        except Exception as e:
            return {'image_path': image_path, 'error': str(e)}

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
        
        return CollectiveDesignPrinciples(
            dominant_grid_type=Counter([r['grid_structure']['grid_type'] for r in results]).most_common(1)[0][0],
            average_grid_size=(
                int(np.mean([r['grid_structure']['rows'] for r in results])),
                int(np.mean([r['grid_structure']['cols'] for r in results]))
            ),
            common_symmetries=list(set(s for r in results for s in r['symmetries']['rotational'])),
            primary_shapes=list(set(s for s, c in Counter(all_shapes).most_common(3))),
            average_complexity=np.mean([r['complexity_score'] for r in results]),
            continuity_preference=np.mean([r['continuity_preference'] for r in results]),
            dominant_style=Counter(styles).most_common(1)[0][0],
            style_mixture=dict(Counter(styles))
        )


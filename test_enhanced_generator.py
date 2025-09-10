import os
import time
from src.generator import ArtisticKolamGenerator

def test_enhanced_generator():
    print("\n🧪 Testing Enhanced Kolam Generator...\n")
    
    # Test with default parameters
    start_time = time.time()
    generator = ArtisticKolamGenerator(analysis_path='output_analysis/collective_analysis_results.json')
    kolam = generator.generate()
    default_time = time.time() - start_time
    print(f"✅ Default generation completed in {default_time:.2f} seconds")
    
    # Test with different palettes
    palettes = ["earthy", "festival", "rangoli", "traditional", "pongal"]
    for palette in palettes:
        print(f"\n🔍 Testing {palette} palette...")
        start_time = time.time()
        kolam = generator.generate(palette_name=palette)
        palette_time = time.time() - start_time
        print(f"✅ {palette} palette generation completed in {palette_time:.2f} seconds")
    
    # Test with different styles by modifying principles directly
    styles = ["Sikku", "Pulli"]
    for style in styles:
        print(f"\n🔍 Testing {style} style generation...")
        start_time = time.time()
        # Create a new generator with modified principles
        test_generator = ArtisticKolamGenerator(analysis_path='output_analysis/collective_analysis_results.json')
        test_generator.principles["style"] = style
        kolam = test_generator.generate()
        style_time = time.time() - start_time
        print(f"✅ {style} style generation completed in {style_time:.2f} seconds")
    
    # Test with different symmetries
    symmetries = ["90", "180", "none"]
    for symmetry in symmetries:
        print(f"\n🔍 Testing {symmetry} degrees symmetry...")
        start_time = time.time()
        test_generator = ArtisticKolamGenerator(analysis_path='output_analysis/collective_analysis_results.json')
        test_generator.principles["symmetry"] = symmetry
        kolam = test_generator.generate()
        symmetry_time = time.time() - start_time
        print(f"✅ {symmetry} degrees symmetry completed in {symmetry_time:.2f} seconds")
    
    # Test with traditional patterns
    patterns = ["lotus", "star", "chikku", "kambi"]
    for pattern in patterns:
        print(f"\n🔍 Testing {pattern} traditional pattern...")
        start_time = time.time()
        test_generator = ArtisticKolamGenerator(analysis_path='output_analysis/collective_analysis_results.json')
        test_generator.principles["primary_shapes"] = [pattern]
        kolam = test_generator.generate()
        pattern_time = time.time() - start_time
        print(f"✅ {pattern} pattern generation completed in {pattern_time:.2f} seconds")
    
    # Test with different complexity levels
    complexities = [20, 40, 60]
    for complexity in complexities:
        print(f"\n🔍 Testing complexity level {complexity}...")
        start_time = time.time()
        test_generator = ArtisticKolamGenerator(analysis_path='output_analysis/collective_analysis_results.json')
        test_generator.principles["complexity"] = complexity
        kolam = test_generator.generate()
        complexity_time = time.time() - start_time
        print(f"✅ Complexity level {complexity} completed in {complexity_time:.2f} seconds")
    
    print("\n🎉 All tests completed successfully!")

if __name__ == "__main__":
    test_enhanced_generator()
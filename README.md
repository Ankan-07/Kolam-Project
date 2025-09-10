# 🎨 Kolam Analysis and Artistic Generation Pipeline

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

<div align="center">
  <img src="input_kolams/kolam art.jpg" alt="Kolam Art Example" width="300"/>
</div>

## 📖 Overview

This project is an innovative pipeline that combines traditional art with modern technology to:

- Analyze collections of Kolam designs (traditional Indian art form)
- Extract core design principles and patterns
- Generate new, unique, and artistically beautiful Kolam artworks
- Convert the final artwork into professional, scalable SVG files

> **Kolam** is an ancient art form from South India where geometric patterns are drawn using rice flour. This project aims to preserve and evolve this beautiful tradition through computational art.

## Enhanced Artistic Kolam Generator

The generator has been enhanced with the following features:

### 1. Traditional Pattern Support

- Added support for traditional Tamil Kolam patterns: lotus, star, chikku, and kambi
- Implemented pattern-based generation with cultural authenticity

### 2. Improved Symmetry and Weaving

- Enhanced symmetry application based on design principles
- Implemented Bezier curve-based weaving paths for smoother designs
- Added secondary geometric elements for more complex patterns

### 3. Expanded Color Palettes

- Added traditional and pongal-themed color palettes
- Improved gradient filling for more vibrant designs

### 4. Enhanced Embellishments

- Added more intricate internal embellishments
- Improved artistic line drawing for better visual appeal
- Enhanced ornate border decoration

### 5. Optimized Performance

- Improved computational efficiency through caching and vectorization
- Optimized texture generation and gradient filling

## 📁 Project Structure

````bash
kolam_project/
│
├── 📂 input_kolams/      # Store your input Kolam images (.jpg/.png)
├── 📂 output_artworks/   # Generated artworks (.png and .svg)
├── 📂 output_analysis/   # Analysis results in JSON format
│
├── 📂 src/               # Source code directory
│   ├── __init__.py
│   ├── analyzer.py       # Image analysis module
│   ├── generator.py      # Kolam generation engine
│   └── converter.py      # PNG to SVG converter
│
├── 🚀 main.py            # Main execution script
├── 📋 requirements.txt    # Project dependencies
└── 📖 README.md          # Documentation

## 🚀 Getting Started

### Prerequisites

- Python 3.13 or higher
- Visual Studio Code (recommended)
- Basic understanding of terminal/command line

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ankan-07/Kolam-Project.git
   cd Kolam-Project
````

2. **Set up Python Virtual Environment**

   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Prepare Input Images**

   - Place your Kolam images (JPG/PNG) in the `input_kolams/` directory
   - Ensure images are clear and well-lit

2. **Run the Pipeline**

   ```bash
   # Step 1: Analyze Kolam Images
   python main.py analyze
   # Creates collective_analysis_results.json in output_analysis/

   # Step 2: Generate New Kolam
   python main.py generate
   # Creates artistic_kolam_final.png in output_artworks/

   # Step 3: Convert to SVG
   python main.py convert
   # Creates artistic_kolam_final.svg in output_artworks/
   ```

### 🌟 Tips for Best Results

- Use high-resolution input images
- Include diverse Kolam patterns for better analysis
- Ensure good contrast in input images
- Keep the virtual environment active during execution

## 🧪 Testing

Validate the enhanced Kolam generator:

```bash
python test_enhanced_generator.py
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Traditional Kolam artists for inspiration
- The Python community for excellent image processing libraries
- Contributors and testers

## 📞 Contact

Ankan - [@github](https://github.com/Ankan-07)

Project Link: [https://github.com/Ankan-07/Kolam-Project](https://github.com/Ankan-07/Kolam-Project)

This script tests:

- Different color palettes (earthy, festival, rangoli, traditional, pongal)
- Different styles (Sikku, Pulli)
- Various symmetry options (90°, 180°, none)
- Traditional patterns (lotus, star, chikku, kambi)
- Different complexity levels (20, 40, 60)

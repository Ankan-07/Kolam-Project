Kolam Analysis and Artistic Generation Pipeline
This project provides a complete pipeline to analyze a collection of Kolam designs, extract their core design principles, and then use those principles to generate new, unique, and artistically beautiful Kolam artworks. The final artwork can be converted into a professional, scalable SVG file.

Project Structure
kolam_project/
│
├── input_kolams/         # <-- Place your input .jpg or .png images here
├── output_artworks/      # <-- Final generated artworks (.png and .svg) are saved here
├── output_analysis/      # <-- The JSON analysis result is saved here
│
├── src/                  # <-- All Python source code
│   ├── __init__.py
│   ├── analyzer.py       # (Handles analysis of input images)
│   ├── generator.py      # (Generates new artistic Kolams)
│   └── converter.py      # (Converts final PNG to SVG)
│
├── main.py               # <-- The main script you will run
├── requirements.txt      # (Lists all necessary Python libraries)
└── README.md             # (This instruction file)

How to Set Up and Run the Pipeline
Follow these steps to get the project running on your local machine using VS Code.

Step 1: Set Up the Python Environment
It is highly recommended to use a virtual environment to keep your project dependencies clean and isolated.

Open a terminal in VS Code (Ctrl+``  or View > Terminal).

Create a virtual environment:

python -m venv venv

Activate the environment:

On Windows: .\venv\Scripts\activate

On macOS/Linux: source venv/bin/activate

(Your terminal prompt should now show (venv) at the beginning).

Step 2: Install Required Libraries
Install all the necessary libraries at once using the requirements.txt file.

pip install -r requirements.txt

Step 3: Add Your Input Images
Place the Kolam images you want to analyze into the input_kolams/ directory.

Step 4: Run the Full Pipeline
The main.py script is your control panel. The full workflow now has three steps that should be run in order.

Action 1: Analyze Your Kolam Images
This command will process all images in input_kolams and create a collective_analysis_results.json file in output_analysis.

python main.py analyze

Action 2: Generate a New Artistic Kolam
This command reads the JSON file and creates a new, beautiful Kolam, saving it as artistic_kolam_final.png in output_artworks.

python main.py generate

Action 3: Convert the Artwork to SVG
This final command takes the generated PNG and converts it into a high-quality, scalable artistic_kolam_final.svg file in the same output_artworks directory.

python main.py convert

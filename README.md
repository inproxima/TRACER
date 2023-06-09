# TRACER
TRACER is designed to extract themes from different types of files, such as Word documents, plain text files, and PDFs. TRACER uses Langchain prompts for analyzing the themes and identifying the four most frequent and central aspects related to teaching, materials, and assessment. The results of the analysis are saved in a text file. The code requires OpenAI and other Python libraries to be installed and some environment variables to be set. Please alter the prompts to customize TRACER based on your data set. 
 
# Installation
1. Download or clone the Github repository to your local machine.
2. Open a command-line interface and navigate to the directory containing the downloaded repository.
3. Create a virtual environment and activate it (optional but recommended).
4. Install the required dependencies using the command pip install -r requirements.txt.
5. Create a directory named "tempDir" in the root of the project directory.
6. Place the documents you want to analyze in the "tempDir" directory. Supported file types are .docx, .pdf, and .txt.
7. Open the ".env" file and replace the value of OPENAI_API_KEY with your OpenAI API key.
8. Open the command line and navigate to the project directory.
9. Run the program using the command python app.py.
10. Wait for the program to finish executing. The output file will be saved in the "results" directory.

# Changing the program
Please alter the prompts to customize the number of themes you want TRACER to find.

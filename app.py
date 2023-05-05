#External librearies 
import openai
from docx import Document as DocxDocument
from pypdf import PdfReader



#Python libraries
import re
import os
import mimetypes
from dotenv import load_dotenv
import json
import datetime


#langchain libraries
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain import PromptTemplate
from langchain.chains import LLMChain


#Llama libraries
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, QuestionAnswerPrompt, ServiceContext


#Save the file in directory
def read_docx(directory_path: str) -> None:
    # Get the root directory of the given directory_path
    root_dir = os.path.dirname(os.path.abspath(directory_path))

    text_dir = os.path.join(root_dir, 'texts')
    os.makedirs(text_dir, exist_ok=True)

    for filename in os.listdir(directory_path):
        if filename.endswith('.docx'):
            filepath = os.path.join(directory_path, filename)
            document = DocxDocument(filepath)
            
            output_file = os.path.join(text_dir, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                for paragraph in document.paragraphs:
                    text = paragraph.text
                    # Merge hyphenated words
                    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                    # Fix newlines in the middle of sentences
                    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
                    # Remove multiple newlines
                    text = re.sub(r"\n\s*\n", "\n\n", text)
                    f.write(text + '\n')


def read_txt(directory_path: str) -> None:
    root_dir = os.path.dirname(os.path.abspath(directory_path))

    text_dir = os.path.join(root_dir, 'texts')
    os.makedirs(text_dir, exist_ok=True)

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory_path, filename)
            
            output_file = os.path.join(text_dir, f"{os.path.splitext(filename)[0]}_processed.txt")
            with open(filepath, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                text = f_in.read()
                # Merge hyphenated words
                text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                # Fix newlines in the middle of sentences
                text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
                # Remove multiple newlines
                text = re.sub(r"\n\s*\n", "\n\n", text)
                f_out.write(text)


#Golden Function for pdf
def read_pdf(directory_path: str) -> None:
    # Get the root directory of the given directory_path
    root_dir = os.path.dirname(os.path.abspath(directory_path))

    text_dir = os.path.join(root_dir, 'texts')
    os.makedirs(text_dir, exist_ok=True)

    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory_path, filename)
            
            output_file = os.path.join(text_dir, f"{os.path.splitext(filename)[0]}.txt")
            with open(filepath, 'rb') as pdf_file, open(output_file, 'w', encoding='utf-8') as text_file:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    text = page.extract_text()
                    # Merge hyphenated words
                    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                    # Fix newlines in the middle of sentences
                    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
                    # Remove multiple newlines
                    text = re.sub(r"\n\s*\n", "\n\n", text)
                    text_file.write(text + '\n')

def save_output_to_file(theme_dir: str, original_filename: str, output_theme: str) -> None:
    theme_filename = f"{os.path.splitext(original_filename)[0]}_theme.txt"

    theme_filepath = os.path.join(theme_dir, theme_filename)

    with open(theme_filepath, "w", encoding="utf-8") as theme_file:
        theme_file.write(output_theme)

def format_transcript(data):
    transcript = json.loads(data)
    result = ""
    for item in transcript:
        if len(item) == 2:
            question, answer = item
            answer = answer.replace("\\n", "\n")
            result += f"{question}\n{answer}\n"
        else:
            print(f"Skipping item due to irregular structure: {item}")
    return result

def clear_files(directory):
    # Delete any files in directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def create_temp_dir():
    if not os.path.exists("tempDir"):
        os.makedirs("tempDir")

def combine_text_files():
    input_dir = 'theme_analysis'
    output_file = 'themes.txt'

    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' not found.")
        return

    # Create an empty list to store the contents of the text files
    combined_contents = []

    # Iterate through the files in the input directory
    for filename in os.listdir(input_dir):
        # Check if the file has a .txt extension
        if filename.endswith('.txt'):
            # Open the file and read its contents
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
                contents = file.read()
                combined_contents.append(contents)

    # Combine the contents of all text files
    combined_text = '\n'.join(combined_contents)

    # Save the combined contents into the output file
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write(combined_text)


def construct_index(directory_path, api):
    openai.api_key = api
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 1000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", max_tokens=num_outputs, openai_api_key=api))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # Get the root directory of the given directory_path
    root_dir = os.path.dirname(os.path.abspath(directory_path))

    # create a folder to store the json files
    json_folder = os.path.join(root_dir, 'json_files')
    os.makedirs(json_folder, exist_ok=True)

    # read documents from the directory
    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    for idx, doc in enumerate(documents):
        index = GPTSimpleVectorIndex.from_documents(
            [doc], service_context=service_context
        )
        # save the index as a json file using the index of the document in the list
        json_file = os.path.join(json_folder, f"document_{idx}.json")
        index.save_to_disk(json_file)

def construct_index_content(directory_path, api):
    openai.api_key = api
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 1000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", max_tokens=num_outputs, openai_api_key = api))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )

    index.save_to_disk('index.json')

def theme(query_str):
    QA_PROMPT_TMPL = (
    "You are a thematic analysis bot. Please conduct a thematic analysis on the following {context_str} which are transcripts between a researcher and a teacher about a new model of teaching mathematics. Your task is to analyze the conversation and identify themes which are patterns or recurring concepts that emerges that emerge from the conversation. Please read through the transcript thoroughly, extract the themes related to teaching mathematics and assessment. Present your findings in a clear but detailed manner. Additionally, provide a brief summary of each theme to give context on how it relates to the theme. Given this information, please answer the question: {query_str}\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
    json_folder = "json_files"

    # Get the root directory of the json_folder
    root_dir = os.path.dirname(os.path.abspath(json_folder))

    # Create a folder to store the theme analysis files
    theme_analysis_folder = os.path.join(root_dir, 'theme_analysis')
    os.makedirs(theme_analysis_folder, exist_ok=True)

    # Loop through the files in the "json_files" directory
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            json_file = os.path.join(json_folder, filename)

            # Load the index from the JSON file
            index = GPTSimpleVectorIndex.load_from_disk(json_file)

            # Query the index and store the response
            response = index.query(query_str, response_mode="default", text_qa_template=QA_PROMPT, mode="embedding")

            # Save the response to a file in the theme_analysis folder with the same name as the JSON file
            analysis_file = os.path.join(theme_analysis_folder, f"{os.path.splitext(filename)[0]}_analysis.txt")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(str(response))

def save_output_with_timestamp(content):
    # Create a directory called 'results' if it doesn't exist
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the current timestamp and format it as a string
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Create a unique file name using the timestamp
    output_filename = f"output_{timestamp}.txt"

    # Save the content in a text file in the 'results' directory
    with open(os.path.join(output_dir, output_filename), 'w', encoding='utf-8') as output_file:
        output_file.write(content)

    print(f"Output content has been saved in '{output_dir}/{output_filename}'.")


if __name__ == '__main__':

    # Load .env file
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY") 
    api = openai.api_key

    tempDir = "tempDir"
    content_folder = "content_analysis"
    theme_folder = "theme_analysis"
    json_files = "json_files"
    texts = "texts"
    theme_analysis = "theme_analysis"
                
    for docx_files in os.listdir(tempDir):
        filepath = os.path.join(tempDir, docx_files)
        mime_type, _ = mimetypes.guess_type(filepath)
        #mime_type = filetype.guess(filepath)
        file_extension = os.path.splitext(tempDir)[1]

    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        raw_text = read_docx(tempDir)
        embeddings = construct_index(texts, api)
        user_input = "What are four main themes?"
        response_theme = theme(user_input)
        output_theme = str(response_theme)

        if not os.path.exists(theme_folder):
            os.makedirs(theme_folder)

        for filename in os.listdir(tempDir):
            original_filepath = os.path.join(tempDir, filename)
        
    elif mime_type == "text/plain":
        raw_text = read_txt(tempDir)
        embeddings = construct_index(texts, api)
        user_input = "What are four main themes?"
        response_theme = theme(user_input)
        output_theme = str(response_theme)

        if not os.path.exists(theme_folder):
            os.makedirs(theme_folder)

        for filename in os.listdir(tempDir):
            original_filepath = os.path.join(tempDir, filename)
        
    elif file_extension.lower() == ".pdf" or mime_type == "application/pdf":
        raw_text = read_pdf(tempDir)
        embeddings = construct_index(texts, api)
        user_input = "What are four main themes?"
        response_theme = theme(user_input)
        output_theme = str(response_theme)

        if not os.path.exists(theme_folder):
            os.makedirs(theme_folder)

        for filename in os.listdir(tempDir):
            original_filepath = os.path.join(tempDir, filename)

        
    else:
        print(f"Skipping file {filepath} with MIME type {mime_type}")

    
    combine_text_files()
    with open("themes.txt") as f:
        themes = f.read()


    template = """
    You are given content containing a list of themes gathered from multiple transcripts. Your task is to analyze these themes and identify the four most frequent and central aspects related to teaching, materials, and assessment. Please consider the following instructions:

    1. Recognize semantically similar themes, even if they have different wording, and group them together.
    2. Focus on the common threads connecting various ideas within the context of teaching and assessment.
    3. Rank the themes based on their frequency, and present the top 4 most common themes.
    4. For each of the top 4 themes, provide a brief explanation on how it relates to teaching and assessment.

    Here is the content:

    {themes}

    Analyze the themes, and provide your insights on the most frequent and central aspects related to teaching and assessment.

    """
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=ChatOpenAI(), prompt=prompt)
    output_content = chain.run(themes)
    save_output_with_timestamp(output_content)
    print(output_content)
    # Clear files if you are conducting multiple excutions
    #clear_files(json_files)
    #clear_files(theme_analysis)

#External librearies 
import openai
from docx import Document
from pypdf import PdfReader



#Python libraries
import re
import os
import mimetypes
from dotenv import load_dotenv
import json


#langchain libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, QuestionAnswerPrompt, ServiceContext




#Save the file in directory
def read_docx(directory_path: str) -> str:
    output = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.docx'):
            filepath = os.path.join(directory_path, filename)
            document = Document(filepath)
            for paragraph in document.paragraphs:
                text = paragraph.text
                # Merge hyphenated words
                text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                # Fix newlines in the middle of sentences
                text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
                # Remove multiple newlines
                text = re.sub(r"\n\s*\n", "\n\n", text)
                output.append(text)
    return output


def read_txt(directory_path: str) -> str:
    output = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                # Merge hyphenated words
                text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                # Fix newlines in the middle of sentences
                text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
                # Remove multiple newlines
                text = re.sub(r"\n\s*\n", "\n\n", text)
                output.append(text)
    return output

#Golden Function for pdf
def read_pdf(directory_path: str) -> str:
    output = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    # Merge hyphenated words
                    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                    # Fix newlines in the middle of sentences
                    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
                    # Remove multiple newlines
                    text = re.sub(r"\n\s*\n", "\n\n", text)
                    output.append(text)
    return output


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

def clear_files():
    # Delete any files in tempDir
    for filename in os.listdir(tempDir):
        file_path = os.path.join(tempDir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete index.json file
    index_path = os.path.join(os.getcwd(), "index.json")
    if os.path.exists(index_path):
        os.remove(index_path)

def create_temp_dir():
    if not os.path.exists("tempDir"):
        os.makedirs("tempDir")

#Indexing the uploaded files    
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
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.2, model_name="text-embedding-ada-002", max_tokens=num_outputs, openai_api_key = api))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )

    index.save_to_disk('index.json')

    return index 


#Generate responses
def content(query_str):
    QA_PROMPT_TMPL = (
    "You are a content analysis bot. You analyze content from {context_str}. The goal is to analyze the frequency of specific words, themes, or other characteristics in a set of content. Given this information, please answer the question: {query_str}\n. Always list your response."
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
    # Build GPTSimpleVectorIndex
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(query_str, response_mode="compact", text_qa_template=QA_PROMPT, mode="embedding")
    return response

def theme(query_str):
    QA_PROMPT_TMPL = (
    "You are a content analysis bot. You analyze content from {context_str}. The goal is to identifying recurring themes or patterns in a set of content. Given this information, please answer the question: {query_str}\n. Allways list your response."
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
    # Build GPTSimpleVectorIndex
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(query_str, response_mode="compact", text_qa_template=QA_PROMPT, mode="embedding")
    return response

# Load .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") 
api = openai.api_key

create_temp_dir()
tempDir = "tempDir/"
            
for docx_files in os.listdir(tempDir):
    filepath = os.path.join(tempDir, docx_files)
    mime_type, _ = mimetypes.guess_type(filepath)
    #mime_type = filetype.guess(filepath)
    file_extension = os.path.splitext(tempDir)[1]

if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    raw_text = read_docx(tempDir)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(raw_text)
    embeddings = construct_index(tempDir, api)
    #index = GPTSimpleVectorIndex.load_from_disk('index.json')
    
elif mime_type == "text/plain":
    raw_text = read_txt(tempDir)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(raw_text)
    embeddings = construct_index(tempDir, api)
    #index = GPTSimpleVectorIndex.load_from_disk('index.json')
    
elif file_extension.lower() == ".pdf" or mime_type == "application/pdf":
    raw_text = read_pdf(tempDir)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(raw_text)
    embeddings = construct_index(tempDir, api)
    #index = GPTSimpleVectorIndex.load_from_disk('index.json')

    
else:
    print(f"Skipping file {filepath} with MIME type {mime_type}")

user_input = "What are four main themes?"
response_content = content(user_input)
output_content = str(response_content)
response_theme = theme(user_input)
output_theme = str(response_theme)
print(output_content + '/n' + output_theme )

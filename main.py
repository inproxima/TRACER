#External librearies 
import openai
from docx import Document as DocxDocument
from pypdf import PdfReader
import streamlit as st
import streamlit_ext as ste 


#Python libraries
import re
import os
import mimetypes
from dotenv import load_dotenv, set_key, find_dotenv
import json
import datetime
from pathlib import Path

#langchain libraries
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

#Llama libraries
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext

#Functions
#Docx to text
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


#PDF to Text
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


@st.cache_data
def save_uploaded_file(uploadedfile):
    for file in uploadedfile:
        with open(os.path.join("tempDir", file.name), "wb") as f:
            f.write(file.getbuffer())
        st.success("Saved {} : To tempDir".format(file.name))

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

#Make JSON
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


#Extract themes
def theme(number, interviewee, topic, focus):
    template = (
    "You are a thematic analysis bot. Please conduct a thematic analysis on the following {context_str} which are transcripts between a researcher and a {interviewee} about {topic}. Your task is to analyze the conversation and identify themes which are patterns or recurring concepts that emerges that emerge from the conversation. Please read through the transcript thoroughly, extract the {number} themes related to {focus}. Present your findings in a clear but detailed manner. Additionally, provide a brief summary of each theme to give context on how it relates to the theme."
    )


    #chat = ChatOpenAI(temperature=0)
    #prompt = PromptTemplate.from_template(template)
    #chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=prompt)
    #output_content = chain.predict(themes=themes, number=number)
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
            response = index.query(template, response_mode="default", mode="embedding")

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

def display_and_download_files(folder_path: str):
    # Read all files in the given folder
    folder = Path(folder_path)
    all_files = list(folder.glob("*"))

    # Display and create download buttons for each file
    for file in all_files:
        # Display the file content
        st.write(f"### {file.name}")
        st.write(file.read_text())

        # Create a download button for each file
        st.download_button(
            f"Download {file.name}", file.read_text(), f"{file.name}"
        )

#stramlit components
st.set_page_config(page_title="TRACER, Transcript Analysis and Concept Extraction Resource!", page_icon="🤖", initial_sidebar_state="expanded")
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

st.sidebar.header("About")
st.sidebar.markdown(
    """
This application was created by [Soroush Sabbaghan](mailto:ssabbagh@ucalgary.ca) using [Streamlit](https://streamlit.io/), [LangChain](https://github.com/hwchase17/langchain), [LlamaIndex](https://gpt-index.readthedocs.io/en/stable/) and [OpenAI API](https://openai.com/api/)'s 
[CHATGPT API](https://platform.openai.com/docs/models/overview) for educational purposes. 
"""
)

st.sidebar.header("Copyright")
st.sidebar.markdown(
    """
- This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
"""
)

st.sidebar.subheader('Notes')
st.sidebar.markdown(
    """
- TRACER does not save or copy any content uploaded or generated. 
"""
)

st.title("Hi, I'm TRACER 👋")
st.subheader("Transcript Analysis and Concept Extraction Resource")
st.markdown("""___""")
st.subheader("1. Please Enter you OpenAI API key")
url = "https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key"
api = st.text_input("If you don't know your OpenAI API key click [here](%s)." % url, type="password", placeholder="Your API Key")

#API variables
env_path = find_dotenv()
if env_path == "":
    with open(".env", "w") as env_file:
        env_file.write("# .env\n")
        env_path = ".env"

# Load .env file
load_dotenv(dotenv_path=env_path)

set_key(env_path, "OPENAI_API_KEY", api)

openai.api_key = api

if st.button("Check key"):
    if api is not None:
    #Delete any files in tempDir
        st.session_state.api = api
        try:
            # Send a test request to the OpenAI API
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt="What is the capital of France?",
                temperature=0.5
            )
            st.markdown("""---""")
            st.success("API key is valid!")
        except Exception as e:
            st.error("API key is invalid: {}".format(e))
st.markdown("""---""") 

#Upload documents
st.subheader("2. Please Upload the files you wish to extract themes from.")
docx_files = st.file_uploader("Upload Document", type=["pdf","docx","txt"], accept_multiple_files=True)
st.markdown("""---""") 
st.subheader("3. Please enter the required information.")
number = st.number_input("How many themes would you like tracer to find?", step=1)
interviewee = st.text_input("Who is the researcher interviewing?")
topic = st.text_input("What are the context of the interview about?", placeholder = "e.g., 'a new model of teaching mathematics")
focus = st.text_input("What would you like TRACER to focus on?", placeholder = "e.g., 'teaching mathematics and assessment")

#Directory
tempDir = "tempDir"
theme_folder = "theme_analysis"
json_files = "json_files"
texts = "texts"
theme_analysis = "theme_analysis"


if number and interviewee and topic and focus is not None:
    if st.button("Get Themes!"):
        save_uploaded_file(docx_files)
        with st.spinner(text="Processing uploaded files..."):
            #identify file type          
            for docx_files in os.listdir(tempDir):
                filepath = os.path.join(tempDir, docx_files)
                mime_type, _ = mimetypes.guess_type(filepath)
                #mime_type = filetype.guess(filepath)
                file_extension = os.path.splitext(tempDir)[1]

            # output themes
            if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                raw_text = read_docx(tempDir)
                embeddings = construct_index(texts, api)
                #user_input = f"What are {number} main themes?"
                response_theme = theme(number, interviewee, topic, focus)
                output_theme = str(response_theme)

                if not os.path.exists(theme_folder):
                    os.makedirs(theme_folder)

                for filename in os.listdir(tempDir):
                    original_filepath = os.path.join(tempDir, filename)
                
            elif mime_type == "text/plain":
                raw_text = read_txt(tempDir)
                embeddings = construct_index(texts, api)
                #user_input = f"What are {number} main themes?"
                response_theme = theme(number, interviewee, topic, focus)
                output_theme = str(response_theme)

                if not os.path.exists(theme_folder):
                    os.makedirs(theme_folder)

                for filename in os.listdir(tempDir):
                    original_filepath = os.path.join(tempDir, filename)
                
            elif file_extension.lower() == ".pdf" or mime_type == "application/pdf":
                raw_text = read_pdf(tempDir)
                embeddings = construct_index(texts, api)
                #user_input = f"What are {number} main themes?"
                response_theme = theme(number, interviewee, topic, focus)
                output_theme = str(response_theme)

                if not os.path.exists(theme_folder):
                    os.makedirs(theme_folder)

                for filename in os.listdir(tempDir):
                    original_filepath = os.path.join(tempDir, filename)

                
            else:
                st.write(f"Skipping file {filepath} with MIME type {mime_type}")

        with st.spinner(text="Extracting themes..."): 
            combine_text_files()
            with open("themes.txt") as f:
                themes = f.read()

            #Langchain prompt
            template = """
            You are given content containing a list of themes gathered from multiple transcripts. Your task is to analyze these themes and identify the {number} most frequent and central aspects related to {focus}. Please consider the following instructions:

            1. Recognize semantically similar themes, even if they have different wording, and group them together.
            2. Focus on the common threads connecting various ideas within the context of teaching and assessment.
            3. Rank the themes based on their frequency, and present the top {number} most common themes.
            4. For each of the top {number} themes, provide a brief explanation on how it relates to teaching and assessment.

            Here is the content:

            {themes}

            Analyze the themes, and provide your insights on the most frequent and central aspects related to teaching and assessment.

            """
            chat = ChatOpenAI(temperature=0)
            prompt = PromptTemplate.from_template(template)
            chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=prompt)
            output_content = chain.predict(themes=themes, number=number, focus=focus)
            #save_output_with_timestamp(output_content)
            st.markdown("""___""")
            st.subheader("Extracted themes for every transcript")
            display_and_download_files("theme_analysis")
            st.markdown("""___""")
            st.subheader("Aggregated Themes:")
            st.write(output_content)
            ste.download_button("Download Themes", output_content, "Themes.txt")
            # Clear files if you are conducting multiple excutions
            clear_files(json_files)
            clear_files(theme_analysis)
            clear_files(texts)
            clear_files(tempDir)
            file_path_theme = "themes.txt"
            if os.path.exists(file_path_theme):
                os.remove(file_path_theme)


from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os


def load_txt_files(folder_path):
    files_content = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):

            file_path = os.path.join(folder_path, filename)
            print(filename, file_path)
            loader = TextLoader(file_path)
            text_document = loader.load()
            for text in text_document:  # Iterate through each text chunk in the document

                text.page_content.replace('\n', ' ')

                files_content.append(text)

    return files_content


def get_vectorstore_from_texts(txt_files_content):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(txt_files_content)
    vector_store = Chroma.from_documents(docs, OpenAIEmbeddings(),
                                         persist_directory="./chroma_db_media_txt")



def main():
    # Load environment variables (if any)
    load_dotenv()

    # Specify the path to your folder containing .txt files
    folder_path = 'data'  # Ensure this points to your actual folder

    # Load contents of .txt files from the folder
    files_content = load_txt_files(folder_path)
    get_vectorstore_from_texts(files_content)

if __name__ == '__main__':
    main()
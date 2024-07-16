from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

class JSONLoader(TextLoader):
    def __init__(self, file_path: Union[str, Path], content_key: Optional[str] = None):
        super().__init__(file_path)
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key

    def create_documents(self, processed_data):
        documents = []
        for item in processed_data:
            content = ''.join(item)
            metadata, main_context = content.split('~')
            metadata = metadata.replace('mediastrom products ', '')
            document = Document(page_content=''.join(metadata + ':' + main_context), metadata={'product': metadata})
            print(document)
            documents.append(document)
        return documents

    def process_item(self, item, prefix=""):
        if isinstance(item, dict):
            result = []
            for key, value in item.items():
                new_prefix = f"{prefix} {key}" if prefix else key
                new_prefix = new_prefix.replace("description", "")
                result.extend(self.process_item(value, new_prefix))
            return result
        elif isinstance(item, list):
            result = []
            for value in item:
                result.extend(self.process_item(value, prefix))
            return result
        else:
            return [f"{prefix}~ {item}"]

    def process_json(self, data):
        if isinstance(data, list):
            processed_data = []
            for item in data:
                processed_data.extend(self.process_item(item))
            return processed_data
        elif isinstance(data, dict):
            return self.process_item(data)
        else:
            return []

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""

        docs = []
        with open(self.file_path, 'r') as json_file:
            try:
                data = json.load(json_file)
                processed_json = self.process_json(data)
                docs = self.create_documents(processed_json)
            except json.JSONDecodeError:
                print("Error: Invalid JSON format in the file.")
        return docs

def load_txt_files(folder_path):
    files_content = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and False:

            file_path = os.path.join(folder_path, filename)
            print(filename, file_path)
            loader = TextLoader(file_path)
            text_document = loader.load()
            for text in text_document:  # Iterate through each text chunk in the document

                text.page_content.replace('\n', ' ')

                files_content.append(text)
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            print(filename, file_path)
            loader = JSONLoader(file_path=file_path)
            docs = loader.load()
            files_content.extend(docs)

    return files_content


def get_vectorstore_from_texts(txt_files_content, chroma_name='./chroma_db_media'):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)
    docs = text_splitter.split_documents(txt_files_content)
    vector_store = Chroma.from_documents(docs, OpenAIEmbeddings(),
                                         persist_directory=chroma_name)



def main():
    # Load environment variables (if any)
    load_dotenv()

    # Specify the path to your folder containing .txt files
    folder_path = 'data/beds_data'  # Ensure this points to your actual folder
    # Load contents of .txt files from the folder
    files_content = load_txt_files(folder_path)
    get_vectorstore_from_texts(files_content, chroma_name='./chroma_db_media_beds')

    # Specify the path to your folder containing .txt files
    folder_path = 'data/mattresses_data'  # Ensure this points to your actual folder
    # Load contents of .txt files from the folder
    files_content = load_txt_files(folder_path)
    get_vectorstore_from_texts(files_content, chroma_name='./chroma_db_media_mattresses')

    # Specify the path to your folder containing .txt files
    folder_path = 'data/pillows_data'  # Ensure this points to your actual folder
    # Load contents of .txt files from the folder
    files_content = load_txt_files(folder_path)
    get_vectorstore_from_texts(files_content, chroma_name='./chroma_db_media_pillows')

if __name__ == '__main__':
    main()
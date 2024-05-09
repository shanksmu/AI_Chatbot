from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
BASE_PATH = os.getenv("BASE_PATH")


extracted_data = load_pdf(BASE_PATH+"/AI_Chatbot/Data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()



index_name="gen-ai-chatbot"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)
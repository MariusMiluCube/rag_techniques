from langchain.vectorstores import FAISS
from evaluation.evalute_rag import evaluate_rag
from helper_functions import EmbeddingProvider, retrieve_context_per_question, replace_t_with_space, get_langchain_embedding_provider, show_context
from pydantic.v1.utils import lenient_isinstance
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain import text_splitter
import os
import sys
from dotenv import load_dotenv

sys.path.append('RAG_TECHNIQUES')

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError(
        'OPENAI_API_KEY is not set. Please add it to your .env file.')

os.environ['OPENAI_API_KEY'] = openai_api_key
os.makedirs('data', exist_ok=True)


path = 'data/Understanding_Climate_Change.pdf'


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.
    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.
    Returns:
        A FAISS vector store containing the encoded book content.
    """
    # Load Documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Creating Embedings
    embeddings = get_langchain_embedding_provider(EmbeddingProvider.OPENAI)

    # Create Vectorstore
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    return vectorstore


chunk_vectore_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)

print(chunk_vectore_store)

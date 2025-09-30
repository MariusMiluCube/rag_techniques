from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from openai.types import vector_store
from langchain.document_loaders import CSVLoader
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import faiss

sys.path.append('RAG_TECHNIQUES')

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini')

os.makedirs('data', exist_ok=True)
file_path = ('data/customers-100.csv')
data = pd.read_csv(file_path)

# print(data.head())
# Load docs
loader = CSVLoader(file_path=file_path)
docs = loader.load_and_split(loader)

# Embeding
embeddings = OpenAIEmbeddings
index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query(' ')))
vector_store = FAISS(embedding_function=OpenAIEmbeddings(
), index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
vector_store.add_documents(documents=docs)

# Retrival Chain
retriver = vector_store.as_retriever()

# Set up System Prompt
system_prompt = (
    'You are an assistent for question answering tasks. '
    'Use the following pieces of retrived context to answer'
    'the question. If you do not know the answer say that'
    'you do not know. Use three sentences maximum and keep the'
    'answer concice'
    '\n\n'
    '{context}'
)

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('human', '{input}'),

])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriver, question_answer_chain)

# Rag Invoke
answer = rag_chain.invoke(
    {'input': "which company does sheryl Baxter work for?"})
print(answer['answer'])

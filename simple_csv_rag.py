from tkinter import N
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from langchain.document_loaders import CSVLoader
from openai.types import vector_store
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import sys
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY missing. Put it in your .env")


def build_retriver(csv_path: str):
    # 1 Load CSV
    loader = CSVLoader(file_path=csv_path, encoding='utf-8')
    docs = loader.load()

    # 2 Split Documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # 3 Embeding + index
    embedidings = OpenAIEmbeddings(model='text-embedding-3-small')
    vs = FAISS.from_documents(splits, embedidings)
    return vs.as_retriever(search_kwargs={'k': 4})


def make_qa_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'Answer concisely using only the provided context. If missing say you do not knwo'),
        ('human', 'Question: {question}\n\nContext:\n{context}'),
    ])
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
    to_str = StrOutputParser()

    def _chain(question: str) -> str:
        docs = retriever.get_relevant_documents(question)
        context = '\n\n'.join(d.page_content for d in docs)
        return (prompt | llm | to_str).invoke({'question': question, 'context': context})

    return _chain


if __name__ == '__main__':
    csv_path = 'data/customers-100.csv'
    retriver = build_retriver(csv_path=csv_path)
    qa = make_qa_chain(retriver)
    print(qa('Give me a short summart of what is inside the CSV'))

    # llm = ChatOpenAI(model='gpt-4o-mini')

    # os.makedirs('data', exist_ok=True)
    # file_path = ('data/customers-100.csv')
    # data = pd.read_csv(file_path)

    # # print(data.head())
    # # Load docs
    # loader = CSVLoader(file_path=file_path)
    # docs = loader.load_and_split()

    # # Embeding
    # embeddings = OpenAIEmbeddings
    # index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query(' ')))
    # vector_store = FAISS(embedding_function=OpenAIEmbeddings(
    # ), index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    # vector_store.add_documents(documents=docs)

    # # Retrival Chain
    # retriver = vector_store.as_retriever()

    # # Set up System Prompt
    # system_prompt = (
    #     'You are an assistent for question answering tasks. '
    #     'Use the following pieces of retrived context to answer'
    #     'the question. If you do not know the answer say that'
    #     'you do not know. Use three sentences maximum and keep the'
    #     'answer concice'
    #     '\n\n'
    #     '{context}'
    # )

    # prompt = ChatPromptTemplate.from_messages([
    #     ('system', system_prompt),
    #     ('human', '{input}'),

    # ])

    # question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # rag_chain = create_retrieval_chain(retriver, question_answer_chain)

    # # Rag Invoke
    # answer = rag_chain.invoke(
    #     {'input': "which company does sheryl Baxter work for?"})
    # print(answer['answer'])

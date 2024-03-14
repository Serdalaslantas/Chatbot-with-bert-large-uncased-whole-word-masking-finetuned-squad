import streamlit as st
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import textwrap
import concurrent.futures

st.markdown('''
    <style>
        body {
            background-color: #D3D3D3;
        }
    </style>
''', unsafe_allow_html=True)

# Lazy loading of the text document
def lazy_load_text(file_path):
    with open(file_path, encoding='UTF-8') as f:
        return f.read()

state_of_the_union = lazy_load_text("text.txt")

# Text Splitting
text_splitter_recursive = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter_recursive.create_documents(text_splitter_recursive.split_text(state_of_the_union))

# Lazy loading of the document
lazy_loaded_document = lambda: docs

# Lazy loading of embeddings
lazy_loaded_embeddings = lambda: HuggingFaceBgeEmbeddings()

# Lazy loading of Faiss index
@st.cache_data
def lazy_loaded_db():
    return FAISS.from_documents(lazy_loaded_document(), lazy_loaded_embeddings())

# Lazy loading of the Question Answering chain
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 128},
                    huggingfacehub_api_token='HUGGIN FACE TOKEN NO')  ### ONEMLI###

#HUGGINGFACE TOKEN GIR, Hugging face tokeninizi hugging face te account aldiktan sonra
#setting -->> sonra access secenegi var ordan bulabilirsiniz

@st.cache_data
def lazy_loaded_chain():
    return load_qa_chain(llm, chain_type="stuff")

# Streamlit interface
st.markdown('''
    <div style="text-align: center">
        <p style="background-color: #FFBF00; color: black; font-size: 30px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.1);">
           OneAmz LangChain Chatbot <br>
            <span style="font-size: 14px; color:green;">Online</span>
        </p>
    </div>
''', unsafe_allow_html=True)

# User input
user_input = st.text_area("Ask me a question:")

if st.button("Ask"):
    # Perform similarity search and run the question answering chain asynchronously
    with concurrent.futures.ThreadPoolExecutor() as executor:
        doc_future = executor.submit(lazy_loaded_db().similarity_search, user_input)
        chain_future = executor.submit(lazy_loaded_chain().run, {"input_documents": doc_future.result(), "question": user_input})

    doc = doc_future.result()
    answer = chain_future.result()

    # Display the answer
    st.markdown(f"**Answer:** {answer}")

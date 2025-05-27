import os
import time
import streamlit as st
from dotenv import load_dotenv
import re

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("AI News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()
store_path = "vector_index_store"

# Set up LLM
llm = OpenAI(temperature=0.9, max_tokens=500)

# Process URLs when button clicked
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started ✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Splitting Text... ✅")
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorindex_openai = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Building Embedding Index... ✅")
    time.sleep(2)

    vectorindex_openai.save_local(store_path)

# Question box
query = main_placeholder.text_input("Question:")

# Handle Q&A
if query:
    if os.path.exists(store_path):
        embeddings = OpenAIEmbeddings()
        vectorindex_openai = FAISS.load_local(
            store_path, embeddings, allow_dangerous_deserialization=True
        )

        retriever = vectorindex_openai.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        # Wrap the answer in triple backticks to prevent Markdown parsing issues
        st.markdown(f"```text\n{result['answer']}\n```")

        if result.get("sources"):
            st.subheader("Sources:")
            for source in result["sources"].split("\n"):
                st.write(source)

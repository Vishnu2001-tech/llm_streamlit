import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

def generate_response(file, openai_api_key, query):
    try:
        # Format file
        reader = PdfReader(file)
        formatted_document = []
        for page in reader.pages:
            formatted_document.append(page.extract_text())
        # Split file
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        docs = text_splitter.create_documents(formatted_document)
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Load to vector database
        store = FAISS.from_documents(docs, embeddings)
        
        # Create retrieval chain
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=store.as_retriever()
        )
        # Run chain with query
        return retrieval_chain.run(query)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

st.set_page_config(
    page_title="Q&A from a long PDF Document"
)

st.title("Q&A from a long PDF Document")

uploaded_file = st.file_uploader(
    "Upload a PDF Document",
    type="pdf"
)

query_text = st.text_input(
    "Enter your Question:",
    placeholder="Write your Question",
    disabled=not uploaded_file
)

result = []
with st.form(
    "myform",
    clear_on_submit=True
):
    open_api_key = st.text_input(
        "OpenAPI Key",
        type="password",
        disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text)
    )
    if submitted:
        if open_api_key.startswith("sk-"):
            with st.spinner(
                "Wait Please, I am working on it..."
            ):
                response = generate_response(
                    uploaded_file,
                    open_api_key,
                    query_text
                )
                if response:
                    result.append(response)
                else:
                    st.error("No response was generated.")
        else:
            st.error("Invalid OpenAI key. It should start with 'sk-'.")

if result:
    st.info(result[0])
        
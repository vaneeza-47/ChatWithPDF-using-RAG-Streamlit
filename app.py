import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Load environment variables ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- Streamlit UI setup ---
st.set_page_config(page_title="ChatWithPDF", page_icon="🤖")

st.markdown("""
<style>
/* --- Uploader Box --- */
div[data-testid="stFileUploader"] > section {
    border: 2px solid black !important;
    border-radius: 50px !important;
    padding: 1.5rem !important;
    background-color: #00ffd2 !important; /* aqua background */
    font-weight: 700 !important; /* bold text */
}

/* Make drag & drop text bold */
div[data-testid="stFileUploader"] label {
    font-weight: 700 !important;
}

/* Make the drag-and-drop area fully cylindrical */
div[data-testid="stFileUploaderDropzone"] {
    border-radius: 40px !important;
    overflow: hidden !important;
}

/* --- Browse Files Button --- */
div[data-testid="stFileUploader"] button {
    background-color: white !important;
    color: black !important;
    border: 2px solid black !important;
    border-radius: 30px !important; /* makes it cylindrical */
    font-weight: 700 !important; /* bold text */
    padding: 0.4rem 1rem !important;
    transition: all 0.2s ease-in-out;
}

/* Add hover effect for Browse button */
div[data-testid="stFileUploader"] button:hover {
    background-color: black !important;
    color: white !important;
}

/* --- Info Box ("Please upload...") --- */
div[data-testid="stAlert"] {
    border: 2px solid black !important;
    border-radius: 40px !important;
    background-color: #f8fbff !important;
    color: #000 !important;
    font-weight: 700 !important; /* bold text */
}

/* Tighten layout */
section[data-testid="stFileUploader"] {
    margin-bottom: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* --- Text Input: Ask a question box --- */
div[data-testid="stTextInput"] {
    border: 2px solid black !important;
    border-radius: 30px !important;  /* makes it cylindrical */
    padding: 0.9rem 1rem !important;
    background-color: white !important;
}

/* Input text inside the box */
div[data-testid="stTextInput"] input {
    font-weight: 600 !important; /* slightly bold text */
    color: black !important;
    background: transparent !important;
    border: none !important;
}

</style>
""", unsafe_allow_html=True)



st.title("Chat with your PDF using Gemini")


# --- PDF Upload ---
uploaded_file = st.file_uploader("📤 **Upload a PDF file**", type=["pdf"])

# We’ll use session state to prevent re-processing on every rerun
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.retriever = None

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("📄 Processing your PDF... Please wait."):
        temp_pdf_path = "temp_uploaded.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # --- Load and split PDF ---
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        # --- Create embeddings and vectorstore ---
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        st.success("✔ PDF processed and indexed successfully!")

# --- Ask questions ---
if st.session_state.retriever:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, max_output_tokens=1024)

    system_prompt = """
    You are a helpful assistant for answering questions using the provided PDF context.
    Use only the retrieved context to answer.
    If unsure, say "I don't know" — do not make up an answer.
    \n\n
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    user_question = st.text_input("💬 Ask a question about your PDF:")

    if user_question:
        with st.spinner("Thinking..."):
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)
            response = rag_chain.invoke({"input": user_question})

            st.write("### **Answer**:")
            st.write(response["answer"])
else:
    st.info("👆 **Please upload a PDF file to get started.**")


# --- Footer ---
st.markdown(
    """
    <hr style="border:1px solid #999; margin-bottom:10px;">
    <div style="text-align:center; color:gray; font-size:14px;">
        Built with <b>Streamlit</b> and <b>Gemini API</b><br>
        Created by <b>Vaneeza</b> 💎
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import tempfile
import os
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

# PAGE CONFIGURATION
st.set_page_config(
    page_title="ResearchBot | AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -1.5px;
}

.gradient-text {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 25px;
}

.feature-card {
    background: #172235;
    border: 1px solid #2a3b53;
    border-radius: 14px;
    padding: 20px;
    min-height: 120px;
}

.feature-title {
    font-size: 16px;
    font-weight: 700;
    color: #f1f5f9;
}

.feature-desc {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 7px;
}

.status-card {
    background: #172235;
    border: 1px solid #2a3b53;
    border-radius: 12px;
    padding: 14px;
}

.stButton > button {
    border-radius: 9px;
    border: 1px solid #38bdf8;
    background: #0ea5e9;
    color: white;
    font-weight: 600;
}

.stButton > button:hover {
    background: #0284c7;
    border-color: #0284c7;
    color: white;
}

[data-testid="stChatMessage"] {
    background: #172235;
    border: 1px solid #263449;
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 12px;
}

[data-testid="stChatInput"] {
    border-color: #334155;
}

hr {
    border-color: #263449;
}

</style>
""", unsafe_allow_html=True)

# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None
# EMBEDDING MODEL
@st.cache_resource(show_spinner=False)
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# PROMPT TEMPLATE
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question,
    rephrase the follow-up question to be a standalone question.

    Chat History:
    {chat_history}

    Follow-up Input:
    {question}

    Standalone Question:"""
)

# PDF PROCESSING

def process_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:

        pdf_reader = PyPDFLoader(temp_path)
        documents = pdf_reader.load()

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not documents:
        raise ValueError("No readable pages were found in this PDF.")

    if not any(doc.page_content.strip() for doc in documents):
        raise ValueError(
            "This PDF contains no extractable text. "
            "It may be a scanned document requiring OCR."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = load_embeddings()

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(
            search_kwargs={"k": 4}
        ),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
        verbose=False
    )

    return qa, len(documents), len(chunks)

# SIDEBAR

with st.sidebar:

    st.markdown("## 📚 ResearchBot")

    st.caption("YOUR PERSONAL AI RESEARCH ASSISTANT")

    st.markdown("---")

    st.markdown("### 📄 Upload Research Paper")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a research paper or academic document."
    )

    if uploaded_file is not None:

        current_hash = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()

        if current_hash != st.session_state.file_hash:

            with st.spinner("Analyzing your research paper..."):

                try:

                    qa, page_count, chunk_count = process_pdf(
                        uploaded_file
                    )

                    st.session_state.qa_chain = qa
                    st.session_state.file_hash = current_hash
                    st.session_state.file_name = uploaded_file.name

                    st.session_state.messages = []
                    st.session_state.chat_history = []

                    st.success("Research paper processed successfully!")

                except Exception as e:

                    st.session_state.qa_chain = None
                    st.error(f"Processing failed: {str(e)}")

    st.markdown("---")

    st.markdown("### ⚙️ Document Information")

    if st.session_state.qa_chain is not None:

        st.markdown(
            f"""
            <div class="status-card">
                <div style="color:#38bdf8;font-weight:700;">
                    🟢 Document Ready
                </div>
                <br>
                <div style="font-size:13px;color:#e2e8f0;">
                    {st.session_state.file_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info("Upload a PDF to activate your research assistant.")

    st.markdown("---")

    st.markdown("### 🧠 AI Configuration")

    st.markdown("""
    - **LLM:** Qwen 2.5 3B
    - **Embeddings:** MiniLM
    - **Vector DB:** FAISS
    - **Framework:** LangChain
    """)

    st.markdown("---")

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        st.session_state.messages = []
        st.session_state.chat_history = []

        st.rerun()

    st.caption("Powered by LangChain + Ollama")

# MAIN INTERFACE

st.markdown(
    """
    <div class="main-title">
        Research<span class="gradient-text">Bot.</span>
    </div>
    <div class="subtitle">
        Your intelligent companion for reading, understanding,
        and exploring academic research papers.
    </div>
    """,
    unsafe_allow_html=True
)

# WELCOME SCREEN

if st.session_state.qa_chain is None:

    st.markdown(
        """
        <div style="
            background:linear-gradient(120deg,#172235,#1e293b);
            border:1px solid #334155;
            padding:35px;
            border-radius:18px;
            margin-bottom:25px;
        ">
            <div style="font-size:35px;">🔬</div>
            <h2 style="color:#f8fafc;">
                Understand research papers faster.
            </h2>
            <p style="color:#94a3b8;font-size:15px;">
                Upload your academic PDF and let AI help you
                summarize complex ideas, explain methodologies,
                identify findings and answer your questions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ✨ What can ResearchBot do?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size:25px;">📝</div>
            <div class="feature-title">Summarization</div>
            <div class="feature-desc">
                Get concise summaries of lengthy research papers.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size:25px;">💡</div>
            <div class="feature-title">Deep Explanations</div>
            <div class="feature-desc">
                Understand technical concepts and research methodologies.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size:25px;">🔎</div>
            <div class="feature-title">Smart Retrieval</div>
            <div class="feature-desc">
                Find relevant information directly from your PDF.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.info(
        "👈 Start by uploading a research paper using the sidebar."
    )

# CHAT INTERFACE

else:

    st.markdown(
        f"**Currently analyzing:** `{st.session_state.file_name}`"
    )

    st.markdown("---")

    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):

                with st.expander("📚 View supporting PDF passages"):

                    for i, source in enumerate(
                        message["sources"], start=1
                    ):

                        page = source.metadata.get("page", "Unknown")

                        if isinstance(page, int):
                            page = page + 1

                        st.markdown(f"**Source {i} — Page {page}**")

                        st.write(
                            source.page_content
                        )

                        st.markdown("---")

    # User input

    user_question = st.chat_input(
        "Ask anything about your research paper..."
    )

    if user_question:

        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):

            with st.spinner("ResearchBot is thinking..."):

                try:

                    result = st.session_state.qa_chain.invoke({
                        "question": user_question,
                        "chat_history": st.session_state.chat_history
                    })

                    answer = result["answer"]

                    sources = result.get(
                        "source_documents", []
                    )

                    st.markdown(answer)

                    with st.expander("📚 View supporting PDF passages"):

                        if sources:

                            for i, source in enumerate(
                                sources, start=1
                            ):

                                page = source.metadata.get(
                                    "page", "Unknown"
                                )

                                if isinstance(page, int):
                                    page = page + 1

                                st.markdown(
                                    f"**Source {i} — Page {page}**"
                                )

                                st.write(source.page_content)
                                st.markdown("---")

                        else:
                            st.write("No source passages returned.")

                    st.session_state.chat_history.append(
                        (user_question, answer)
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    st.error(
                        f"Something went wrong: {str(e)}"
                    )

                    st.info(
                        "Check that Ollama is running and "
                        "the Qwen model is installed."
                    )

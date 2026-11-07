# Import necessary libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
llm = ChatOllama(
    model="qwen2.5:3b"
)
# print(llm.invoke("Explain EDA in just two lines"))
# Using custom dataset
pdf_reader = PyPDFLoader(r"C:\Users\HP\Desktop\New ML\Researchbot\RAGPaper+(1).pdf")
documents = pdf_reader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.from_documents(documents=chunks,embedding=embeddings)
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question,
    rephrase the follow-up question to be a standalone question.

    Chat History:
    {chat_history}

    Follow-up Input:
    {question}

    Standalone Question:"""
)
qa = ConversationalRetrievalChain.from_llm(llm=llm,retriever=db.as_retriever(),condense_question_prompt=CONDENSE_QUESTION_PROMPT,return_source_documents=True,verbose=False)
# Ask a query
chat_history = []
query = """What is the research paper all about,explain in just 3-4 lines."""
result = qa({"question":query,"chat_history":chat_history})
print(result['answer'])
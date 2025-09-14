import os
import pickle
import fitz  # PyMuPDF
import spacy
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Fix for Windows OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --- 1. Environment Setup ---
def setup_environment():
    """Loads environment variables and returns API keys."""
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not cohere_api_key or not google_api_key:
        raise ValueError("⚠️ COHERE_API_KEY and GOOGLE_API_KEY must be set in a .env file.")
    return cohere_api_key, google_api_key


# --- 2. Document Processing ---
def get_retriever(knowledge_base_path, cohere_api_key):
    """
    Creates or loads a hybrid retriever (FAISS + BM25) from a PDF knowledge base.
    """
    doc_name = os.path.splitext(os.path.basename(knowledge_base_path))[0]
    vectordb_path = f"faiss_cache_{doc_name}"
    splits_path = f"splits_cache_{doc_name}.pkl"

    embedding = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)

    if os.path.exists(vectordb_path) and os.path.exists(splits_path):
        print("📦 Loading existing FAISS vector store and splits...")
        vectordb = FAISS.load_local(vectordb_path, embedding, allow_dangerous_deserialization=True)
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
    else:
        print(f"🔧 Building new vector store for {doc_name}...")
        loader = PyPDFLoader(knowledge_base_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = splitter.split_documents(pages)

        if not splits:
            print("⚠️ No splits were created from the document.")
            return None

        vectordb = FAISS.from_documents(splits, embedding=embedding)
        vectordb.save_local(vectordb_path)
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)
        print("💾 Vector store and splits saved.")

    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )


# --- 3. Resume Analysis ---
def analyze_resume(resume_path):
    """Extracts text and finds SQL/database-related experience in resume."""
    nlp = spacy.load("en_core_web_sm")
    text = ""
    try:
        with fitz.open(resume_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"❌ Error reading resume PDF: {e}")
        return []

    sql_keywords = [
        "sql", "mysql", "postgresql", "mssql", "sqlite", "sql server",
        "oracle", "database", "t-sql", "pl/sql", "query", "queries",
        "nosql", "mongodb", "cassandra", "data modeling", "data warehousing"
    ]

    doc = nlp(text)
    sql_sentences = [
        sent.text.strip().replace("\n", " ")
        for sent in doc.sents
        if any(kw in sent.text.lower() for kw in sql_keywords)
    ]

    print(f"✅ Found {len(sql_sentences)} SQL-related sentences.")
    return sql_sentences


# --- 4. RAG Context Retrieval ---
def get_relevant_context(retriever, queries):
    """Uses retriever to fetch relevant knowledge base context."""
    if not retriever or not queries:
        return ""
    all_docs = []
    for query in queries:
        all_docs.extend(retriever.invoke(query))
    return "\n\n---\n\n".join([doc.page_content for doc in all_docs])


# --- 5. Interview Chain ---
def initialize_interview_chain(google_api_key, student_name):
    """Initializes conversational interview chain."""
    llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)

    prompt_template_text = """
### Persona:
You are an expert Hiring Manager at a top tech company. You are interviewing "{student_name}".

### Contextual Data:
{related_data}

### Ongoing Transcript:
{chat_history}

---
### Task:
1. <evaluation> Give a private assessment of candidate’s last answer.
2. <question> Acknowledge their answer & ask ONE open-ended follow-up.
"""

    prompt = PromptTemplate(
        template=prompt_template_text,
        input_variables=["related_data", "chat_history"]
    )
    return prompt | llm | StrOutputParser()


# --- 6. Feedback Report Chain ---
def generate_feedback_report_chain(google_api_key):
    """Initializes chain to generate final structured feedback report."""
    llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)

    prompt_template_text = """
You are a hiring manager. Analyze the interview transcript below and generate a report.

ONLY use provided chat history. Do not invent or add fake dialogues.

---
INTERVIEW TRANSCRIPT:
{chat_history}
---

### Overall Summary
(2–3 sentences summarizing performance)

### Strengths
* Point 1
* Point 2

### Areas for Improvement
* Point 1

### Hiring Recommendation
("Strong Recommend", "Recommend", "No Hire") – one sentence justification.
"""

    prompt = PromptTemplate(template=prompt_template_text, input_variables=["chat_history"])
    return prompt | llm | StrOutputParser()


# --- 7. Wrap-up ---
def conclude_interview(chat_history, google_api_key):
    """Generates final feedback report with cleaned transcript."""
    if not chat_history:
        return "⚠️ No chat history found."

    formatted = []
    for i, turn in enumerate(chat_history):
        role = "Interviewer" if i % 2 == 0 else "Candidate"
        formatted.append(f"{role}: {turn}")
    transcript = "\n".join(formatted)

    report_chain = generate_feedback_report_chain(google_api_key)
    return report_chain.invoke({"chat_history": transcript})

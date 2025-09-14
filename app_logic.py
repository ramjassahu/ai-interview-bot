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

# This resolves a common issue on Windows with multiple OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 1. Environment and API Key Setup ---

def setup_environment():
    """Loads environment variables and returns API keys."""
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not cohere_api_key or not google_api_key:
        raise ValueError("COHERE_API_KEY and GOOGLE_API_KEY must be set in a .env file.")
    return cohere_api_key, google_api_key

# --- 2. Document Processing and Retriever Creation ---

def get_retriever(knowledge_base_path, cohere_api_key):
    """
    Creates or loads a hybrid retriever (FAISS + BM25) from a PDF document.
    Caches the vector store and splits for faster re-runs.
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
            print("⚠️ No splits were created from the document. Cannot build retriever.")
            return None

        vectordb = FAISS.from_documents(splits, embedding=embedding)
        vectordb.save_local(vectordb_path)
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)
        print("💾 Vector store and splits saved.")

    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    print("✅ Ensemble retriever created.")
    return ensemble_retriever

# --- 3. Resume Analysis ---

def analyze_resume(resume_path):
    """Extracts text from a resume and finds sentences related to SQL/databases."""
    nlp = spacy.load("en_core_web_sm")
    text = ""
    try:
        with fitz.open(resume_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading resume PDF: {e}")
        return []

    if not text:
        return []

    sql_keywords = [
        "sql", "mysql", "postgresql", "mssql", "sqlite", "sql server",
        "oracle", "database", "t-sql", "pl/sql", "query", "queries", "nosql",
        "mongodb", "cassandra", "data modeling", "data warehousing"
    ]
    doc = nlp(text)
    sql_sentences = []
    for sentence in doc.sents:
        if any(keyword in sentence.text.lower() for keyword in sql_keywords):
            sql_sentences.append(sentence.text.strip().replace("\n", " "))
    
    print(f"✅ Found {len(sql_sentences)} relevant sentences in the resume.")
    return sql_sentences

# --- 4. RAG Context Retrieval ---

def get_relevant_context(retriever, queries):
    """Uses the retriever to find document chunks relevant to the resume queries."""
    if not retriever or not queries:
        return ""
    
    all_docs = []
    for query in queries:
        relevant_docs = retriever.invoke(query)
        all_docs.extend(relevant_docs)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc in all_docs])
    print("✅ Generated context from relevant document chunks.")
    return context_text

# --- 5. Conversational Chain Initialization ---

def initialize_interview_chain(google_api_key, student_name):
    """
    Initializes the LangChain chain for conducting the interactive part of the interview.
    """
    llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)

    prompt_template_text = f"""
### Persona:
You are an expert Hiring Manager at a top tech company: insightful, professional, and friendly. You are interviewing "{student_name}" for a technical role. Your goal is to assess their skills and engage them in a natural, coherent conversation.

### Primary Goal:
Conduct a realistic interview. Your entire response MUST be structured in two parts: an <evaluation> block and a <question> block.

---

### **Contextual Data (Candidate's Resume & Job-Related Info):**
{{related_data}}
---

### **Ongoing Interview Transcript:**
{{chat_history}}
---

### **Your Two-Part Task:**

1.  **Internal Evaluation (Think Step):** Inside an `<evaluation>` XML tag, write a brief, private analysis of the candidate's last answer. Assess its technical depth, clarity, and relevance. This is your internal monologue.

2.  **Formulate Next Question (Act Step):** Inside a `<question>` XML tag, write your response to the candidate. Acknowledge their last point, then transition to your next single, open-ended question.

**Example Output Structure:**
<evaluation>The candidate's overview was good but lacked specific metrics. I need to probe for details.</evaluation>
<question>That's a helpful summary. Could you walk me through the most significant technical challenge you faced and how you overcame it?</question>

**Your Turn:**
"""
    prompt = PromptTemplate(
        template=prompt_template_text,
        input_variables=["related_data", "chat_history"]
    )
    
    return prompt | llm | StrOutputParser()

# --- 6. Feedback Report Generation ---

def generate_feedback_report_chain(google_api_key):
    """
    Initializes a separate LangChain chain to generate a final feedback report.
    """
    llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)

    # ### MODIFIED PROMPT ###
    # This prompt is now much more direct to prevent the model from getting confused.
    # It clearly states the task and separates the instructions from the input data.
    prompt_template_text = """
### Your Role and Goal:
You are an expert Hiring Manager. Your task is to analyze the complete interview transcript provided below and write a comprehensive, constructive performance report for the hiring committee.

### Instructions:
Based *only* on the transcript, generate a feedback report with these exact sections:
1.  **Overall Summary:** A brief, 2-3 sentence paragraph summarizing your impression of the candidate.
2.  **Strengths:** A bulleted list of 2-4 key strengths demonstrated by the candidate.
3.  **Areas for Improvement:** A bulleted list of 1-3 areas where the candidate could improve. Frame this constructively.
4.  **Hiring Recommendation:** A clear recommendation (e.g., "Strong Recommend," "Recommend," "Leaning No," "No Hire") with a one-sentence justification.

---
### **Complete Interview Transcript to Analyze:**
{{chat_history}}
---

### **Hiring Manager's Report:**
"""
    prompt = PromptTemplate(
        template=prompt_template_text,
        input_variables=["chat_history"]
    )
    
    return prompt | llm | StrOutputParser()

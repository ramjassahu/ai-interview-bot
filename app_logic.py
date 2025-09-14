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
    """
    Extracts text from a resume and finds sentences related to technical skills.
    This version is more generic than the original SQL-focused one.
    """
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

    # Expanded list of technical keywords
    tech_keywords = [
        "python", "java", "c++", "javascript", "sql", "nosql", "react", "angular", "vue",
        "django", "flask", "spring", "node.js", "docker", "kubernetes", "aws", "azure", "gcp",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "api", "rest", "graphql",
        "mysql", "postgresql", "mongodb", "redis", "cassandra", "data modeling", "data warehousing",
        "machine learning", "deep learning", "natural language processing", "computer vision",
        "langchain", "faiss", "bm25", "fastapi"
    ]
    doc = nlp(text)
    skill_sentences = []
    for sentence in doc.sents:
        if any(keyword in sentence.text.lower() for keyword in tech_keywords):
            skill_sentences.append(sentence.text.strip().replace("\n", " "))
    
    print(f"✅ Found {len(skill_sentences)} relevant sentences in the resume.")
    return skill_sentences

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
You are an expert Hiring Manager at a top tech company. You are interviewing "{student_name}". Your goal is to assess their skills based on their resume and the provided context.

### Primary Goal:
Your response MUST be structured in two parts: an <evaluation> block and a <question> block.

---
### **Contextual Data (Candidate's Resume & Job-Related Info):**
{{related_data}}
---
### **Ongoing Interview Transcript:**
{{chat_history}}
---
### **Your Two-Part Task:**
1.  **Internal Evaluation (Think Step):** In an `<evaluation>` tag, write a brief, private analysis of the candidate's last answer. Assess its technical depth and clarity.
2.  **Formulate Next Question (Act Step):** In a `<question>` tag, write your response to the candidate. Briefly acknowledge their last point and then ask your next single, open-ended question.

**Your Turn:**
"""

    prompt = PromptTemplate(
        template=prompt_template_text,
        input_variables=["related_data", "chat_history"]
    )
    print(prompt)
    return prompt | llm | StrOutputParser()

# --- 6. Feedback Report Generation ---

def generate_feedback_report_chain(google_api_key):
    """
    Initializes a separate LangChain chain to generate a final feedback report.
    This version uses a highly direct and constrained prompt to prevent hallucinations.
    """
    llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)

    # This prompt is structured to eliminate any possible confusion for the LLM.
    # It contains no examples, only direct commands and the data placeholder.
    prompt_template_text = """
You are a hiring manager. Your only task is to analyze the interview transcript provided below and generate a performance report based *only* on that transcript.

**YOUR OUTPUT MUST STRICTLY FOLLOW THIS FORMAT:**

### Overall Summary
A 2-3 sentence summary of the candidate's performance during the interview.

### Strengths
* A bullet point listing a key strength demonstrated in the transcript.
* Another bullet point listing a different strength.

### Areas for Improvement
* A bullet point with constructive feedback based on their answers.

### Hiring Recommendation
A clear recommendation (e.g., "Recommend," "Strong Recommend," "No Hire") followed by a single sentence justification based on the interview.

---
**INTERVIEW TRANSCRIPT TO ANALYZE:**
{{chat_history}}
---

**PERFORMANCE REPORT:**
"""
    prompt = PromptTemplate(
        template=prompt_template_text,
        input_variables=["chat_history"]
    )
    
    return prompt | llm | StrOutputParser()

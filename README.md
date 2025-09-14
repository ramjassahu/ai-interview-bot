AI Interview Bot 🤖
A smart, interactive interview simulator powered by Google Gemini and LangChain. This application conducts a personalized technical interview based on a candidate's resume and a predefined knowledge base.

➡️ Live Demo Link

A quick walkthrough of the AI Interview Bot in action.

📋 Overview
This project aims to help students and developers practice for technical interviews. The AI acts as an experienced hiring manager, asking relevant, open-ended questions that are tailored to the skills mentioned in the user's resume. The conversation is guided by a Retrieval-Augmented Generation (RAG) system, ensuring the questions are contextually relevant to a specific knowledge domain (in this case, Excel and SQL).

✨ Key Features
Personalized Experience: The interview starts by addressing the candidate by their name.

Dynamic Resume Analysis: Automatically parses the uploaded resume (PDF) to identify key skills and topics to focus on.

RAG-Powered Questions: Uses a hybrid search (FAISS Vector Search + BM25 Keyword Search) to retrieve relevant context from a knowledge base, leading to smarter, more relevant questions.

Interactive Chat Interface: A clean and intuitive user interface built with Streamlit.

Powered by Gemini: Leverages Google's Gemini 1.5 Flash model for generating fluent, insightful, and context-aware dialogue.

🛠️ How It Works
The application follows a streamlined process to create a realistic interview simulation:

Setup: The user enters their name and uploads their resume.

Resume Parsing: The backend uses PyMuPDF and spaCy to extract text and identify sentences related to key technical skills (e.g., "SQL", "database", "MySQL").

Context Retrieval (RAG):

A local PDF document serves as the "knowledge base" for the interview.

The skill-related sentences from the resume are used as queries against an EnsembleRetriever.

This hybrid retriever combines semantic search (via Cohere embeddings and FAISS) and keyword search (BM25) to find the most relevant document chunks.

LLM Chain Invocation:

The retrieved context, along with the chat history, is passed to a prompt template.

The Google Gemini model receives this enriched prompt and generates the hiring manager's next response, always ending with a follow-up question.

User Interaction: The response is displayed in the chat interface, and the loop continues.

🚀 Tech Stack
Frontend: Streamlit

LLM & AI: LangChain, Google Gemini 1.5 Flash, Cohere Embeddings

Backend: Python

Vector Store: FAISS (Facebook AI Similarity Search)

Document Processing: PyMuPDF, spaCy, pypdf

Deployment: Streamlit Community Cloud

🔧 Running the Project Locally
To run this application on your own machine, follow these steps:

1. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

2. Create a .env File
Create a file named .env in the root of the project folder and add your API keys:

COHERE_API_KEY="your_cohere_api_key_here"
GOOGLE_API_KEY="your_google_api_key_here"

3. Install Dependencies
It's recommended to use a virtual environment.

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt

# Download the spaCy model
python -m spacy download en_core_web_sm

4. Run the Streamlit App

streamlit run main.py

The application should now be running in your local browser!
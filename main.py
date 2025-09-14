import streamlit as st
import os
import app_logic # Import our backend logic

# --- Configuration ---
# SET THE LOCAL PATH TO YOUR KNOWLEDGE BASE DOCUMENT HERE
KNOWLEDGE_BASE_PATH = "767888691-Excel-Interview-Questions.pdf" 

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Interview Bot 🤖",
    page_icon="🤖",
    layout="centered"
)

# --- Main App Interface ---
st.title("AI Interview Bot 🤖")
st.markdown("Enter the student's name and upload their resume to begin.")

# --- Session State Initialization ---
if 'session_vars' not in st.session_state:
    st.session_state.session_vars = {
        "context_data": "",
        "chain": None,
        "chat_history": [],
        "interview_started": False
    }

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Setup Interview")

    student_name = st.text_input("Enter Student's Name", placeholder="e.g., Alex Doe")
    resume_file = st.file_uploader("Upload Student's Resume (PDF)", type="pdf")

    if st.button("Start Interview", use_container_width=True):
        # --- Validation ---
        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            st.error(f"Error: Knowledge base file not found at '{KNOWLEDGE_BASE_PATH}'. Please check the path in main.py.")
        elif not resume_file:
            st.warning("Please upload the student's resume.")
        elif not student_name:
            st.warning("Please enter the student's name.")
        else:
            with st.spinner("Processing documents and preparing interview..."):
                # Create a temporary directory to store uploaded resume
                if not os.path.exists("temp"):
                    os.makedirs("temp")
                
                resume_path = os.path.join("temp", resume_file.name)
                with open(resume_path, "wb") as f:
                    f.write(resume_file.getvalue())

                try:
                    # 1. Setup environment
                    cohere_api_key, google_api_key = app_logic.setup_environment()

                    # 2. Create retriever from the LOCAL knowledge base
                    retriever = app_logic.get_retriever(KNOWLEDGE_BASE_PATH, cohere_api_key)
                    
                    # 3. Analyze resume
                    queries = app_logic.analyze_resume(resume_path)

                    # 4. Get and STORE the context
                    context_data = app_logic.get_relevant_context(retriever, queries)
                    st.session_state.session_vars["context_data"] = context_data

                    # 5. Initialize chain
                    chain = app_logic.initialize_chain(google_api_key, student_name)
                    st.session_state.session_vars["chain"] = chain

                    # 6. Set up initial chat messages
                    initial_question = f"Hello {student_name}, thanks for coming in today. I've had a look at your resume. Could you start by telling me about a project you're particularly proud of?"
                    st.session_state.session_vars["chat_history"] = [
                        {"role": "assistant", "content": initial_question}
                    ]
                    st.session_state.session_vars["interview_started"] = True
                    st.success("Interview setup complete! You can start now.")
                
                except ValueError as e:
                    st.error(f"Error: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

# --- Chat Interface ---
if st.session_state.session_vars["interview_started"]:
    for message in st.session_state.session_vars["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Your response..."):
        st.session_state.session_vars["chat_history"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.spinner("Thinking..."):
            formatted_history = "\n".join(
                [f"Student: {msg['content']}" if msg['role'] == 'user' else f"Hiring Manager: {msg['content']}"
                for msg in st.session_state.session_vars["chat_history"]]
            )
            
            # Invoke the chain using the stored context
            manager_response = st.session_state.session_vars["chain"].invoke({
                "related_data": st.session_state.session_vars["context_data"],
                "chat_history": formatted_history
            })

            st.session_state.session_vars["chat_history"].append({"role": "assistant", "content": manager_response})
            with st.chat_message("assistant"):
                st.markdown(manager_response)
else:
    st.info("Please provide the details in the sidebar and click 'Start Interview'.")
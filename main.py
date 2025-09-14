import streamlit as st
import os
import app_logic # Import our backend logic
import re

# --- Configuration ---
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
        "interview_chain": None,
        "report_chain": None,
        "chat_history": [],
        "evaluation_history": [],
        "interview_started": False,
        "interview_finished": False
    }

# --- Helper Function to Parse Model Output ---
def parse_model_response(response):
    """Extracts content from <evaluation> and <question> tags."""
    evaluation = re.search(r"<evaluation>(.*?)</evaluation>", response, re.DOTALL)
    question = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
    
    eval_text = evaluation.group(1).strip() if evaluation else ""
    ques_text = question.group(1).strip() if question else "Sorry, I seem to have lost my train of thought. Could you please repeat your last point?"
    
    return eval_text, ques_text

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Setup Interview")

    student_name = st.text_input("Enter Student's Name", placeholder="e.g., Alex Doe")
    resume_file = st.file_uploader("Upload Student's Resume (PDF)", type="pdf")

    if st.button("Start Interview", use_container_width=True):
        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            st.error(f"Error: Knowledge base file not found at '{KNOWLEDGE_BASE_PATH}'.")
        elif not resume_file:
            st.warning("Please upload the student's resume.")
        elif not student_name:
            st.warning("Please enter the student's name.")
        else:
            with st.spinner("Processing documents and preparing interview..."):
                if not os.path.exists("temp"):
                    os.makedirs("temp")
                
                resume_path = os.path.join("temp", resume_file.name)
                with open(resume_path, "wb") as f:
                    f.write(resume_file.getvalue())

                try:
                    cohere_api_key, google_api_key = app_logic.setup_environment()
                    retriever = app_logic.get_retriever(KNOWLEDGE_BASE_PATH, cohere_api_key)
                    queries = app_logic.analyze_resume(resume_path)
                    context_data = app_logic.get_relevant_context(retriever, queries)
                    
                    st.session_state.session_vars["context_data"] = context_data
                    st.session_state.session_vars["interview_chain"] = app_logic.initialize_interview_chain(google_api_key, student_name)
                    st.session_state.session_vars["report_chain"] = app_logic.generate_feedback_report_chain(google_api_key)

                    initial_question = f"Hello {student_name}, thanks for coming in today. I've had a look at your resume. Could you start by telling me about a project you're particularly proud of?"
                    st.session_state.session_vars["chat_history"] = [{"role": "assistant", "content": initial_question}]
                    st.session_state.session_vars["interview_started"] = True
                    st.session_state.session_vars["interview_finished"] = False
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

    if not st.session_state.session_vars["interview_finished"]:
        if user_prompt := st.chat_input("Your response..."):
            st.session_state.session_vars["chat_history"].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.spinner("Thinking..."):
                formatted_history = "\n".join([f"Student: {msg['content']}" if msg['role'] == 'user' else f"Hiring Manager: {msg['content']}" for msg in st.session_state.session_vars["chat_history"]])
                
                raw_response = st.session_state.session_vars["interview_chain"].invoke({
                    "related_data": st.session_state.session_vars["context_data"],
                    "chat_history": formatted_history
                })
                
                evaluation, next_question = parse_model_response(raw_response)
                st.session_state.session_vars["evaluation_history"].append(evaluation)

                st.session_state.session_vars["chat_history"].append({"role": "assistant", "content": next_question})
                with st.chat_message("assistant"):
                    st.markdown(next_question)

    # --- Interview Controls ---
    if not st.session_state.session_vars["interview_finished"]:
        if st.button("Finish Interview", use_container_width=True):
            st.session_state.session_vars["interview_finished"] = True
            st.rerun()

    if st.session_state.session_vars["interview_finished"]:
        st.info("Interview concluded. Generating your feedback report...")
        with st.spinner("Generating Report..."):
            full_transcript = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in st.session_state.session_vars["chat_history"]])
            
            # --- CORRECTED LINE ---
            feedback_report = st.session_state.session_vars["report_chain"].invoke(full_transcript)

            st.subheader("Interview Performance Report")
            st.markdown(feedback_report)

            with st.expander("Show Interviewer's Internal Evaluations"):
                for i, thought in enumerate(st.session_state.session_vars["evaluation_history"]):
                    st.markdown(f"**Evaluation after response #{i+1}:**\n> {thought}")

else:
    st.info("Please provide the details in the sidebar and click 'Start Interview'.")

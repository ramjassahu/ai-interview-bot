import streamlit as st
import os
import app_logic  # Import our backend logic
import re

# --- Configuration ---
# Make sure the knowledge base file is in the same directory or provide a full path
KNOWLEDGE_BASE_PATH = "767888691-Excel-Interview-Questions.pdf" 

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Interview Bot 🤖",
    page_icon="🤖",
    layout="centered"
)

# --- Main App Interface ---
st.title("AI Interview Bot 🤖")
st.markdown("Enter the student's name and upload their resume to begin the interview process.")

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
    evaluation_match = re.search(r"<evaluation>(.*?)</evaluation>", response, re.DOTALL)
    question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
    
    eval_text = evaluation_match.group(1).strip() if evaluation_match else "No evaluation was generated."
    ques_text = question_match.group(1).strip() if question_match else "Sorry, I lost my train of thought. Could you tell me more about your last point?"
    
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
                # Create a temporary directory if it doesn't exist
                if not os.path.exists("temp"):
                    os.makedirs("temp")
                
                resume_path = os.path.join("temp", resume_file.name)
                with open(resume_path, "wb") as f:
                    f.write(resume_file.getvalue())

                try:
                    # Setup environment and API keys
                    cohere_api_key, google_api_key = app_logic.setup_environment()
                    
                    # Create retriever and analyze resume
                    retriever = app_logic.get_retriever(KNOWLEDGE_BASE_PATH, cohere_api_key)
                    queries = app_logic.analyze_resume(resume_path)
                    context_data = app_logic.get_relevant_context(retriever, queries)
                    
                    # Store data and chains in session state
                    st.session_state.session_vars["context_data"] = context_data
                    st.session_state.session_vars["interview_chain"] = app_logic.initialize_interview_chain(google_api_key, student_name)
                    st.session_state.session_vars["report_chain"] = app_logic.generate_feedback_report_chain(google_api_key)

                    # Set up initial chat message and state
                    initial_question = f"Hello {student_name}, thanks for coming in today. I've reviewed your resume. Could you start by telling me about a project you're particularly proud of?"
                    st.session_state.session_vars["chat_history"] = [{"role": "assistant", "content": initial_question}]
                    st.session_state.session_vars["evaluation_history"] = []
                    st.session_state.session_vars["interview_started"] = True
                    st.session_state.session_vars["interview_finished"] = False
                    st.success("Interview setup complete!")
                    st.rerun() # Rerun to display the chat interface immediately
                
                except ValueError as e:
                    st.error(f"Error: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

# --- Chat Interface ---
if st.session_state.session_vars["interview_started"]:
    # Display chat history
    for message in st.session_state.session_vars["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle ongoing interview
    if not st.session_state.session_vars["interview_finished"]:
        if user_prompt := st.chat_input("Your response..."):
            st.session_state.session_vars["chat_history"].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.spinner("Thinking..."):
                formatted_history = "\n".join([f"Student: {msg['content']}" if msg['role'] == 'user' else f"Hiring Manager: {msg['content']}" for msg in st.session_state.session_vars["chat_history"]])
                
                # Invoke the interview chain
                raw_response = st.session_state.session_vars["interview_chain"].invoke({
                    "related_data": st.session_state.session_vars["context_data"],
                    "chat_history": formatted_history
                })
                
                # Parse the response and update state
                evaluation, next_question = parse_model_response(raw_response)
                st.session_state.session_vars["evaluation_history"].append(evaluation)
                st.session_state.session_vars["chat_history"].append({"role": "assistant", "content": next_question})
                st.rerun()

    # --- Interview Controls and Report Generation ---
    if not st.session_state.session_vars["interview_finished"]:
        if st.button("Finish Interview", use_container_width=True):
            st.session_state.session_vars["interview_finished"] = True
            st.rerun()

    if st.session_state.session_vars["interview_finished"]:
        st.info("Interview concluded. Generating your feedback report...")
        with st.spinner("Generating Report..."):
            full_transcript = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in st.session_state.session_vars["chat_history"]])
            
            # Invoke the report generation chain
            feedback_report = st.session_state.session_vars["report_chain"].invoke({"chat_history": full_transcript})

            st.subheader("Interview Performance Report")
            st.markdown(feedback_report)

            # Show the AI's internal thoughts
            with st.expander("Show Interviewer's Internal Evaluations"):
                if st.session_state.session_vars["evaluation_history"]:
                    for i, thought in enumerate(st.session_state.session_vars["evaluation_history"]):
                        st.markdown(f"**Evaluation after response #{i+1}:**\n> {thought}")
                else:
                    st.write("No evaluations were recorded.")

else:
    st.info("Please provide the required details in the sidebar and click 'Start Interview'.")

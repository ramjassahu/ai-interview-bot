import streamlit as st
import os
import re
import app_logic  # Import backend logic

# --- Configuration ---
KNOWLEDGE_BASE_PATH = "767888691-Excel-Interview-Questions.pdf"

# --- Page Config ---
st.set_page_config(
    page_title="AI Interview Bot 🤖",
    page_icon="🤖",
    layout="centered"
)

# --- Main Title ---
st.title("AI Interview Bot 🤖")
st.markdown("Enter the student's name and upload their resume to begin.")

# --- Session State Initialization ---
if "session_vars" not in st.session_state:
    st.session_state.session_vars = {
        "context_data": "",
        "interview_chain": None,
        "report_chain": None,
        "chat_history": [],
        "evaluation_history": [],
        "interview_started": False,
        "interview_finished": False,
        "google_api_key": None,
    }

# --- Parse Model Output ---
def parse_model_response(response):
    """Extracts content from <evaluation> and <question> tags."""
    evaluation = re.search(r"<evaluation>(.*?)</evaluation>", response, re.DOTALL)
    question = re.search(r"<question>(.*?)</question>", response, re.DOTALL)

    eval_text = evaluation.group(1).strip() if evaluation else ""
    ques_text = (
        question.group(1).strip()
        if question
        else "Sorry, I seem to have lost my train of thought. Could you repeat that?"
    )

    return eval_text, ques_text


# --- Sidebar ---
with st.sidebar:
    st.header("Setup Interview")

    student_name = st.text_input("Enter Student's Name", placeholder="e.g., Alex Doe")
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

    if st.button("Start Interview", use_container_width=True):
        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            st.error(f"Error: Knowledge base not found at '{KNOWLEDGE_BASE_PATH}'.")
        elif not resume_file:
            st.warning("Please upload the student's resume.")
        elif not student_name:
            st.warning("Please enter the student's name.")
        else:
            with st.spinner("Preparing interview..."):
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

                    st.session_state.session_vars.update({
                        "context_data": context_data,
                        "interview_chain": app_logic.initialize_interview_chain(google_api_key, student_name),
                        "report_chain": app_logic.generate_feedback_report_chain(google_api_key),
                        "google_api_key": google_api_key,
                        "chat_history": [{"role": "assistant", "content": f"Hello {student_name}, thanks for coming in today. I've had a look at your resume. Could you start by telling me about a project you're particularly proud of?"}],
                        "interview_started": True,
                        "interview_finished": False,
                    })

                    st.success("Interview setup complete! You can start now.")

                except ValueError as e:
                    st.error(f"Error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


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
                formatted_history = "\n".join(
                    [f"Candidate: {msg['content']}" if msg["role"] == "user" else f"Interviewer: {msg['content']}" for msg in st.session_state.session_vars["chat_history"]]
                )

                raw_response = st.session_state.session_vars["interview_chain"].invoke({
                    "related_data": st.session_state.session_vars["context_data"],
                    "chat_history": formatted_history,
                })

                evaluation, next_question = parse_model_response(raw_response)
                st.session_state.session_vars["evaluation_history"].append(evaluation)

                st.session_state.session_vars["chat_history"].append({"role": "assistant", "content": next_question})
                with st.chat_message("assistant"):
                    st.markdown(next_question)

    # --- Finish Interview ---
    if not st.session_state.session_vars["interview_finished"]:
        if st.button("Finish Interview", use_container_width=True):
            st.session_state.session_vars["interview_finished"] = True
            st.rerun()

    # --- Final Report ---
    if st.session_state.session_vars["interview_finished"]:
        st.subheader("✅ Interview concluded. Generating feedback report...")

        with st.spinner("Generating report..."):
            feedback_report = app_logic.conclude_interview(
                chat_history=st.session_state.session_vars["chat_history"],
                google_api_key=st.session_state.session_vars["google_api_key"],
            )

            st.markdown("## 📊 Interview Performance Report")
            st.write(feedback_report)

            if st.session_state.session_vars["evaluation_history"]:
                st.markdown("## 📝 Interviewer's Internal Evaluations")
                for i, thought in enumerate(st.session_state.session_vars["evaluation_history"]):
                    with st.expander(f"Evaluation after response #{i+1}"):
                        st.write(thought)

else:
    st.info("Please provide details in the sidebar and click 'Start Interview'.")

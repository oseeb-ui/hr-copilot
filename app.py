import os
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from google import genai

load_dotenv()

st.set_page_config(page_title="HR Recruiter Copilot", layout="wide")

st.title("🎯 HR Recruiter Copilot (AI-Powered)")
st.caption("Recruiter-level ATS analysis, tailored cold outreach, and interview question preparation.")

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Job Posting")
    company_name = st.text_input("Target Company Name", placeholder="e.g. Acme Corp")
    recruiter_name = st.text_input("Recruiter / Manager Name (Optional)", placeholder="e.g. Sarah Jenkins")
    job_description = st.text_area("Paste Job Description", height=240)

with col2:
    st.subheader("2. Candidate Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    resume_text = ""
    if uploaded_file is not None:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                resume_text += extracted + " "
        st.success(f"Resume loaded ({len(resume_text.split())} words loaded)")

st.divider()

if st.button("Generate Complete Candidate Report", type="primary", use_container_width=True):
    if not job_description or not resume_text:
        st.warning("Please provide both a job description and a resume.")
    elif not client:
        st.error("Missing Gemini API Key. Please verify GEMINI_API_KEY in your .env file.")
    else:
        prompt = f"""
You are a Senior Corporate Recruiter and Talent Acquisition Lead.
Evaluate this candidate's resume against the target job description.

TARGET COMPANY: {company_name or 'the company'}
RECRUITER NAME: {recruiter_name or 'Hiring Team'}

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

Provide a structured evaluation in Markdown with the following 4 sections:
1. **ATS Match & Core Strengths**: Estimated fit score (out of 100), key strengths, and direct qualification matches.
2. **Skill Gaps & Weaknesses**: Exactly what the candidate is missing that might trigger recruiter rejection.
3. **Tailored Recruiter Outreach Pitch**: A concise, professional cold email (under 130 words) pitching the candidate directly to {recruiter_name or 'the recruiter'}.
4. **Top 3 Interview Questions**: The 3 hardest behavioral questions this specific candidate will likely be asked for this role.
"""
        report_placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("Analyzing profile via HR AI model..."):
                response = client.models.generate_content_stream(
                    model="gemini-flash-lite-latest",
                    contents=prompt,
                )

                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        report_placeholder.markdown(full_response + "▌")

                report_placeholder.markdown(full_response)

            # --- Export & Action Buttons Section ---
            st.divider()
            st.subheader("📋 Save & Export Results")

            col_down, col_copy = st.columns([1, 1])

            with col_down:
                st.download_button(
                    label="📥 Download Full Report (.txt)",
                    data=full_response,
                    file_name="HR_Evaluation_Report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with col_copy:
                with st.expander("📄 Click to View Clean Text (For 1-Click Copy)"):
                    st.caption("Hover over the box below and click the copy icon in the top-right corner:")
                    st.code(full_response, language="markdown")

        except Exception as e:
            st.error(f"Error encountered: {e}")
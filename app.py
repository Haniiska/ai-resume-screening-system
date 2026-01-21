import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Screening with Feedback",
    layout="wide"
)

# ---------------- BASIC STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: #e5e7eb;
}
h1, h2, h3 {
    color: #f8fafc;
}
textarea, input {
    background-color: #1e293b !important;
    color: #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SKILL LIST ----------------
SKILLS_LIST = [
    "python", "sql", "excel", "aws", "docker", "react",
    "java", "machine learning", "data analysis",
    "power bi", "tableau", "api", "git"
]

# ---------------- FUNCTIONS ----------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS_LIST if skill in text]

def generate_suggestions(missing_skills):
    return [f"Improve knowledge or projects in {skill}" for skill in missing_skills]

# ---------------- TITLE ----------------
st.title("📄 AI Resume Screening with Candidate Feedback")
st.caption("Shortlist resumes and provide clear feedback for rejected candidates")

st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    job_description = st.text_area(
        "Give the Job Description",
        height=220
    )

with col2:
    uploaded_files = st.file_uploader(
        "📤 Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

# ---------------- PROCESSING ----------------
if job_description and uploaded_files:
    resume_texts = []
    resume_names = []

    for file in uploaded_files:
        resume_texts.append(extract_text_from_pdf(file))
        resume_names.append(file.name)

    documents = [job_description] + resume_texts

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    results = pd.DataFrame({
        "Resume": resume_names,
        "Match %": np.round(scores * 100, 2)
    })

    results = results.sort_values(by="Match %", ascending=False).reset_index(drop=True)
    results["Rank"] = results.index + 1
    results["Status"] = results["Match %"].apply(
        lambda x: "Shortlisted" if x >= 40 else "Rejected"
    )

    # ---------------- SKILL ANALYSIS ----------------
    jd_skills = extract_skills(job_description)
    resume_skills_map = {
        name: extract_skills(text)
        for name, text in zip(resume_names, resume_texts)
    }

    missing_skills = []
    suggestions = []

    for _, row in results.iterrows():
        r_skills = resume_skills_map.get(row["Resume"], [])
        missing = list(set(jd_skills) - set(r_skills))
        missing_skills.append(missing)
        suggestions.append(generate_suggestions(missing))

    results["Missing Skills"] = missing_skills
    results["Suggestions"] = suggestions

    # ---------------- DISPLAY RESULTS ----------------
    st.subheader("📊 Screening Results")
    st.dataframe(
        results[["Rank", "Resume", "Match %", "Status"]],
        use_container_width=True
    )

    # ---------------- FEEDBACK SECTION ----------------
    st.subheader("📌 Feedback for Rejected Candidates")

    rejected = results[results["Status"] == "Rejected"]

    if rejected.empty:
        st.success("No rejected candidates 🎉")
    else:
        for _, row in rejected.iterrows():
            with st.expander(f"Feedback for {row['Resume']}"):
                st.write("❌ **Reason for Rejection**")
                st.write("Some key skills required for this role were missing.")

                st.write("🧩 **Missing Skills**")
                if row["Missing Skills"]:
                    st.write(", ".join(row["Missing Skills"]))
                else:
                    st.write("No major skill gaps identified.")

                st.write("🛠️ **Suggested Improvements**")
                for s in row["Suggestions"]:
                    st.write("- " + s)

else:
    st.info("Please upload resumes and paste the job description to start.")


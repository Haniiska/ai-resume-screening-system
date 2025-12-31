import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="📄",
    layout="wide"
)

# ---------------- DARK GREEN UI ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b3d2e, #145a32);
    color: #eafaf1;
}
h1, h2, h3 {
    color: #eafaf1;
}
textarea, input {
    background-color: #1c2833 !important;
    color: #eafaf1 !important;
    border-radius: 8px;
}
div[data-testid="stFileUploader"] {
    background-color: #1c2833;
    padding: 16px;
    border-radius: 10px;
}
.stDataFrame {
    background-color: #1c2833;
}
.result-card {
    background: #1c2833;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📄 Resume Screening AI")
st.caption("AI-assisted resume screening prototype for HR teams")

# ---------------- FUNCTIONS ----------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

def calculate_match(jd_text, resumes):
    documents = [jd_text] + resumes
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(documents)
    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return scores

# ---------------- LAYOUT ----------------
left, right = st.columns([1, 1.2])

# ---------------- JOB DESCRIPTION ----------------
with left:
    st.subheader("📝 Job Description")
    job_description = st.text_area(
        label="Paste Job Description",
        placeholder="Paste job description here...",
        height=220
    )

    st.subheader("📤 Upload Resumes (PDF)")
    uploaded_files = st.file_uploader(
        "Upload minimum 1 resume (max ~100)",
        type=["pdf"],
        accept_multiple_files=True
    )

# ---------------- PROCESS ----------------
with right:
    st.subheader("📊 Screening Results")

    if job_description and uploaded_files:
        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            resume_texts.append(extract_text_from_pdf(file))
            resume_names.append(file.name)

        scores = calculate_match(job_description.lower(), resume_texts)

        results = []
        for name, score in zip(resume_names, scores):
            percentage = round(score * 100, 2)
            status = "Shortlisted ✅" if percentage >= 40 else "Rejected ❌"
            results.append([name, percentage, status])

        df = pd.DataFrame(
            results,
            columns=["Resume", "Match %", "Status"]
        ).sort_values(by="Match %", ascending=False).reset_index(drop=True)

        # Best match card
        best = df.iloc[0]
        st.markdown(f"""
        <div class="result-card">
            🏆 <b>Best Match</b><br><br>
            <b>{best['Resume']}</b><br>
            Match Score: <b>{best['Match %']}%</b>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(df, use_container_width=True)

    else:
        st.info("👉 Paste Job Description and upload at least one resume to start screening.")

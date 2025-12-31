import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="📄",
    layout="wide"
)

# ---------------- CUSTOM UI (PRO THEME) ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e5e7eb;
}
h1, h2, h3 {
    color: #f8fafc;
}
textarea, input {
    background-color: #1e293b !important;
    color: #e5e7eb !important;
    border-radius: 10px;
}
div[data-testid="stFileUploader"] {
    background-color: #1e293b;
    padding: 16px;
    border-radius: 12px;
}
.stDataFrame {
    background-color: #020617;
}
.result-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 16px;
    border-left: 5px solid #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("## 📄 Resume Screening AI")
st.markdown(
    "AI-assisted resume screening tool for HR teams — **prototype**"
)

st.divider()

# ---------------- LAYOUT ----------------
left, right = st.columns([1, 1.4])

# ---------------- JOB DESCRIPTION ----------------
with left:
    st.subheader("📝 Job Description")
    job_description = st.textarea(
        "Paste Job Description",
        height=220,
        placeholder="Paste JD here..."
    )

    st.subheader("📤 Upload Resumes (PDF)")
    uploaded_files = st.file_uploader(
        "Upload minimum 1 resume (max ~100)",
        type=["pdf"],
        accept_multiple_files=True
    )

# ---------------- FUNCTIONS ----------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------------- PROCESSING ----------------
with right:
    st.subheader("📊 Screening Results")

    if job_description and uploaded_files:
        resumes_text = []
        resume_names = []

        for file in uploaded_files:
            resumes_text.append(extract_text_from_pdf(file))
            resume_names.append(file.name)

        documents = [job_description] + resumes_text

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity_scores = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:]
        )[0]

        results = pd.DataFrame({
            "Resume": resume_names,
            "Match %": np.round(similarity_scores * 100, 2)
        })

        results = results.sort_values(by="Match %", ascending=False).reset_index(drop=True)
        results["Rank"] = results.index + 1
        results["Status"] = results["Match %"].apply(
            lambda x: "Shortlisted" if x >= 40 else "Rejected"
        )

        best = results.iloc[0]

        # -------- BEST MATCH CARD --------
        st.markdown(f"""
        <div class="result-card">
            🏆 <b>Best Match</b><br><br>
            <b>{best['Resume']}</b><br>
            Match Score: <b>{best['Match %']}%</b>
        </div>
        """, unsafe_allow_html=True)

        # -------- TABLE --------
        st.dataframe(
            results[["Rank", "Resume", "Match %", "Status"]],
            use_container_width=True
        )

    else:
        st.info("👉 Paste Job Description and upload at least one resume")


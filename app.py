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

# ---------------- DARK GREEN THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f3d2e, #145a32);
    color: #ecf0f1;
}

h1, h2, h3 {
    color: #e8f5e9;
}

textarea, input {
    background-color: #1e272e !important;
    color: #ecf0f1 !important;
}

div[data-testid="stFileUploader"] {
    background-color: #1e272e;
    padding: 15px;
    border-radius: 10px;
}

.result-card {
    background: #1e272e;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.4);
}

.stDataFrame {
    background-color: #1e272e;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📄 Resume Screening AI")
st.caption("AI-assisted resume screening tool for HR teams")

st.markdown("---")

# ---------------- LAYOUT ----------------
left, right = st.columns([1, 1.4])

# ---------------- JOB DESCRIPTION ----------------
with left:
    st.subheader("📝 Job Description")

    # ✅ FIXED textarea (NO ERROR)
    job_description = st.text_area(
        label="Paste Job Description",
        placeholder="Paste job description here...",
        height=220
    )

    st.subheader("📤 Upload Resumes (PDF)")
    uploaded_files = st.file_uploader(
        label="Upload minimum 1 resume (max ~100)",
        type=["pdf"],
        accept_multiple_files=True
    )

# ---------------- FUNCTIONS ----------------
def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------------- PROCESSING ----------------
with right:
    if job_description and uploaded_files:

        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            resume_texts.append(extract_text(file))
            resume_names.append(file.name)

        documents = [job_description] + resume_texts

        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(documents)

        similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
        scores = similarities * 100

        results = pd.DataFrame({
            "Resume": resume_names,
            "Match %": scores.round(2)
        }).sort_values(by="Match %", ascending=False)

        results["Status"] = results["Match %"].apply(
            lambda x: "Shortlisted" if x >= 40 else "Rejected"
        )

        best = results.iloc[0]

        # -------- BEST MATCH CARD --------
        st.markdown(f"""
        <div class="result-card">
            <h3>🏆 Best Match</h3>
            <b>{best['Resume']}</b><br>
            Match Score: <b>{best['Match %']}%</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Screening Results")

        st.dataframe(
            results.reset_index(drop=True),
            use_container_width=True
        )

    else:
        st.info("👉 Paste Job Description and upload at least one resume to see results.")

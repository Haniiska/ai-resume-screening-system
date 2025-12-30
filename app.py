import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Resume Screening AI",
    layout="wide",
    page_icon="📄"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top left, #2b1055, #0f0c29);
    color: white;
}
.main {
    background: transparent;
}
h1, h2, h3 {
    color: #ffffff;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}
.best {
    background: linear-gradient(135deg, #7f00ff, #e100ff);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 20px;
}
table {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ------------------ FUNCTIONS ------------------
def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.lower()

def calculate_similarity(jd, resumes):
    docs = [jd] + resumes
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(docs)
    scores = cosine_similarity(matrix[0:1], matrix[1:])[0]
    return scores

# ------------------ TITLE ------------------
st.markdown("<h1>📄 Resume Screening AI</h1>", unsafe_allow_html=True)
st.markdown("AI-assisted resume screening tool for HR teams")

# ------------------ LAYOUT ------------------
left, right = st.columns([1.1, 1.9])

# ------------------ JOB DESCRIPTION ------------------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📝 Job Description")

    job_description = st.text_area(
        "Paste Job Description",
        height=220,
        placeholder="Paste JD here..."
    )

    st.subheader("📤 Upload Resumes (PDF)")
    resumes = st.file_uploader(
        "Upload minimum 1 resume (max ~100)",
        type="pdf",
        accept_multiple_files=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ PROCESSING ------------------
with right:
    if job_description and resumes:

        resume_texts = []
        resume_names = []

        for r in resumes:
            resume_texts.append(extract_text(r))
            resume_names.append(r.name)

        scores = calculate_similarity(job_description.lower(), resume_texts)
        results = []

        for name, score in zip(resume_names, scores):
            percent = round(score * 100, 2)
            status = "Shortlisted" if percent >= 20 else "Rejected"
            results.append((name, percent, status))

        results.sort(key=lambda x: x[1], reverse=True)
        best = results[0]

        # -------- BEST MATCH --------
        st.markdown(f"""
        <div class="best">
        🏆 <b>Best Match</b><br><br>
        {best[0]}<br>
        Match Score: <b>{best[1]}%</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------- RESULTS TABLE --------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Screening Results")

        st.markdown("""
        <table>
        <tr>
            <th>Rank</th>
            <th>Resume</th>
            <th>Match %</th>
            <th>Status</th>
        </tr>
        """, unsafe_allow_html=True)

        for i, r in enumerate(results, start=1):
            color = "#00ff99" if r[2] == "Shortlisted" else "#ff4d4d"
            st.markdown(f"""
            <tr>
                <td>{i}</td>
                <td>{r[0]}</td>
                <td>{r[1]}%</td>
                <td style="color:{color}; font-weight:bold">{r[2]}</td>
            </tr>
            """, unsafe_allow_html=True)

        st.markdown("</table></div>", unsafe_allow_html=True)

    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.info("👈 Paste Job Description and upload at least one resume")
        st.markdown("</div>", unsafe_allow_html=True)

import streamlit as st
import PyPDF2
import pandas as pd
import re
import smtplib
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== CONFIG =====================
SENDER_EMAIL = "yourgmail@gmail.com"     # 👈 CHANGE
APP_PASSWORD = "your_app_password_here"  # 👈 CHANGE
ROLE_NAME = "AI / Data Analyst Intern"

# ===================== FUNCTIONS =====================

def extract_text_from_pdf(pdf):
    reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.lower()

def extract_email(text):
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group() if match else None

def send_email(to_email, subject, body):
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)

def generate_rejection_email(name, missing_skills):
    skills = ", ".join(missing_skills)

    return f"""
Hi {name},

Thank you for applying for the {ROLE_NAME} role.

After reviewing your resume, we found that some key skills required for this role are currently missing.

Missing skills:
{skills}

Suggested improvements:
- Work on projects related to the above skills
- Add hands-on experience or certifications
- Update your resume with relevant keywords

This feedback is shared to help you improve and apply again confidently.

Best wishes,  
HR Team
"""

# ===================== UI =====================

st.set_page_config(page_title="AI Resume Screening with Feedback", layout="wide")

st.title("📄 AI Resume Screening with Candidate Feedback")
st.caption("Shortlist resumes and provide clear feedback for rejected candidates")

job_description = st.text_area("📌 Paste Job Description")

uploaded_files = st.file_uploader(
    "📤 Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if job_description and uploaded_files:

    resumes = []
    emails = []

    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)
        emails.append(extract_email(text))

    documents = [job_description.lower()] + resumes

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    results = []

    for i, file in enumerate(uploaded_files):
        score = round(similarity_scores[i] * 100, 2)
        status = "Shortlisted" if score >= 50 else "Rejected"

        results.append({
            "Resume": file.name,
            "Match %": score,
            "Status": status,
            "Email": emails[i]
        })

    df = pd.DataFrame(results).sort_values(by="Match %", ascending=False)

    st.subheader("📊 Screening Results")
    st.dataframe(df, use_container_width=True)

    # ===================== FEEDBACK + EMAIL =====================

    st.subheader("📩 Feedback for Rejected Candidates")

    for idx, row in df.iterrows():
        if row["Status"] == "Rejected":
            with st.expander(f"❌ {row['Resume']}"):

                resume_text = resumes[df.index.get_loc(idx)]
                jd_words = set(job_description.lower().split())
                resume_words = set(resume_text.split())

                missing_skills = list(jd_words - resume_words)[:5]

                st.write("**Missing Skills:**")
                st.write(missing_skills)

                email_body = generate_rejection_email(
                    name=row["Resume"].replace(".pdf", ""),
                    missing_skills=missing_skills
                )

                st.code(email_body)

                if row["Email"]:
                    if st.button(f"📨 Send Email to {row['Email']}", key=row["Resume"]):
                        send_email(
                            row["Email"],
                            "Application Feedback",
                            email_body
                        )
                        st.success("✅ Email sent successfully")
                else:
                    st.warning("⚠️ Email not found in resume")

else:
    st.info("📌 Please upload resumes and paste the job description to start.")

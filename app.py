import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.set_page_config(page_title="Resume-Job Match Analyzer", page_icon="📄")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_keywords(text):
    common_skills = [
        'python', 'sql', 'excel', 'power bi', 'tableau', 'machine learning',
        'communication', 'data analysis', 'statistics', 'r programming',
        'aws', 'deep learning', 'nlp', 'data visualization', 'analytics',
        'java', 'javascript', 'react', 'node.js', 'git', 'docker',
        'leadership', 'teamwork', 'project management'
    ]
    text_lower = text.lower()
    return set(skill for skill in common_skills if skill in text_lower)

st.title("📄 Resume-Job Match Analyzer")
st.write("Upload your resume and paste a job description to see how well they match.")

resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
jd_text = st.text_area("Paste Job Description", height=200)

if st.button("Analyze Match"):
    if resume_file and jd_text:
        resume_text = extract_text_from_pdf(resume_file)

        resume_emb = model.encode([resume_text])
        jd_emb = model.encode([jd_text])
        score = cosine_similarity(resume_emb, jd_emb)[0][0] * 100

        resume_skills = extract_keywords(resume_text)
        jd_skills = extract_keywords(jd_text)
        matching = resume_skills & jd_skills
        missing = jd_skills - resume_skills

        st.metric("Match Score", f"{score:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Matching Skills")
            st.write(matching if matching else "None found")
        with col2:
            st.subheader("❌ Missing Skills")
            st.write(missing if missing else "None")
    else:
        st.warning("Please upload a resume and paste a job description.")

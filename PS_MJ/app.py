import streamlit as st
import pandas as pd
import glob
import re
import os
import requests
import time
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# --- UTILITY: TYPING EFFECT ---
def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.01)

# --- UTILITY: CSS LOADER ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- CONFIG & CREDENTIALS ---
GROQ_API_KEY = "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"
ADZUNA_ID = "4bbe2a1d"
ADZUNA_KEY = "1de4b30cfddaabfc8892d3b934d0d3a3"
CURRENT_MODEL = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="SkillSync Analytics", layout="wide")
local_css("app.css")

# --- ENGINE INITIALIZATION ---
@st.cache_resource
def startup_engine():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_files = glob.glob("*.csv")
    master_list = []
    for filename in all_files:
        if "chatbot_qa" in filename: continue
        try:
            df = pd.read_csv(filename, encoding='latin1')
            df.columns = [f"col_{i}" for i in range(len(df.columns))]
            df = df.rename(columns={"col_1": "Role", "col_2": "Job_Description", "col_3": "Skills", "col_4": "Projects"})
            df = df.fillna("Not Specified").astype(str)
            df['Industry'] = filename.replace(".csv", "")
            master_list.append(df[["Role", "Job_Description", "Skills", "Projects", "Industry"]])
        except: continue
    
    try:
        qa_db = pd.read_csv('chatbot_qa_dataset.csv').fillna("").astype(str)
        qa_embeddings = model.encode(qa_db['question'].tolist(), convert_to_tensor=True)
    except:
        qa_db, qa_embeddings = pd.DataFrame(), None

    return model, pd.concat(master_list, ignore_index=True), qa_db, qa_embeddings

semantic_model, master_db, qa_db, qa_embeddings = startup_engine()

# --- HELPER FUNCTIONS ---
def extract_text(file):
    if file.name.endswith('.pdf'):
        pdf = PdfReader(file)
        return "".join([p.extract_text() or "" for p in pdf.pages])
    return " ".join([p.text for p in Document(file).paragraphs])

def get_live_jobs(role, location="india"):
    url = f"https://api.adzuna.com/v1/api/jobs/in/search/1"
    params = {
        "app_id": ADZUNA_ID, "app_key": ADZUNA_KEY,
        "results_per_page": 5, "what": role, "where": location,
        "content-type": "application/json"
    }
    try:
        res = requests.get(url, params=params)
        return res.json().get('results', []) if res.status_code == 200 else []
    except: return []

# --- SESSION STATE INITIALIZATION ---
if "analysis_done" not in st.session_state: st.session_state.analysis_done = False
if "messages" not in st.session_state: st.session_state.messages = []
if "audit" not in st.session_state: st.session_state.audit = ""
if "roadmap" not in st.session_state: st.session_state.roadmap = ""

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.subheader("Intelligence Metrics")
    st.write(f"Roles Indexed: {len(master_db)}")
    st.write(f"Knowledge Base: {len(qa_db)} entries")
    st.divider()

# --- MAIN INTERFACE ---
st.title("🛡️ SkillSync Analytics: Career Architect")

c1, c2 = st.columns(2)
with c1: target_job = st.text_area("Target Job Description", height=200, placeholder="Paste requirements...")
with c2: resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if st.button("RUN ANALYSIS"):
    if resume_file and target_job:
        with st.spinner("Analyzing Professional Vectors..."):
            res_raw = extract_text(resume_file)
            
            # 1. Matching Logic
            master_db['Fingerprint'] = master_db['Role'] + " " + master_db['Skills']
            kb_embeds = semantic_model.encode(master_db['Fingerprint'].tolist(), convert_to_tensor=True)
            res_embed = semantic_model.encode(res_raw, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(res_embed, kb_embeds)
            
            if scores.max().item() < 0.18: 
                st.warning("Low professional alignment detected.")
            else:
                match = master_db.iloc[scores.argmax().item()]
                tfidf = TfidfVectorizer().fit_transform([re.sub(r'[^a-z0-9\s]', '', res_raw.lower()), re.sub(r'[^a-z0-9\s]', '', target_job.lower())])
                
                # 2. Store findings in Session State
                st.session_state.match_data = match
                st.session_state.ats_score = (cosine_similarity(tfidf)[0][1] * 100)
                st.session_state.status = "CRITICAL_LOW" if st.session_state.ats_score < 40 else "OPTIMIZED"
                
                # 3. Generate static content for persistent display
                st.session_state.audit = client.chat.completions.create(
                    model=CURRENT_MODEL, 
                    messages=[{"role": "user", "content": f"Audit this resume for {match['Role']}: {res_raw[:1500]}"}]
                ).choices[0].message.content
                
                st.session_state.roadmap = client.chat.completions.create(
                    model=CURRENT_MODEL, 
                    messages=[{"role": "user", "content": f"Create a roadmap for {match['Role']} in {match['Industry']}."}]
                ).choices[0].message.content
                
                st.session_state.analysis_done = True
                st.session_state.messages = []
                st.rerun()

# --- RESULTS DISPLAY (PERSISTENT) ---
# --- RESULTS DISPLAY (WITH INTELLIGENT STREAMING) ---
if st.session_state.analysis_done:
    m = st.session_state.match_data
    ats = st.session_state.ats_score
    st.divider()
    
    col_a, col_b = st.columns(2)
    if st.session_state.status == "CRITICAL_LOW":
        col_a.metric("ATS Alignment", f"{round(ats, 1)}%", delta="- Critical Gap", delta_color="inverse")
        st.error(f"⚠️ **High Rejection Risk:** Alignment is too low for the target job.")
        st.info(f"💡 **Strategic Pivot:** You are better suited for: **{m['Role']}** ({m['Industry']})")
    else:
        col_a.metric("ATS Alignment", f"{round(ats, 1)}%", delta="Optimal", delta_color="normal")
        col_b.success(f"Optimized Career Fit: {m['Role']}")

    # --- THE MAGIC FIX FOR "ROBOTIC" LOADING ---
    st.subheader("🕵️ Senior Recruiter Audit")
    # If this is the FIRST time we see this audit, stream it. 
    # If we are just refreshing for the chatbot, show it instantly.
    if "audit_streamed" not in st.session_state:
        st.write_stream(stream_data(st.session_state.audit))
        st.session_state.audit_streamed = True
    else:
        st.markdown(st.session_state.audit)

    st.subheader("🚀 Tactical Roadmap")
    if "roadmap_streamed" not in st.session_state:
        st.write_stream(stream_data(st.session_state.roadmap))
        st.session_state.roadmap_streamed = True
    else:
        st.markdown(st.session_state.roadmap)
        
    # IMPORTANT: In your "RUN ANALYSIS" button block, 
    # you MUST add these lines to reset the streamers:
    # if "audit_streamed" in st.session_state: del st.session_state.audit_streamed
    # if "roadmap_streamed" in st.session_state: del st.session_state.roadmap_streamed
    # --- CHATBOT SECTION ---
    st.divider()
    st.subheader("💬 SkillSync Knowledge Assistant")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if user_chat := st.chat_input("Ask a follow-up question..."):
        st.session_state.messages.append({"role": "user", "content": user_chat})
        with st.chat_message("user"): st.markdown(user_chat)
        
        with st.chat_message("assistant"):
            q_embed = semantic_model.encode(user_chat, convert_to_tensor=True)
            qa_res = util.pytorch_cos_sim(q_embed, qa_embeddings)
            
            if qa_res.max().item() > 0.5:
                context = qa_db.iloc[qa_res.argmax().item()]['answer']
                sys_msg = f"Use ONLY this context: {context}. Be professional. Refuse non-career topics."
            else:
                sys_msg = "You are a specialized Career AI. Please keep questions professional."

            ans = client.chat.completions.create(
                model=CURRENT_MODEL,
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_chat}]
            ).choices[0].message.content
            st.write_stream(stream_data(ans))
        st.session_state.messages.append({"role": "assistant", "content": ans})
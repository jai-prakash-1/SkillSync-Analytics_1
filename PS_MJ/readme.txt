# 🛡️ SkillSync Analytics: AI-Powered Career Architect

SkillSync Analytics is a high-performance career optimization platform that uses **Hybrid RAG (Retrieval-Augmented Generation)** to bridge the gap between job seekers and industry requirements.

## 🚀 Core Features
* **Semantic Resume Audit:** Deep analysis of resume gaps using local vector embeddings.
* **Strategic Roadmaps:** AI-generated 3-step plans for career advancement.
* **ATS Alignment Scoring:** Real-time calculation of resume-to-job compatibility.
* **Strategic Gatekeeper:** Automatic fit-detection that suggests alternative career paths if ATS scores are critically low.
* **Live Market Integration:** Real-time job fetching via the Adzuna API.

## 🛠️ Technical Stack
* **Frontend:** Streamlit (Minimalist UX)
* **LLM Inference:** Groq Llama 3.1 (8B Instant)
* **Embedding Model:** HuggingFace `all-MiniLM-L6-v2` (Local Vectorization)
* **Backend:** Python (Pandas, Scikit-Learn, PyPDF2)
* **Data Source:** Proprietary CSV database (440+ Roles, 3,236 QA Pairs)

## 📦 Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 🧩 Architecture
The system utilizes **Cosine Similarity** to map user resumes into a 384-dimensional vector space, comparing them against indexed professional benchmarks to find the mathematical "Nearest Neighbor."
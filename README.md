# Vector-Rag-3gpp - Initialize
Local 3GPP Vector‑Based RAG Engineering Tool
(Offline, Free, Daily‑Use, Reproducible)
This document consolidates all working knowledge from our discussion into one clean, step‑by‑step reference. It is intended to be shared with other engineers and used on different systems without paid services.
The goal is to build a local, offline 3GPP chatbot that:
•	Uses vector‑based RAG (FAISS + embeddings)
•	Works with PDF and Word (.docx) 3GPP specs
•	Runs entirely locally (no OpenAI / no cloud)
•	Is safe for daily engineering usage (no hallucination)
•	Is reproducible and isolated using a Python virtual environment
No debug history, no errors, only what actually worked.

1. High‑Level Architecture
1     3GPP Specs (PDF / DOCX)
2             ↓
3     Text Extraction & Chunking
4             ↓
5     Sentence Embeddings (local CPU)
6             ↓
7     FAISS Vector Index (per spec)
8             ↓
9     Query Routing (TS‑aware)
10             ↓
11     Context Retrieval
12             ↓
13     Local LLM (Ollama – Mistral / Qwen)
14     
Key design principles:
•	One index per TS (no cross‑spec pollution)
•	Strict RAG prompting (answer only from retrieved context)
•	Local‑only inference (no paid APIs)

2. System Requirements
Hardware (recommended)
•	CPU: ≥ 8 cores (20 vCPU works very well)
•	RAM: ≥ 16 GB
•	Disk: 10–20 GB free
•	GPU: Not required
OS
•	Linux or WSL2 (tested and recommended)

3. Project Directory Layout
Create a clean project directory:
1     spec_rag/
2     ├── 3gpp_rag_tool.py
3     ├── requirements.txt
4     ├── data/
5     │   ├── TS_38_331/
6     │   │   └── TS_38_331_Rel17_i51.docx
7     │   ├── TS_38_413/
8     │   │   └── TS_38_413_Rel17_i50.docx
9     │   └── TS_28_552/
10     │       └── TS_28_552_Rel17_i60.docx
11     └── vector_db/        # auto‑created (do not create manually)
12     
Important notes:
•	One TS per folder
•	Remove any Zone.Identifier files copied from Windows

4. Python Environment (Isolated)
Create and activate a virtual environment:
1     python3 -m venv .venv
2     source .venv/bin/activate
3     
This guarantees no impact on other users or system Python.

5. Dependency Versions (Known‑Good, Stable)
Create requirements.txt:
1     langchain==0.1.16
2     langchain-core==0.1.46
3     langchain-community==0.0.33
4     
5     sentence-transformers==2.6.1
6     faiss-cpu==1.8.0
7     pypdf==4.2.0
8     unstructured==0.13.6
9     python-docx==1.1.0
10     tiktoken==0.7.0
11     
Install:
1     pip install -r requirements.txt
2     
These exact versions avoid all LangChain compatibility issues and are suitable for long‑term internal tools.

6. Required NLP Resources (One‑Time Setup)
The Word loader (unstructured) requires NLTK resources. Run once inside the environment:
1     python - <<EOF
2     import nltk
3     nltk.download("punkt")
4     nltk.download("punkt_tab")
5     nltk.download("averaged_perceptron_tagger")
6     nltk.download("averaged_perceptron_tagger_eng")
7     EOF
8     
After this step, ingestion works fully offline.

7. Local LLM Setup (No Paid Services)
Install Ollama
1     curl -fsSL https://ollama.com/install.sh | sh
2     
Verify:
1     ollama --version
2     
Pull a Model (One‑Time)
Recommended models for 3GPP specs:
1     ollama pull mistral
2     # or
3     ollama pull qwen2.5:7b
4     
These models work well on CPU and are accurate for technical text when used with RAG.

8. Core Python Tool (Single File)
3gpp_rag_tool.py responsibilities:
•	PDF / DOCX loading
•	Clause‑aligned chunking
•	Embedding generation (HuggingFace, CPU)
•	FAISS index creation (per TS)
•	Deterministic TS routing
•	Context‑only answering using Ollama
LLM Configuration (Local)
1     from langchain_community.llms import Ollama
2     
3     llm = Ollama(
4         model="mistral",
5         temperature=0
6     )
7     
FAISS Load (Required Flag)
1     FAISS.load_local(
2         db_dir,
3         embeddings,
4         allow_dangerous_deserialization=True
5     )
6     
Safe because the vector DB is locally generated and trusted.
CPU‑Only Guard (Recommended)
At the very top of the file:
1     import os
2     os.environ["CUDA_VISIBLE_DEVICES"] = ""
3     os.environ["TOKENIZERS_PARALLELISM"] = "false"
4     

9. Ingestion Workflow (One‑Time per Spec)
Ingest a single spec:
1     python 3gpp_rag_tool.py ingest 38_331
2     
Repeat for each TS:
1     python 3gpp_rag_tool.py ingest 38_413
2     python 3gpp_rag_tool.py ingest 28_552
3     
Expected output (example):
1     ✅ TS 38_331 indexed successfully
2     
After ingestion, the directory will exist:
1     vector_db/TS_38_331/
2     
Do not re‑ingest unless the document changes.

10. Daily Usage (Normal Operation)
Start the chatbot:
1     python 3gpp_rag_tool.py
2     
Example queries:
•	What does TS 38.331 specify?
•	What is prbUsedDl in TS 28.552?
•	Which spec defines NGAP procedures?
•	What are Event A measurement triggers in 38.331?
Characteristics:
•	No internet access
•	No API keys
•	Low CPU usage
•	Deterministic, review‑safe answers

11. Question Style (Important for Correct Answers)
3GPP specs:
•	Describe scope and definitions, not motivations
Preferred query style:
•	✅ What does TS 28.552 specify?
•	✅ What is prbUsedDl in TS 28.552?
•	✅ Which events are defined for measurement reporting in 38.331?
Avoid overly vague questions like:
•	❌ Why is this used?
The tool is intentionally conservative to avoid hallucination.

12. Token Awareness (Optional Monitoring)
Token usage comes only from the LLM step, not from FAISS.
Typical local usage:
•	Prompt (context): 2–4K tokens
•	Answer: 100–300 tokens
Local models have no cost impact.

13. Maintenance & Sharing
Share with Others
•	Share the entire spec_rag/ directory
•	Others only need to:
o	Create a venv
o	Install requirements
o	Install Ollama
Update a Spec
1     rm -rf vector_db/TS_38_331
2     python 3gpp_rag_tool.py ingest 38_331
3     
Other specs remain untouched.

14. Final Characteristics
✅ Fully local / offline ✅ Free (no API cost) ✅ Reproducible ✅ Safe for daily engineering ✅ Scales to 30+ specs ✅ Suitable for reviews and design discussions

This document represents the clean, stable baseline.
All future enhancements (clause citation, diff between releases, UI) can be built on top without changing this foundation.

Example :
 python 3gpp_rag_tool.py

📡 3GPP RAG Engineering Chatbot
Type 'exit' to quit

You> what is event d1?

Bot>
🤔 Thinking... planning the reasoning steps.
⏳ Press Ctrl+C to stop the current model operation and return to the prompt.
✅ Planning complete: 32 step(s) detected./
🧠 Reasoning through 32 step(s)...
   • Retrieved 4 excerpt(s) from TS 38_331 for step 1.
   → Step 1/32: To decompose the query "What is event d1?" into step-by-step reasoning tasks, we can break it down as follows:
⏳ Press Ctrl+C to stop the current model operation and return to the prompt.
⏳ Running model... Press Ctrl+C to stop. \

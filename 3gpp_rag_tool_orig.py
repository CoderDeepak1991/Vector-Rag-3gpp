"""
3GPP Vector-Based RAG Chatbot (Daily Engineering Tool)

- Single-file, end-to-end
- Isolated Python environment friendly
- Supports PDF and Word (.docx)
- Per-spec vector indexes
- Strict, no-hallucination answers
- Designed for daily 3GPP engineering usage

Author: You
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import sys
from typing import List


# ---- FIXED LANGCHAIN IMPORTS ----
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.llms import OpenAI
# from langchain_openai import OpenAI
from langchain_community.llms import Ollama


# =========================
# CONFIGURATION
# =========================

DATA_DIR = "./data"              # data/TS_38_331/*.pdf
VECTOR_DB_DIR = "./vector_db"    # vector_db/TS_38_331/
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenAI key must be set via environment variable
# export OPENAI_API_KEY=xxxx
LLM_TEMPERATURE = 0


SYSTEM_PROMPT = """
You are a 3GPP standards expert.

Rules:
- Answer strictly from the provided context
- Do NOT infer or guess
- Always mention TS number and clause (if visible)
- If the answer is not defined, explicitly say:
  "Not specified in the provided 3GPP text"
"""


# =========================
# UTILITIES
# =========================

def clause_aware_splitter():
    """
    Clause-oriented text splitter.
    Keeps chunks aligned with 3GPP clause structure.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n\d+\.\d+\.\d+",
            "\n\d+\.\d+",
            "\n",
            " "
        ]
    )


def detect_ts_from_query(query: str) -> str:
    """
    Deterministic routing based on keywords.

    Priority:
      1. Explicit TS number in query (e.g. "ts 38.331", "38331", "38-331")
      2. Domain keyword matching
      3. Return "" if no match (caller decides how to handle)
    """
    q = query.lower()

    # --- 1. Direct TS number reference ---
    # Matches patterns like: ts38.331, 38.331, 38-331, 38_331, 38331
    _TS_NUMBER_MAP = {
        "38.331": "38_331", "38331": "38_331", "38-331": "38_331",
        "38.413": "38_413", "38413": "38_413", "38-413": "38_413",
        "28.552": "28_552", "28552": "28_552", "28-552": "28_552",
        "32.425": "32_425", "32425": "32_425", "32-425": "32_425",
    }
    for pattern, ts_key in _TS_NUMBER_MAP.items():
        if pattern in q:
            return ts_key

    # --- 2. Keyword-based routing ---
    # TS 38.331 — NR RRC: radio resource control, measurements, events
    #
    # Event families covered:
    #   A (intra/inter-freq NR), B (inter-RAT), C (CGI), D (distance/NTN),
    #   F (CLI SRS-RSRP), I (inter-freq RSSI), V (CLI RSSI),
    #   W (WLAN), X (CGI variant), Y (channel occupancy), Z (PRS)
    #
    # Regex matches both spaced and compact forms:
    #   "event d1", "event-d1", "event_d1", "eventd1"  → all match
    _rrc_event_re = re.compile(
        r"event[\s_-]?([abcdfivwxyz]\d+)",
        re.IGNORECASE
    )
    _rrc_keywords = [
        "rrc", "radio resource control",
        # Keep specific, not generic 'measurement'/'handover' — those are claimed by 32.425
        "measurement event", "measurement gap", "measurement report",
        "ho trigger", "cell reselection", "sib", "system information",
        "paging", "rlc", "pdcp", "srb", "drb", "beam", "ssb",
        "serving cell", "a3 offset", "hysteresis",
        "time to trigger", "ttt",
    ]
    if _rrc_event_re.search(q) or any(k in q for k in _rrc_keywords):
        return "38_331"

    # TS 38.413 — NG-RAN: NGAP (N2 interface between gNB and AMF)
    if any(k in q for k in [
        "ngap", "ng-ap", "n2 interface", "amf", "core network",
        "initial context", "ue context", "pdu session",
        "path switch", "handover preparation", "handover command",
        "overload", "nas transport", "ran", "ng setup",
        "ue radio capability", "location reporting",
    ]):
        return "38_413"

    # TS 28.552 — 5G NR OAM performance measurements
    if any(k in q for k in [
        "oam", "o&m", "fault management", "performance measurement",
        "configuration management", "pm data", "kpi",
        "measurement counter", "measurement job", "gnb pm",
        "cell availability", "prb utilization", "throughput",
        "dl prb", "ul prb", "ho success", "admission",
        "mean number", "active ue", "drb setup",
    ]):
        return "28_552"

    # TS 32.425 — UTRAN Performance Measurements (3G/UMTS RAN PM)
    # Covers RNC, NodeB, IuB, HSDPA/HSUPA, RAB, RACH, cell-level counters
    _utran_pm_event_re = re.compile(
        r"pm[\.\s_-]?32[\.\s_-]?425",
        re.IGNORECASE
    )
    if _utran_pm_event_re.search(q) or any(k in q for k in [
        "utran", "umts", "rnc", "nodeb", "node b", "iub", "iur",
        "hsdpa", "hsupa", "hsdsch", "hsupa", "e-dch", "hs-dsch",
        "rab setup", "rab release", "radio access bearer",
        "rach", "fach", "dch", "cpich", "wcdma",
        "cell traffic", "cell availability utran",
        "uu interface", "spreading factor",
        "rnti", "rnc capacity", "soft handover", "softer handover",
        "compressed mode", "srns",
        # HO measurement / neighbour cell counters (section 4.3.x of 32.425)
        "ho measurement", "handover measurement",
        "neighbour cell basis", "neighbor cell basis",
        "neighbouring cell", "neighboring cell",
        "intra-rnc", "inter-rnc", "intra-frequency handover", "inter-frequency handover",
        "inter-system handover", "inter-rat handover",
        "ho attempt", "ho success", "ho failure", "ho preparation",
        "handover attempt", "handover success", "handover failure",
        "measurement on neighbour", "measurement on neighbor",
    ]):
        return "32_425"

    return ""


# =========================
# INGESTION
# =========================

def load_documents(ts_dir: str, ts_name: str):
    docs = []

    for file in os.listdir(ts_dir):
        path = os.path.join(ts_dir, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            continue

        loaded = loader.load()
        for d in loaded:
            d.metadata = {
                "ts": ts_name,
                "source": file
            }
        docs.extend(loaded)

    return docs


def ingest_ts(ts_name: str):
    print(f"\n📥 Ingesting TS {ts_name}")

    ts_dir = os.path.join(DATA_DIR, f"TS_{ts_name}")
    db_dir = os.path.join(VECTOR_DB_DIR, f"TS_{ts_name}")

    if not os.path.isdir(ts_dir):
        print(f"❌ Data directory not found: {ts_dir}")
        return

    docs = load_documents(ts_dir, ts_name)

    splitter = clause_aware_splitter()
    chunks = splitter.split_documents(docs)

    print(f"  → Chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(chunks, embeddings)

    os.makedirs(db_dir, exist_ok=True)
    vectordb.save_local(db_dir)

    print(f"✅ TS {ts_name} indexed successfully")


def ingest_all():
    for d in os.listdir(DATA_DIR):
        if d.startswith("TS_"):
            ingest_ts(d.replace("TS_", ""))


# =========================
# CHATBOT
# =========================

def load_vector_db(ts_name: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64},
    )
    db_dir = os.path.join(VECTOR_DB_DIR, f"TS_{ts_name}")
    return FAISS.load_local(
        db_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )

def normalize_query(q: str, ts: str = "") -> str:
    ql = q.lower()
    # 38.331 expansions
    if "measurement events" in ql:
        return q + " event triggered reporting A1 A2 A3 A4 A5 A6 B1 B2"
    if "why 28.552" in ql:
        return q + " purpose scope specification defines"
    # 32.425 expansions — neighbour-cell HO measurement counters
    if ts == "32_425" and any(k in ql for k in [
        "ho measurement", "handover measurement",
        "neighbour cell", "neighbor cell", "neighbouring cell", "neighboring cell",
        "ho attempt", "ho success", "ho failure",
    ]):
        return (
            q + " handover measurement neighbour cell basis"
            " intra-RNC inter-RNC inter-system inter-frequency"
            " HO attempt success failure preparation"
            " 4.3.1.3 performance measurement UTRAN"
        )
    return q

def answer_query(query: str):
    ts = detect_ts_from_query(query)

    if not ts:
        return "Unable to determine relevant 3GPP specification from the query."

    db_path = os.path.join(VECTOR_DB_DIR, f"TS_{ts}")
    if not os.path.isdir(db_path):
        return f"Vector DB for TS {ts} not found. Please ingest it first."

    vectordb = load_vector_db(ts)
    query = normalize_query(query, ts)
    docs = vectordb.similarity_search(query, k=8)

    context = "\n\n".join(
        f"[TS {d.metadata['ts']} | {d.metadata['source']}]\n{d.page_content}"
        for d in docs
    )

    llm = Ollama(
        model="qwen2.5:7b",
        temperature=0
    )


    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt).strip()


# =========================
# CLI
# =========================

def chat_loop():
    print("\n📡 3GPP RAG Engineering Chatbot")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You> ").strip()
        if query.lower() in ("exit", "quit"):
            break

        print("\nBot>")
        print(answer_query(query))
        print("")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "ingest":
            if len(sys.argv) == 3:
                ingest_ts(sys.argv[2])
            else:
                ingest_all()
            return

    chat_loop()


if __name__ == "__main__":
    main()

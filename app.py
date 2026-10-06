from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os

# Load secrets (works in all environments)
try:
    # Streamlit Cloud / local with secrets.toml
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "JWT_SECRET" in st.secrets:
        os.environ["JWT_SECRET"] = st.secrets["JWT_SECRET"]
except Exception:
    # Render / other platforms (env vars already set)
    pass
import streamlit as st
import os, shutil, sqlite3, bcrypt
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

DB_FILE = "app.db"
CHROMA_DIR = "./chroma_db"
UPLOAD_DIR = "./uploads"
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- INIT (once) ----------
@st.cache_resource
def init_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    return embeddings, llm

embeddings, llm = init_models()


# ---------- DB ----------
def init_db():
    con = sqlite3.connect(DB_FILE)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    con.commit(); con.close()

def db():
    con = sqlite3.connect(DB_FILE, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def hash_pw(p): return bcrypt.hashpw(p.encode()[:72], bcrypt.gensalt()).decode()
def verify_pw(p, h):
    try: return bcrypt.checkpw(p.encode()[:72], h.encode())
    except: return False


# ---------- RAG ----------
def ingest_file(user_id, file_path):
    docs = PyPDFLoader(file_path).load() if file_path.endswith(".pdf") \
           else TextLoader(file_path, encoding="utf-8").load()
    for d in docs:
        d.metadata["user_id"] = user_id
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"user_{user_id}",
        persist_directory=CHROMA_DIR,
    )
    return len(chunks)


def ask(user_id, question, history):
    vectorstore = Chroma(
        collection_name=f"user_{user_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    for m in history:
        (memory.chat_memory.add_user_message if m["role"]=="user"
         else memory.chat_memory.add_ai_message)(m["content"])

    strict_prompt = PromptTemplate.from_template(
        """You are a strict document-based assistant.
Answer ONLY using the context below.
If the answer is NOT in the context, respond EXACTLY with:
"I don't have information about that in your documents."

Context:
{context}

Question: {question}
Answer:"""
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": strict_prompt},
    )
    result = chain.invoke({"question": question})
    sources = [{"file": os.path.basename(d.metadata.get("source","?")),
                "page": d.metadata.get("page")} for d in result["source_documents"]]
    return result["answer"], sources


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
init_db()

st.title("🤖 RAG Chatbot")
st.caption("Upload your docs and ask questions")

# Session state
if "user" not in st.session_state:
    st.session_state.user = None

# ---------- LOGIN / REGISTER ----------
if st.session_state.user is None:
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Login"):
            con = db()
            row = con.execute("SELECT * FROM users WHERE username=?", (u,)).fetchone()
            con.close()
            if row and verify_pw(p, row["password_hash"]):
                st.session_state.user = dict(row)
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New username", key="reg_u")
        p = st.text_input("New password", type="password", key="reg_p")
        if st.button("Register"):
            con = db()
            try:
                con.execute("INSERT INTO users (username, password_hash) VALUES (?,?)",
                            (u, hash_pw(p)))
                con.commit()
                st.success("Account created! Now login.")
            except sqlite3.IntegrityError:
                st.error("Username taken")
            con.close()

    st.stop()

# ---------- MAIN CHAT ----------
user = st.session_state.user
user_id = user["id"]

# Sidebar
with st.sidebar:
    st.write(f"👤 **{user['username']}**")
    st.divider()

    st.subheader("📁 Upload documents")
    uploaded = st.file_uploader("Choose PDF or TXT", type=["pdf", "txt"])
    if uploaded:
        user_dir = os.path.join(UPLOAD_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        dest = os.path.join(user_dir, uploaded.name)
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        with st.spinner("Embedding..."):
            n = ingest_file(user_id, dest)
        st.success(f"✅ {uploaded.name} ({n} chunks)")

    st.divider()
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    files = os.listdir(user_dir) if os.path.isdir(user_dir) else []
    if files:
        st.subheader("📂 Your files")
        for f in files:
            st.write(f"• {f}")

    st.divider()
    if st.button("🧹 Clear chat"):
        con = db()
        con.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
        con.commit(); con.close()
        st.session_state.messages = []
        st.rerun()

    if st.button("🚪 Logout"):
        st.session_state.user = None
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    con = db()
    rows = con.execute(
        "SELECT role, content FROM messages WHERE user_id=? ORDER BY id",
        (user_id,)
    ).fetchall()
    con.close()
    st.session_state.messages = [{"role": r["role"], "content": r["content"]} for r in rows]

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Save user message
    con = db()
    con.execute("INSERT INTO messages (user_id, role, content, created_at) VALUES (?,?,?,?)",
                (user_id, "user", prompt, datetime.utcnow().isoformat()))
    con.commit(); con.close()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history = [m for m in st.session_state.messages[:-1]]
            answer, sources = ask(user_id, prompt, history)
            st.write(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for i, s in enumerate(sources, 1):
                        page = f" (p.{s['page']})" if s["page"] is not None else ""
                        st.write(f"{i}. {s['file']}{page}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

    con = db()
    con.execute("INSERT INTO messages (user_id, role, content, created_at) VALUES (?,?,?,?)",
                (user_id, "assistant", answer, datetime.utcnow().isoformat()))
    con.commit(); con.close()

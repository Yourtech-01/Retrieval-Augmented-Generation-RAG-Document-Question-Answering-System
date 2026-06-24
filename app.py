"""
ui/app.py  —  Streamlit chat interface for the RAG Q&A system
Run: streamlit run ui/app.py
"""

import os, requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Q&A", page_icon="", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.5rem}
.src{background:#f8f9fa;border-left:3px solid #4361ee;border-radius:0 6px 6px 0;
     padding:10px 14px;margin:6px 0;font-size:13px}
.src-lbl{font-weight:500;color:#4361ee;font-size:12px;margin-bottom:4px}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Document Q&A")
    top_k = st.slider("Sources to retrieve", 1, 10, 5)
    st.markdown("---")
    st.markdown("### Indexed documents")
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=3).json()
        for d in docs["documents"]:
            st.markdown(f"**{d['name']}** — {d['chunks']} chunks")
        st.caption(f"Total: {docs['total_chunks']} chunks")
    except:
        st.warning("API not reachable.")
    st.markdown("---")
    try:
        h = requests.get(f"{API_URL}/health", timeout=2).json()
        st.caption(f"LLM: `{h.get('llm_model','N/A')}`")
        st.caption(f"Embed: `{h.get('embed_model','N/A')}`")
    except: pass
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("# Ask your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} retrieved)"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f'<div class="src"><div class="src-lbl">Source {i}: {s["source"]} (chunk {s["chunk_index"]})</div>{s["excerpt"]}</div>',
                                unsafe_allow_html=True)

if q := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching + generating..."):
            try:
                resp = requests.post(f"{API_URL}/ask",
                    json={"question": q, "top_k": top_k}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(data["answer"])
                    if data.get("sources"):
                        with st.expander(f"Sources ({len(data['sources'])} retrieved)"):
                            for i, s in enumerate(data["sources"], 1):
                                st.markdown(f'<div class="src"><div class="src-lbl">Source {i}: {s["source"]} (chunk {s["chunk_index"]})</div>{s["excerpt"]}</div>',
                                            unsafe_allow_html=True)
                    st.session_state.messages.append({"role":"assistant","content":data["answer"],"sources":data.get("sources",[])})
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

import os
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

st.set_page_config(page_title="Doc Q&A Bot", page_icon="📄🤖")
st.title("📄🤖 Document Q&A Bot")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set. Set it in your terminal and restart VS Code.")
    st.stop()

client = OpenAI(api_key=api_key)

def extract_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        return "\n".join(pages_text)

    return ""

st.sidebar.header("1) Upload a document")
uploaded = st.sidebar.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
doc_text = extract_text(uploaded)

if uploaded and not doc_text.strip():
    st.sidebar.warning("Uploaded file has no extractable text (some PDFs are scanned images).")

st.sidebar.header("2) Ask questions")
st.sidebar.caption("Tip: Upload a short resume, job description, or any text doc.")

# Keep a short portion to avoid huge prompts
MAX_CHARS = 12000
doc_context = doc_text[:MAX_CHARS]

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask a question about the uploaded document...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    if not doc_context.strip():
        answer = "Please upload a .txt or .pdf with text first."
        st.chat_message("assistant").write(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Reading the document..."):
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You answer questions ONLY using the provided document context. "
                            "If the answer isn't in the document, say you can't find it."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"DOCUMENT CONTEXT:\n{doc_context}\n\nQUESTION:\n{question}",
                    },
                ]

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                )
                answer = resp.choices[0].message.content
                st.write(answer)

        st.session_state.history.append({"role": "assistant", "content": answer})

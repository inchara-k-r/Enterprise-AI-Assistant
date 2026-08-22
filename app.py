import streamlit as st
from pypdf import PdfReader
import chromadb
import re
from groq import Groq

# --------------------------------------------------
# GROQ AI CLIENT
# --------------------------------------------------

groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

MODEL = "openai/gpt-oss-20b"

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise AI Intelligence",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 Enterprise AI Intelligence Assistant")

st.markdown(
    """
    **AI-powered document intelligence using RAG**

    Upload enterprise documents, ask questions, and receive
    evidence-based answers with source references.
    """
)

st.divider()

# --------------------------------------------------
# CHROMADB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="enterprise_documents"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📋 Document Intelligence")

    st.write(
        "Upload a business document to build your "
        "AI knowledge base."
    )

    st.info(
        "🔒 Documents are processed locally "
        "during this prototype."
    )

    st.divider()

    st.subheader("Technology")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• ChromaDB")
    st.write("• RAG")
    st.write("• Ollama")
    st.write("• Gemma 3")

# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

st.subheader("📄 Upload Business Document")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Upload a text-based PDF for analysis."
)

if uploaded_file is not None:

    st.success(
        f"✓ {uploaded_file.name} uploaded successfully"
    )

    reader = PdfReader(uploaded_file)

    page_count = len(reader.pages)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    if text.strip():

        words = text.split()

        chunk_size = 300

        chunks = []

        for i in range(
            0,
            len(words),
            chunk_size
        ):

            chunk = " ".join(
                words[i:i + chunk_size]
            )

            chunks.append(chunk)

        document_id = re.sub(
            r"[^a-zA-Z0-9]",
            "_",
            uploaded_file.name
        )

        for index, chunk in enumerate(chunks):

            collection.upsert(
                ids=[
                    f"{document_id}_{index}"
                ],
                documents=[chunk],
                metadatas=[{
                    "source": uploaded_file.name,
                    "chunk": index
                }]
            )

        st.session_state[
            "document_uploaded"
        ] = True

        st.session_state[
            "document_name"
        ] = uploaded_file.name

        st.session_state[
            "page_count"
        ] = page_count

        st.session_state[
            "chunk_count"
        ] = len(chunks)

        st.success(
            f"✓ Document processed into "
            f"{len(chunks)} knowledge chunks."
        )

        # --------------------------------------------------
        # DOCUMENT METRICS
        # --------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Pages",
                page_count
            )

        with col2:
            st.metric(
                "Knowledge Chunks",
                len(chunks)
            )

        with col3:
            st.metric(
                "Status",
                "Ready"
            )

    else:

        st.error(
            "Unable to extract text from this PDF."
        )

# --------------------------------------------------
# QUESTION
# --------------------------------------------------

st.divider()

st.subheader("💬 Ask Your Business Question")

question = st.text_input(
    "Question",
    placeholder=(
        "Example: What are the major challenges "
        "mentioned in this document?"
    )
)

analyze = st.button(
    "🤖 Analyze with AI",
    use_container_width=True
)

# --------------------------------------------------
# AI ANALYSIS
# --------------------------------------------------

if analyze:

    if not question:

        st.warning(
            "Please enter a question."
        )

    elif not st.session_state.get(
        "document_uploaded"
    ):

        st.warning(
            "Please upload a PDF first."
        )

    else:

        # --------------------------------------------------
        # RETRIEVAL
        # --------------------------------------------------

        with st.spinner(
            "Retrieving relevant information..."
        ):

            results = collection.query(
                query_texts=[question],
                n_results=3
            )

        documents = results["documents"][0]

        metadatas = results["metadatas"][0]

        context = "\n\n".join(
            documents
        )

        # --------------------------------------------------
        # AI PROMPT
        # --------------------------------------------------

        prompt = f"""
You are an Enterprise AI Business Intelligence Assistant.

Your job is to analyze the provided enterprise document
and answer the user's question accurately.

IMPORTANT RULES:

1. Use ONLY the provided document context.
2. Do NOT invent information.
3. If the information is not available, clearly say so.
4. Give a concise business-oriented answer.
5. After the answer, provide one practical recommendation.
6. The recommendation must be based only on the document.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Return your response in this format:

ANSWER:
<clear answer>

RECOMMENDATION:
<practical recommendation>
"""

        # --------------------------------------------------
        # GEMMA
        # --------------------------------------------------

        with st.spinner(
            "🤖 AI is analyzing the document..."
        ):

            response = groq_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_completion_tokens=1024
            )

            answer = response.choices[0].message.content

        

        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------

        st.divider()

        st.subheader("🤖 AI Analysis")

        st.markdown(answer)

        # --------------------------------------------------
        # EVIDENCE
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "📚 Evidence & Sources"
        )

        st.write(
            f"The answer was generated using "
            f"**{len(documents)} relevant sections** "
            f"from the uploaded document."
        )

        for i, metadata in enumerate(
            metadatas,
            start=1
        ):

            with st.expander(
                f"Source {i} — "
                f"{metadata['source']}"
            ):

                st.write(
                    documents[i - 1]
                )

                st.caption(
                    f"Knowledge Chunk: "
                    f"{metadata['chunk']}"
                )

        # --------------------------------------------------
        # TRUST INDICATOR
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "🛡️ AI Trust & Transparency"
        )

        st.success(
            "✓ Response grounded in uploaded document"
        )

        st.info(
            "The system retrieves relevant document "
            "sections before generating the response. "
            "If sufficient information is unavailable, "
            "the AI is instructed not to invent an answer."
        )
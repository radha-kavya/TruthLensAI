
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FC;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.section-heading {
    color: #101B46 !important;
    font-size: 23px;
    font-weight: 700;
    margin-top: 15px;
}

.stButton button {
    background-color: #1565C0;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton button:hover {
    background-color: #00ACC1;
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E0E7F0;
}

.verdict-card {
    padding: 22px;
    border-radius: 14px;
    margin: 15px 0;
    font-size: 23px;
    font-weight: 700;
    text-align: center;
}

.supported {
    background-color: #DFF5E5;
    color: #176B35;
}

.contradicted {
    background-color: #FCE4E4;
    color: #A32121;
}

.partial {
    background-color: #FFF2D6;
    color: #8A5700;
}

.insufficient {
    background-color: #E8EAF0;
    color: #424A5A;
}

.footer {
    text-align: center;
    color: #667085;
    padding: 15px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

V4_DIR = BASE_DIR / "v4_reranking"
DOCUMENTS_DIR = V4_DIR / "documents"
CHROMA_DIR = V4_DIR / "chroma_db"


# ==================================================
# 4. LOAD API KEY
# ==================================================

load_dotenv(V4_DIR / ".env")

api_key = os.getenv("GOOGLE_API_KEY")


# ==================================================
# 5. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🔎 TruthLens AI")

    st.markdown("---")

    st.subheader("About the Project")

    st.write(
        "TruthLens AI is a multi-document intelligence "
        "and fact-checking application."
    )

    st.markdown("---")

    st.subheader("Technology Stack")

    st.markdown("""
    - Python
    - LangChain
    - Gemini LLM
    - ChromaDB
    - HuggingFace Embeddings
    - Cross-Encoder Reranking
    - Streamlit
    """)

    st.markdown("---")

    st.subheader("Fact-Checking Verdicts")

    st.markdown("""
    🟢 Supported

    🔴 Contradicted

    🟡 Partially Supported

    ⚪ Insufficient Evidence
    """)

    st.markdown("---")

    st.caption("AI Engineer Project | 2026")


# ==================================================
# 6. LOAD MODELS AND CHROMADB
# ==================================================

@st.cache_resource(
    show_spinner="Loading AI models and document database..."
)
def load_resources():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma(
        collection_name="truthlens_v4",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    if vector_store._collection.count() == 0:

        pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(
                "No PDF files found in v4_reranking/documents."
            )

        documents = []

        for pdf_file in pdf_files:

            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = pdf_file.name
                page.metadata["document_type"] = "Research Report"

            documents.extend(pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(documents)

        vector_store.add_documents(chunks)

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return vector_store, reranker


# ==================================================
# 7. GEMINI LLM
# ==================================================

def load_llm():

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Check v4_reranking/.env."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0
    )


# ==================================================
# 8. FACT-CHECKING PROMPT
# ==================================================

prompt = ChatPromptTemplate.from_template("""

You are TruthLens AI, a document-based fact-checking assistant.

Analyze the user's claim using only the evidence provided below.

CLAIM:
{claim}

RERANKED EVIDENCE:
{context}

Instructions:

1. Choose exactly one verdict:
   - Supported
   - Contradicted
   - Partially Supported
   - Insufficient Evidence

2. Explain your reasoning clearly.
3. Mention source documents and page numbers.
4. Do not invent information.
5. Identify evidence limitations.
6. Distinguish direct emissions from lifecycle emissions
   when relevant.
7. If evidence is incomplete or unrelated, choose
   Insufficient Evidence rather than guessing.

Use this report format:

Claim:
Verdict:
Evidence and Explanation:
Source Documents:
Limitations:

""")


def create_chain(llm):
    return prompt | llm | StrOutputParser()


# ==================================================
# 9. CLAIM INPUT
# ==================================================

st.markdown(
    '<div class="section-heading">🔍 Fact-Check a Claim</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter a factual statement below. TruthLens AI will "
    "search the research documents and evaluate the evidence."
)

with st.form("claim_form"):

    claim = st.text_area(
        "Enter a claim to fact-check:",
        placeholder=(
            "Example: Global electric car sales exceeded "
            "17 million in 2024."
        ),
        height=110
    )

    analyze_button = st.form_submit_button(
        "🔍 Analyze Claim",
        use_container_width=True
    )


# ==================================================
# 10. RUN RAG PIPELINE
# ==================================================

if analyze_button:

    if not claim.strip():

        st.warning("Please enter a claim before analyzing.")

    else:

        try:

            # LOAD RESOURCES

            with st.spinner("Loading AI models and database..."):
                vector_store, reranker = load_resources()

            # RETRIEVAL

            with st.spinner("Retrieving relevant evidence..."):

                retriever = vector_store.as_retriever(
                    search_kwargs={"k": 20}
                )

                retrieved_docs = retriever.invoke(claim)

            if not retrieved_docs:
                st.warning("No relevant evidence was found.")
                st.stop()

            # RERANKING

            with st.spinner("Reranking evidence..."):

                pairs = [
                    (claim, doc.page_content)
                    for doc in retrieved_docs
                ]

                scores = reranker.predict(pairs)

                ranked_results = sorted(
                    zip(scores, retrieved_docs),
                    key=lambda item: float(item[0]),
                    reverse=True
                )

                top_results = ranked_results[:5]

            # PREPARE CONTEXT

            context_parts = []

            for score, doc in top_results:

                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")

                page_number = (
                    page + 1 if isinstance(page, int) else page
                )

                context_parts.append(
                    f"""
Source: {source}
Page: {page_number}
Evidence:
{doc.page_content}
"""
                )

            context = "\n\n".join(context_parts)

            # ======================================
            # 11. RETRIEVED EVIDENCE
            # ======================================

            st.divider()

            st.markdown(
                '<div class="section-heading">📚 Retrieved Evidence</div>',
                unsafe_allow_html=True
            )

            st.success(
                "Document retrieval and reranking completed successfully."
            )

            st.caption(
                f"Retrieved: {len(retrieved_docs)} chunks | "
                f"Reranked: {len(top_results)} chunks"
            )

            for index, (score, doc) in enumerate(
                top_results,
                start=1
            ):

                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")

                page_number = (
                    page + 1 if isinstance(page, int) else page
                )

                with st.expander(
                    f"Evidence {index} | {source} | Page {page_number}",
                    expanded=(index == 1)
                ):

                    st.write("Source:", source)
                    st.write("Page:", page_number)

                    st.write(
                        f"Cross-Encoder Relevance Score: {float(score):.4f}"
                    )

                    st.markdown("**Evidence Content**")

                    st.text_area(
                        label=f"Evidence Text {index}",
                        value=doc.page_content,
                        height=250,
                        disabled=True,
                        key=f"evidence_text_{index}"
                    )

            # ======================================
            # 12. PIPELINE METRICS
            # ======================================

            st.divider()

            st.markdown(
                '<div class="section-heading">⚙️ Pipeline Information</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Retrieved Chunks", len(retrieved_docs))
            col2.metric("Reranked Chunks", len(top_results))
            col3.metric(
                "PDF Database",
                vector_store._collection.count()
            )

            # ======================================
            # 13. GEMINI ANSWER GENERATION
            # ======================================

            st.divider()

            st.markdown(
                '<div class="section-heading">📋 AI Fact-Checking Report</div>',
                unsafe_allow_html=True
            )

            try:

                with st.spinner("Generating AI fact-checking report..."):

                    llm = load_llm()
                    chain = create_chain(llm)

                    response = chain.invoke({
                        "claim": claim,
                        "context": context
                    })

                # EXTRACT VERDICT

                verdict_match = re.search(
                    r"Verdict\s*:\s*"
                    r"(Supported|Contradicted|Partially Supported|Insufficient Evidence)",
                    response,
                    re.IGNORECASE
                )

                if verdict_match:

                    verdict = verdict_match.group(1)

                    verdict_styles = {
                        "Supported": ("supported", "🟢 Supported"),
                        "Contradicted": ("contradicted", "🔴 Contradicted"),
                        "Partially Supported": ("partial", "🟡 Partially Supported"),
                        "Insufficient Evidence": ("insufficient", "⚪ Insufficient Evidence")
                    }

                    style, display_verdict = verdict_styles.get(
                        verdict,
                        ("insufficient", verdict)
                    )

                    st.markdown(
                        f"""
                        <div class="verdict-card {style}">
                            {display_verdict}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # DISPLAY REPORT AS NATIVE TEXT

                st.subheader("Complete AI Report")

                st.text_area(
                    label="Fact-Checking Analysis",
                    value=response,
                    height=350,
                    disabled=True,
                    key="final_ai_report"
                )

                st.success("Fact-checking analysis completed successfully.")

            except Exception as gemini_error:

                error_message = str(gemini_error)

                if (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED" in error_message
                    or "quota" in error_message.lower()
                ):

                    st.warning(
                        "Gemini API free-tier quota has been exhausted. "
                        "The final AI report cannot be generated right now."
                    )

                    st.info(
                        "Document retrieval and reranking completed. "
                        "The retrieved evidence is displayed above."
                    )

                    st.markdown(
                        "[Check Gemini API usage and limits](https://ai.dev/rate-limit)"
                    )

                elif "GOOGLE_API_KEY" in error_message:

                    st.error(
                        "Google API key is missing. Check "
                        "v4_reranking/.env."
                    )

                else:

                    st.error(
                        f"Gemini generation failed: {error_message}"
                    )

        except Exception as e:

            st.error(
                "The retrieval or reranking pipeline could not be completed."
            )

            st.exception(e)


# ==================================================
# 14. FOOTER
# ==================================================

st.divider()

st.markdown("""
<div class="footer">
    🔎 TruthLens AI | Multi-Document Intelligence & Fact-Checking
    <br>
    Built with Python, LangChain, ChromaDB, Gemini and Streamlit
</div>
""", unsafe_allow_html=True)

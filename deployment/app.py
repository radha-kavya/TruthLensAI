
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


# Different retrieval counts for each PDF

RETRIEVE_PER_PDF = {
    "electric_vehicles.pdf": 5,
    "EV_Handbook_Final_14Oct.pdf": 10,
    "GlobalEVOutlook2025.pdf": 15
}

# Default retrieval count for any new PDF

DEFAULT_RETRIEVE_PER_PDF = 10

# Final evidence chunks selected from each PDF

TOP_PER_PDF = 2


# ==================================================
# 4. LOAD API KEY
# ==================================================

load_dotenv(V4_DIR / ".env")

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
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
        "and fact-checking application. It analyzes claims "
        "using evidence from multiple research PDFs."
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
# 6. HELPER FUNCTION
# ==================================================

def normalize_filename(source):

    return str(source).replace("\\", "/").split("/")[-1]


# ==================================================
# 7. LOAD MODELS AND INDEX ALL PDFS
# ==================================================

@st.cache_resource(show_spinner="Loading AI models and PDF database...")
def load_resources():

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCUMENTS_DIR}"
        )

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in v4_reranking/documents."
        )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

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

    # Get existing ChromaDB metadata.

    existing_data = vector_store.get(
        include=["metadatas"]
    )

    existing_metadata = existing_data.get("metadatas") or []

    # Find already indexed PDF filenames.

    existing_sources = {
        normalize_filename(metadata.get("source"))
        for metadata in existing_metadata
        if metadata and metadata.get("source")
    }

    # Index PDFs that are not already present.

    new_documents = []

    for pdf_file in pdf_files:

        if pdf_file.name in existing_sources:
            continue

        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:

            page.metadata["source"] = pdf_file.name
            page.metadata["document_type"] = "Research Report"

        new_documents.extend(pages)

    if new_documents:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(new_documents)

        if chunks:
            vector_store.add_documents(chunks)

    # Refresh metadata after indexing.

    existing_data = vector_store.get(
        include=["metadatas"]
    )

    existing_metadata = existing_data.get("metadatas") or []

    # Map each PDF filename to the exact source value
    # stored in ChromaDB.

    source_values = {}

    for metadata in existing_metadata:

        if not metadata:
            continue

        source = metadata.get("source")

        if source:

            filename = normalize_filename(source)
            source_values[filename] = source

    indexed_pdf_files = [
        pdf_file.name
        for pdf_file in pdf_files
        if pdf_file.name in source_values
    ]

    # Load Cross-Encoder reranker.

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return (
        vector_store,
        reranker,
        indexed_pdf_files,
        source_values
    )


# ==================================================
# 8. GEMINI LLM
# ==================================================

def load_llm():

    if not api_key:

        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to Streamlit Secrets "
            "or your local v4_reranking/.env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0
    )


# ==================================================
# 9. FACT-CHECKING PROMPT
# ==================================================

prompt = ChatPromptTemplate.from_template("""
You are TruthLens AI, a document-based fact-checking assistant.

Analyze the user's claim using only the evidence provided below.

CLAIM:
{claim}

EVIDENCE FROM MULTIPLE PDF DOCUMENTS:
{context}

Instructions:

1. Choose exactly one verdict:

   - Supported
   - Contradicted
   - Partially Supported
   - Insufficient Evidence

2. Examine evidence from every source document provided.

3. Do not assume that every document supports the claim.

4. Identify evidence that is directly relevant to the claim.

5. Ignore unrelated evidence when deciding the verdict.

6. If documents disagree, clearly explain the disagreement
   and identify the relevant sources.

7. Do not treat missing information as proof that a claim is false.

8. Explain your reasoning clearly.

9. Mention source document names and page numbers.

10. Do not invent information or citations.

11. Distinguish direct emissions from lifecycle emissions
    when relevant.

12. If evidence is incomplete or unrelated, choose
    Insufficient Evidence rather than guessing.

Use this report format:

Claim:
Verdict:
Evidence and Explanation:
Comparison Across Documents:
Source Documents:
Limitations:
""")


def create_chain(llm):

    return prompt | llm | StrOutputParser()


# ==================================================
# 10. CLAIM INPUT
# ==================================================

st.markdown(
    '<div class="section-heading">🔍 Fact-Check a Claim</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter a factual statement below. TruthLens AI will "
    "search all indexed PDFs independently, combine the "
    "retrieved evidence and evaluate the claim."
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
# 11. RUN RAG PIPELINE
# ==================================================

if analyze_button:

    if not claim.strip():

        st.warning("Please enter a claim before analyzing.")

    else:

        try:

            # LOAD RESOURCES

            with st.spinner("Loading AI models and indexing PDFs..."):

                (
                    vector_store,
                    reranker,
                    indexed_pdf_files,
                    source_values
                ) = load_resources()

            if not indexed_pdf_files:

                st.error("No indexed PDF documents were found.")
                st.stop()

            # ======================================
            # RETRIEVE FROM EACH PDF SEPARATELY
            # ======================================

            retrieved_by_pdf = {}

            with st.spinner(
                "Searching all PDF documents independently..."
            ):

                for pdf_name in indexed_pdf_files:

                    exact_source = source_values[pdf_name]

                    # Get retrieval count for this PDF.

                    retrieve_count = RETRIEVE_PER_PDF.get(
                        pdf_name,
                        DEFAULT_RETRIEVE_PER_PDF
                    )

                    docs = vector_store.similarity_search(
                        query=claim,
                        k=retrieve_count,
                        filter={"source": exact_source}
                    )

                    retrieved_by_pdf[pdf_name] = docs

            total_retrieved = sum(
                len(docs)
                for docs in retrieved_by_pdf.values()
            )

            if total_retrieved == 0:

                st.warning(
                    "No relevant evidence was retrieved from the PDFs."
                )

                st.stop()

            # ======================================
            # COMBINE RESULTS FROM ALL PDFS
            # ======================================

            combined_docs = []

            for pdf_name, docs in retrieved_by_pdf.items():

                combined_docs.extend(docs)

            # ======================================
            # CROSS-ENCODER RERANKING
            # ======================================

            with st.spinner(
                "Reranking combined evidence from all PDFs..."
            ):

                pairs = [
                    (claim, doc.page_content)
                    for doc in combined_docs
                ]

                scores = reranker.predict(pairs)

                ranked_results = sorted(
                    zip(scores, combined_docs),
                    key=lambda item: float(item[0]),
                    reverse=True
                )

            # ======================================
            # SELECT TOP 2 EVIDENCE FROM EACH PDF
            # ======================================

            top_results = []

            for pdf_name in indexed_pdf_files:

                pdf_results = [
                    (score, doc)
                    for score, doc in ranked_results
                    if normalize_filename(
                        doc.metadata.get("source", "")
                    ) == pdf_name
                ]

                top_results.extend(
                    pdf_results[:TOP_PER_PDF]
                )

            if not top_results:

                st.warning(
                    "No evidence could be selected after reranking."
                )

                st.stop()

            # ======================================
            # PREPARE GEMINI CONTEXT
            # ======================================

            context_parts = []

            for score, doc in top_results:

                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                page_number = (
                    page + 1 if isinstance(page, int) else page
                )

                context_parts.append(
                    f"""
Source Document: {source}
Page: {page_number}
Cross-Encoder Relevance Score: {float(score):.4f}

Evidence:
{doc.page_content}
"""
                )

            context = "\n\n".join(context_parts)

            # ======================================
            # 12. RETRIEVED EVIDENCE
            # ======================================

            st.divider()

            st.markdown(
                '<div class="section-heading">📚 Retrieved Evidence</div>',
                unsafe_allow_html=True
            )

            st.success(
                "Evidence retrieval and reranking completed."
            )

            st.caption(
                f"PDF documents searched: {len(indexed_pdf_files)} | "
                f"Retrieved chunks: {total_retrieved} | "
                f"Selected evidence chunks: {len(top_results)}"
            )

            # Show retrieval count for each PDF.

            st.subheader("Document-wise Retrieval")

            metric_columns = st.columns(len(indexed_pdf_files))

            for index, pdf_name in enumerate(indexed_pdf_files):

                metric_columns[index].metric(
                    pdf_name,
                    f"{len(retrieved_by_pdf[pdf_name])} chunks"
                )

            # Show evidence selected from each PDF.

            st.subheader("Evidence Selected from Each PDF")

            for pdf_name in indexed_pdf_files:

                pdf_results = [
                    (score, doc)
                    for score, doc in top_results
                    if normalize_filename(
                        doc.metadata.get("source", "")
                    ) == pdf_name
                ]

                with st.expander(
                    f"📄 {pdf_name} — {len(pdf_results)} selected chunks",
                    expanded=True
                ):

                    if not pdf_results:

                        st.info(
                            "No evidence was selected from this document."
                        )

                    for index, (score, doc) in enumerate(
                        pdf_results,
                        start=1
                    ):

                        page = doc.metadata.get(
                            "page",
                            "Unknown"
                        )

                        page_number = (
                            page + 1
                            if isinstance(page, int)
                            else page
                        )

                        st.markdown(
                            f"**Evidence {index} | Page {page_number}**"
                        )

                        st.caption(
                            "Cross-Encoder Relevance Score: "
                            f"{float(score):.4f}"
                        )

                        st.write(doc.page_content)

                        if index < len(pdf_results):
                            st.divider()

            # ======================================
            # 13. PIPELINE INFORMATION
            # ======================================

            st.divider()

            st.markdown(
                '<div class="section-heading">⚙️ Pipeline Information</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Retrieved Chunks",
                total_retrieved
            )

            col2.metric(
                "Selected Evidence",
                len(top_results)
            )

            col3.metric(
                "PDF Database Chunks",
                vector_store._collection.count()
            )

            st.caption(
                "Indexed PDF files: "
                + ", ".join(indexed_pdf_files)
            )

            st.caption(
                "The Cross-Encoder relevance score is not a "
                "confidence percentage or probability."
            )

            # ======================================
            # 14. GEMINI ANSWER GENERATION
            # ======================================

            st.divider()

            st.markdown(
                '<div class="section-heading">📋 AI Fact-Checking Report</div>',
                unsafe_allow_html=True
            )

            try:

                with st.spinner(
                    "Generating fact-checking report using Gemini..."
                ):

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
                        "Supported": (
                            "supported",
                            "🟢 Supported"
                        ),

                        "Contradicted": (
                            "contradicted",
                            "🔴 Contradicted"
                        ),

                        "Partially Supported": (
                            "partial",
                            "🟡 Partially Supported"
                        ),

                        "Insufficient Evidence": (
                            "insufficient",
                            "⚪ Insufficient Evidence"
                        )
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

                st.subheader("Complete AI Report")

                st.text_area(
                    label="Fact-Checking Analysis",
                    value=response,
                    height=350,
                    disabled=True,
                    key="final_ai_report"
                )

                st.success(
                    "Fact-checking analysis completed successfully."
                )

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
                        "Retrieval and reranking completed. "
                        "Evidence from the indexed PDFs is displayed above."
                    )

                    st.markdown(
                        "[Check Gemini API usage and limits]"
                        "(https://ai.dev/rate-limit)"
                    )

                elif "GOOGLE_API_KEY" in error_message:

                    st.error(
                        "Google API key is missing. Add GOOGLE_API_KEY "
                        "in Streamlit Cloud Secrets."
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
# 15. FOOTER
# ==================================================

st.divider()

st.markdown("""
<div class="footer">
    🔎 TruthLens AI | Multi-Document Intelligence & Fact-Checking
    <br>
    Built with Python, LangChain, ChromaDB, Gemini and Streamlit
</div>
""", unsafe_allow_html=True)

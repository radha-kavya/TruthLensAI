
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")


# --------------------------------------------------
# 2. LOAD PDF DOCUMENTS
# --------------------------------------------------

documents_path = BASE_DIR / "documents"

pdf_files = list(documents_path.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found.")

documents = []

print("\nLoading PDF documents...")

for pdf_file in pdf_files:

    print("Loading:", pdf_file.name)

    loader = PyPDFLoader(str(pdf_file))

    pages = loader.load()

    for page in pages:
        page.metadata["source"] = pdf_file.name
        page.metadata["document_type"] = "Research Report"

    documents.extend(pages)

print("Total PDFs:", len(pdf_files))
print("Total pages:", len(documents))


# --------------------------------------------------
# 3. SPLIT DOCUMENTS
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# --------------------------------------------------
# 4. LOAD HUGGING FACE EMBEDDINGS
# --------------------------------------------------

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding model loaded!")


# --------------------------------------------------
# 5. CREATE CHROMADB
# --------------------------------------------------

print("\nCreating ChromaDB...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="truthlens_v4",
    persist_directory=str(BASE_DIR / "chroma_db")
)

print("ChromaDB created!")


# --------------------------------------------------
# 6. RETRIEVE TOP 20 CHUNKS
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 20}
)

# --------------------------------------------------
# 7. LOAD CROSS-ENCODER RERANKING MODEL
# --------------------------------------------------

print("\nLoading Cross-Encoder model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross-Encoder loaded successfully!")


# --------------------------------------------------
# 8. GET USER CLAIM
# --------------------------------------------------

claim = input("\nEnter a claim to analyze: ").strip()

if not claim:
    raise ValueError("Please enter a valid claim.")


# --------------------------------------------------
# 9. RETRIEVE RELEVANT CHUNKS
# --------------------------------------------------

print("\nRetrieving top 20 chunks...")

retrieved_docs = retriever.invoke(claim)

if not retrieved_docs:
    print("No relevant documents found.")
    raise SystemExit

print("Retrieved chunks:", len(retrieved_docs))


# --------------------------------------------------
# 10. RERANK RETRIEVED CHUNKS
# --------------------------------------------------

print("\nReranking documents...")

pairs = [
    (claim, doc.page_content)
    for doc in retrieved_docs
]

scores = reranker.predict(pairs)

ranked_results = sorted(
    zip(scores, retrieved_docs),
    key=lambda item: item[0],
    reverse=True
)

# Select best 5 chunks
top_docs = [
    doc for score, doc in ranked_results[:5]
]

print("Reranking completed!")
print("Selected top chunks:", len(top_docs))


# --------------------------------------------------
# 11. PREPARE EVIDENCE CONTEXT
# --------------------------------------------------

context_parts = []

for doc in top_docs:

    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "Unknown")
    document_type = doc.metadata.get(
        "document_type", "Unknown"
    )

    page_number = (
        page + 1 if isinstance(page, int) else page
    )

    context_parts.append(f"""
Source: {source}
Page: {page_number}
Document Type: {document_type}

Evidence:
{doc.page_content}
""")

context = "\n\n".join(context_parts)


# --------------------------------------------------
# 12. INITIALIZE GEMINI LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0
)


# --------------------------------------------------
# 13. CREATE PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are TruthLens AI, a document-based fact-checking assistant.

Analyze the user's claim using only the evidence below.

CLAIM:
{claim}

RERANKED EVIDENCE:
{context}

Instructions:

1. Choose one verdict:
   - Supported
   - Contradicted
   - Partially Supported
   - Insufficient Evidence

2. Explain your reasoning.
3. Mention source documents and page numbers.
4. Do not invent information.
5. Identify evidence limitations.
6. Distinguish direct emissions from lifecycle emissions
   when relevant.

Report format:

Claim:
Verdict:
Evidence and Explanation:
Source Documents:
Limitations:
""")


# --------------------------------------------------
# 14. CREATE LANGCHAIN CHAIN
# --------------------------------------------------

chain = prompt | llm | StrOutputParser()


# --------------------------------------------------
# 15. GENERATE FINAL REPORT
# --------------------------------------------------

print("\nGenerating final report...")

response = chain.invoke({
    "claim": claim,
    "context": context
})


# --------------------------------------------------
# 16. DISPLAY FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 65)
print("          TRUTHLENS AI - V4 RERANKING RAG")
print("=" * 65)

print(response)


# --------------------------------------------------
# 17. DISPLAY RERANKING RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("              RERANKING RESULTS")
print("=" * 65)

print("PDF files:", len(pdf_files))
print("Pages:", len(documents))
print("Chunks:", len(chunks))
print("Initially retrieved:", len(retrieved_docs))
print("After reranking:", len(top_docs))

for index, (score, doc) in enumerate(
    ranked_results[:5], start=1
):

    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "Unknown")

    page_number = (
        page + 1 if isinstance(page, int) else page
    )

    print(f"\nRank: {index}")
    print(f"Relevance score: {float(score):.4f}")
    print("Source:", source)
    print("Page:", page_number)

print("\nV4 Reranking analysis completed!")

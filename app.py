"""
High-Accuracy HackRx LLM-Powered Query Retrieval System (Memory Optimized)
A robust, generalized RAG system designed for maximum accuracy on low-memory environments.
Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import io
import time
import requests
import asyncio
from typing import List, Dict, Any
from urllib.parse import urlparse
import numpy as np
import re

# FastAPI and HTTP
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

# Document processing
import PyPDF2
from docx import Document

# AI/ML
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

# Load environment variables from a .env file
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
#                                 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_TOKEN = os.getenv("HACKRX_API_KEY")

# Validate that the necessary API keys are set on startup
if not GEMINI_API_KEY:
    raise ValueError("FATAL ERROR: GEMINI_API_KEY environment variable is not set.")
if not HACKRX_TOKEN:
    raise ValueError("FATAL ERROR: HACKRX_API_KEY environment variable is not set.")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
LLM_MODEL = genai.GenerativeModel('gemini-1.5-flash')
GENERATION_CONFIG = genai.types.GenerationConfig(
    candidate_count=1,
    max_output_tokens=300,
    temperature=0.0,
)

# --- MEMORY OPTIMIZATION: Lazy Loading for Embedding Model ---
# We define a placeholder for the model and a function to load it on first use.
# This prevents loading the large model into memory when the app starts.
EMBEDDING_MODEL = None
EMBEDDING_DIM = 384

def get_embedding_model():
    """
    Lazily loads and caches the SentenceTransformer model to save memory on startup.
    """
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("🧠 Lazily loading embedding model (this will happen only once)...")
        # You can choose a smaller model here if memory is still an issue
        # e.g., 'paraphrase-MiniLM-L3-v2'
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded into memory.")
    return EMBEDDING_MODEL

# FAISS Index and Document Store
FAISS_INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)
DOCUMENT_STORE = []

# FastAPI App
app = FastAPI(
    title="High-Accuracy HackRx LLM Query Retrieval System",
    description="A generalized RAG system optimized for low-memory deployment.",
    version="3.1.0"
)
security = HTTPBearer()

# ═══════════════════════════════════════════════════════════════════════════════
#                                 DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]

class ClauseMatch(BaseModel):
    content: str
    similarity_score: float

# ═══════════════════════════════════════════════════════════════════════════════
#                           DOCUMENT PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 256
    
    def download_document(self, url: str) -> bytes:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            print(f"✅ Downloaded document: {len(response.content)} bytes")
            return response.content
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download document from URL: {str(e)}")
    
    def get_file_type(self, url: str) -> str:
        parsed_url = urlparse(url)
        file_path = parsed_url.path.lower()
        if file_path.endswith('.pdf'): return 'pdf'
        if file_path.endswith('.docx'): return 'docx'
        return 'pdf'
    
    def extract_text(self, content: bytes, file_type: str) -> str:
        if file_type == 'pdf': return self._extract_text_from_pdf(content)
        if file_type == 'docx': return self._extract_text_from_docx(content)
        raise ValueError(f"Unsupported file type: {file_type}")

    def _extract_text_from_pdf(self, content: bytes) -> str:
        try:
            text = ""
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                if page_text := page.extract_text(): text += page_text + "\n"
            print(f"📄 Extracted {len(text)} characters from PDF.")
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_text_from_docx(self, content: bytes) -> str:
        try:
            doc = Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            print(f"📄 Extracted {len(text)} characters from DOCX.")
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    def sentence_aware_chunking(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks, current_chunk_sentences, current_chunk_length = [], [], 0
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_chunk_length + sentence_length <= self.chunk_size:
                current_chunk_sentences.append(sentence)
                current_chunk_length += sentence_length
            else:
                chunks.append(DocumentChunk(content=" ".join(current_chunk_sentences), metadata={"chunk_id": len(chunks), **metadata}))
                current_chunk_sentences, current_chunk_length = [sentence], sentence_length
        if current_chunk_sentences:
            chunks.append(DocumentChunk(content=" ".join(current_chunk_sentences), metadata={"chunk_id": len(chunks), **metadata}))
        print(f"📊 Created {len(chunks)} sentence-aware chunks.")
        return chunks

# ═══════════════════════════════════════════════════════════════════════════════
#                           SEMANTIC SEARCH SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticSearchService:
    def __init__(self):
        self.faiss_index = FAISS_INDEX
        self.document_store = DOCUMENT_STORE
    
    def embed_and_index(self, chunks: List[DocumentChunk]) -> None:
        if not chunks: return
        self.faiss_index.reset(); self.document_store.clear()
        model = get_embedding_model() # Load model on demand
        texts = [chunk.content for chunk in chunks]
        print(f"🧠 Generating embeddings for {len(texts)} chunks...")
        embeddings = model.encode(texts, normalize_embeddings=True)
        self.faiss_index.add(embeddings.astype('float32'))
        self.document_store.extend([{"content": c.content, "metadata": c.metadata} for c in chunks])
        print(f"✅ Embedded and indexed {self.faiss_index.ntotal} chunks.")
    
    async def search(self, query: str, top_k: int = 8) -> List[ClauseMatch]:
        if self.faiss_index.ntotal == 0: return []
        model = get_embedding_model() # Load model on demand
        query_embedding = model.encode([query], normalize_embeddings=True)
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), min(top_k, self.faiss_index.ntotal))
        return [ClauseMatch(content=self.document_store[idx]["content"], similarity_score=float(dist)) for dist, idx in zip(distances[0], indices[0]) if idx != -1]

# ═══════════════════════════════════════════════════════════════════════════════
#                               LLM SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class LLMService:
    def __init__(self):
        self.model = LLM_MODEL
        self.generation_config = GENERATION_CONFIG
    
    def create_high_accuracy_prompt(self, query: str, relevant_chunks: List[ClauseMatch]) -> str:
        context = "\n\n---\n\n".join([chunk.content for chunk in relevant_chunks])
        return f"""**Role:** You are an expert document analyst. Your task is to answer a question with extreme precision based *only* on the provided text.
**Source Text:**\n---\n{context}\n---\n
**Instructions:**
1. Analyze the entire source text.
2. Synthesize all facts, figures, and conditions related to the user's question into a single, comprehensive sentence.
3. Your answer **MUST** be a single sentence.
4. Do **NOT** add any information not explicitly stated in the source text.
5. Answer directly. Do not start with phrases like "According to the document...".
**User Question:** {query}
**Single-Sentence Answer:**"""
    
    async def generate_answer(self, query: str, relevant_chunks: List[ClauseMatch]) -> str:
        if not relevant_chunks: return "The provided document does not contain information relevant to this question."
        prompt = self.create_high_accuracy_prompt(query, relevant_chunks)
        for attempt in range(3):
            try:
                response = await self.model.generate_content_async(prompt, generation_config=self.generation_config)
                return ' '.join(response.text.strip().replace('\n', ' ').split())
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                    print(f"Rate limit hit. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    return f"An error occurred while generating the answer."
        return "Failed to generate an answer after multiple retries."
    
    async def process_all_queries(self, queries: List[str], search_service: SemanticSearchService) -> List[str]:
        async def _process_one(query: str):
            relevant_chunks = await search_service.search(query, top_k=8)
            return await self.generate_answer(query, relevant_chunks)
        return await asyncio.gather(*[_process_one(q) for q in queries])

# ═══════════════════════════════════════════════════════════════════════════════
#                                 API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

doc_processor = DocumentProcessor()
search_service = SemanticSearchService()
llm_service = LLMService()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme.lower() != "bearer" or credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_submission(request: QueryRequest, _: str = Depends(verify_token)):
    start_time = time.time()
    try:
        print(f"🎯 Starting RAG process for {request.documents}")
        content = doc_processor.download_document(request.documents)
        file_type = doc_processor.get_file_type(request.documents)
        text = doc_processor.extract_text(content, file_type)
        chunks = doc_processor.sentence_aware_chunking(text, {"source": request.documents})
        search_service.embed_and_index(chunks) # This will trigger the one-time model load
        answers = await llm_service.process_all_queries(request.questions, search_service)
        print(f"✅ Process completed in {time.time() - start_time:.2f}s.")
        return QueryResponse(answers=answers)
    except Exception as e:
        print(f"❌ Critical error during RAG process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An internal processing error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "3.1.0", "model": "gemini-1.5-flash"}

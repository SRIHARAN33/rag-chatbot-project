"""
RAG Pipeline - Core implementation
Handles: data loading, preprocessing, embedding, vector store, retrieval, generation
"""

import os
import json
import re
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# ─────────────────────────────────────────────────────────────
# Sample knowledge base for demo purposes
# ─────────────────────────────────────────────────────────────
SAMPLE_KB = [
    {"question": "How do I return a product?", "answer": "You can return any product within 30 days of purchase. Visit our Returns Portal at returns.example.com, enter your order number, and follow the prompts. We offer free return shipping for defective items.", "category": "Returns"},
    {"question": "What is the refund policy?", "answer": "Refunds are processed within 5-7 business days after we receive the returned item. The refund will be credited to your original payment method. For digital products, refunds are available within 7 days if not downloaded.", "category": "Refunds"},
    {"question": "How do I track my order?", "answer": "You can track your order by logging into your account and visiting Order History. You will also receive a tracking email once your order ships. Alternatively, use the tracking number on our Track Order page.", "category": "Orders"},
    {"question": "How long does shipping take?", "answer": "Standard shipping takes 5-7 business days. Express shipping (2-3 days) and Next-Day delivery are also available at checkout. International orders may take 10-21 business days.", "category": "Shipping"},
    {"question": "What payment methods do you accept?", "answer": "We accept Visa, Mastercard, American Express, PayPal, Apple Pay, Google Pay, and UPI. All transactions are secured with 256-bit SSL encryption.", "category": "Payments"},
    {"question": "How do I cancel my order?", "answer": "Orders can be cancelled within 1 hour of placement. Go to Order History, select the order, and click Cancel. After 1 hour, the order may have been processed and you will need to request a return.", "category": "Orders"},
    {"question": "How do I reset my password?", "answer": "Click 'Forgot Password' on the login page, enter your email address, and you will receive a reset link within a few minutes. The link expires in 24 hours. Check your spam folder if you don't see the email.", "category": "Account"},
    {"question": "How do I update my account details?", "answer": "Log in to your account, click on My Profile, and update your name, email, phone number, or address. You will receive a confirmation email for any sensitive changes.", "category": "Account"},
    {"question": "Is my payment information secure?", "answer": "Yes. We use PCI-DSS compliant payment processing. We never store your full card number on our servers. All data is encrypted in transit and at rest.", "category": "Security"},
    {"question": "What do I do if I receive a damaged item?", "answer": "Please take photos of the damage and contact our support team within 48 hours via the Help Center or email support@example.com. We will arrange a free replacement or full refund.", "category": "Returns"},
    {"question": "How do I apply a promo code?", "answer": "Enter your promo code in the Discount Code field at checkout and click Apply. The discount will be reflected in your order total. Promo codes cannot be combined unless specified.", "category": "Offers"},
    {"question": "Can I change my delivery address after placing an order?", "answer": "Address changes are possible within 30 minutes of placing the order. Contact live chat immediately with your order number and new address. After 30 minutes, the order may have been dispatched.", "category": "Orders"},
    {"question": "How do I contact customer support?", "answer": "You can reach us via Live Chat (available 24/7), email at support@example.com (response within 2 hours), or call 1-800-555-0100 (Mon–Sat, 9 AM–6 PM). Our Help Center has answers to common questions.", "category": "Support"},
    {"question": "Do you offer a warranty?", "answer": "All electronics and appliances come with a 1-year manufacturer warranty. Extended warranty plans of 2 and 3 years are available at checkout. Clothing and accessories have a 90-day quality guarantee.", "category": "Warranty"},
    {"question": "How do I delete my account?", "answer": "To delete your account, go to Settings → Privacy → Delete Account. This is permanent and cannot be undone. Your data will be removed within 30 days. Any pending orders must be fulfilled first.", "category": "Account"},
    {"question": "What is your price match policy?", "answer": "We match prices from any major authorized retailer. Submit a price match request with a link to the competitor's listing within 7 days of purchase. The price difference will be refunded to your account.", "category": "Pricing"},
    {"question": "How do I sign up for the loyalty program?", "answer": "Create an account on our website and you are automatically enrolled in the rewards program. Earn 1 point per $1 spent. Redeem 100 points for $1 off. Elite status is achieved at 1000 points/year.", "category": "Loyalty"},
    {"question": "Can I order without creating an account?", "answer": "Yes, guest checkout is available. However, creating an account lets you track orders, save addresses, earn rewards, and request returns more easily.", "category": "Orders"},
    {"question": "What is the exchange policy?", "answer": "You can exchange items within 30 days of purchase for a different size, color, or product of equal or lesser value. Free exchanges on the first request per order. Visit the Returns Portal to initiate.", "category": "Returns"},
    {"question": "Do you ship internationally?", "answer": "Yes, we ship to over 50 countries. International shipping rates and delivery times are shown at checkout. Import duties and taxes are the customer's responsibility and vary by country.", "category": "Shipping"},
]


# ─────────────────────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────────────────────
class TextPreprocessor:
    """Cleans and preprocesses text for embedding."""

    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', text)          # Remove HTML
        text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'\s+', ' ', text)              # Normalize whitespace
        text = text.strip()
        return text

    @staticmethod
    def chunk(text: str, chunk_size: int = 200, overlap: int = 30) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(' '.join(words[start:end]))
            start += chunk_size - overlap
        return chunks


# ─────────────────────────────────────────────────────────────
# Vector Store (FAISS)
# ─────────────────────────────────────────────────────────────
class FAISSVectorStore:
    """Manages FAISS index for fast similarity search."""

    def __init__(self):
        self.index = None
        self.chunks: List[str] = []
        self.dimension: Optional[int] = None

    def build(self, embeddings: np.ndarray, chunks: List[str]):
        self.dimension = embeddings.shape[1]
        self.chunks = chunks
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"[VectorStore] Built index with {self.index.ntotal} vectors (dim={self.dimension})")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> Tuple[List[str], List[float]]:
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        results = [(self.chunks[i], float(scores[0][j]))
                   for j, i in enumerate(indices[0]) if i < len(self.chunks)]
        return [r[0] for r in results], [r[1] for r in results]


# ─────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────
class RAGPipeline:
    """
    Full RAG pipeline:
    load KB → preprocess → embed → FAISS index → retrieve → generate
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        chunk_size: int = 200,
        top_k: int = 3,
    ):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.preprocessor = TextPreprocessor()
        self.vector_store = FAISSVectorStore()
        self.raw_documents: List[str] = []
        self.all_chunks: List[str] = []

        # Embedding model
        print("[RAG] Loading embedding model...")
        self.embedder = SentenceTransformer(self.EMBEDDING_MODEL)

        # Gemini
        self.gemini_model = None
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                print("[RAG] Gemini LLM configured.")
            except Exception as e:
                print(f"[RAG] Gemini setup failed: {e}. Falling back to rule-based answers.")
        else:
            print("[RAG] No Gemini API key — using rule-based fallback.")

    # ── Data loading ──────────────────────────────────────────

    def load_sample_knowledge_base(self):
        """Load built-in sample FAQ data."""
        docs = []
        for item in SAMPLE_KB:
            text = f"Q: {item['question']}\nA: {item['answer']}\nCategory: {item['category']}"
            docs.append(self.preprocessor.clean(text))
        self.raw_documents = docs
        print(f"[RAG] Loaded {len(docs)} sample documents.")

    def load_knowledge_base_from_file(self, uploaded_file):
        """Load KB from a user-uploaded file (CSV / TXT / JSON)."""
        name = uploaded_file.name.lower()
        docs = []

        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            for _, row in df.iterrows():
                parts = []
                for col in ["question", "answer", "text", "content", "description"]:
                    if col in df.columns and pd.notna(row.get(col, "")):
                        parts.append(str(row[col]))
                if parts:
                    docs.append(self.preprocessor.clean(" ".join(parts)))

        elif name.endswith(".json"):
            data = json.load(uploaded_file)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = " ".join(str(v) for v in item.values())
                        docs.append(self.preprocessor.clean(text))
                    else:
                        docs.append(self.preprocessor.clean(str(item)))
            else:
                docs.append(self.preprocessor.clean(str(data)))

        elif name.endswith(".txt"):
            raw = uploaded_file.read().decode("utf-8")
            paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 20]
            docs = [self.preprocessor.clean(p) for p in paragraphs]

        if not docs:
            raise ValueError("Could not extract any text from the uploaded file.")

        self.raw_documents = docs
        print(f"[RAG] Loaded {len(docs)} documents from file.")

    # ── Vector store build ────────────────────────────────────

    def build_vector_store(self):
        """Chunk all documents, embed, and build FAISS index."""
        if not self.raw_documents:
            raise ValueError("No documents loaded. Call load_*() first.")

        for doc in self.raw_documents:
            self.all_chunks.extend(self.preprocessor.chunk(doc, self.chunk_size))

        print(f"[RAG] Embedding {len(self.all_chunks)} chunks...")
        embeddings = self.embedder.encode(self.all_chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        self.vector_store.build(embeddings, self.all_chunks)

    # ── Query ─────────────────────────────────────────────────

    def query(self, user_question: str) -> Dict:
        """Full RAG query: embed → retrieve → generate."""
        cleaned_q = self.preprocessor.clean(user_question)

        # Embed query
        q_emb = self.embedder.encode([cleaned_q])
        q_emb = np.array(q_emb, dtype="float32")

        # Retrieve
        sources, scores = self.vector_store.search(q_emb, self.top_k)

        # Generate
        if self.gemini_model:
            answer = self._generate_with_gemini(cleaned_q, sources)
        else:
            answer = self._generate_fallback(cleaned_q, sources)

        return {"answer": answer, "sources": sources, "scores": scores}

    # ── Generation ────────────────────────────────────────────

    def _generate_with_gemini(self, question: str, context_chunks: List[str]) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        prompt = f"""You are a helpful and professional customer service assistant.
Use ONLY the information provided in the context below to answer the customer's question.
If the context doesn't contain relevant information, say so politely and suggest contacting support.
Be concise, friendly, and helpful.

Context:
{context}

Customer Question: {question}

Answer:"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[RAG] Gemini error: {e}")
            return self._generate_fallback(question, context_chunks)

    def _generate_fallback(self, question: str, context_chunks: List[str]) -> str:
        """Rule-based answer using best matching chunk (no LLM needed)."""
        if not context_chunks:
            return ("I'm sorry, I couldn't find relevant information for your question. "
                    "Please contact our support team at support@example.com or call 1-800-555-0100.")

        best = context_chunks[0]
        # Extract answer portion if it exists
        if "A:" in best:
            answer_part = best.split("A:")[-1].strip()
            # Remove "Category:" trailing text
            if "Category:" in answer_part:
                answer_part = answer_part.split("Category:")[0].strip()
            return answer_part

        # Return cleaned context as answer
        return best[:500] + ("..." if len(best) > 500 else "")

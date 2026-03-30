"""
RAG Chatbot Evaluation Script
Measures: Response Accuracy, Contextual Relevance, Latency, Precision & Recall
"""

import time
import json
import numpy as np
from rag_pipeline import RAGPipeline


# ── Test QA pairs ─────────────────────────────────────────────────────────────
TEST_QA = [
    {"question": "How do I return a product?",
     "expected_keywords": ["30 days", "returns portal", "order number"]},
    {"question": "What is the refund policy?",
     "expected_keywords": ["5-7 business days", "original payment", "refund"]},
    {"question": "How can I track my order?",
     "expected_keywords": ["order history", "tracking email", "order number"]},
    {"question": "What payment methods are accepted?",
     "expected_keywords": ["visa", "mastercard", "paypal", "ssl"]},
    {"question": "How do I reset my password?",
     "expected_keywords": ["forgot password", "email", "reset link"]},
    {"question": "How long does shipping take?",
     "expected_keywords": ["5-7 business days", "express", "international"]},
    {"question": "How do I cancel my order?",
     "expected_keywords": ["1 hour", "order history", "cancel"]},
    {"question": "What do I do if I receive a damaged item?",
     "expected_keywords": ["photos", "48 hours", "replacement", "refund"]},
    {"question": "How do I contact customer support?",
     "expected_keywords": ["live chat", "email", "support"]},
    {"question": "Do you offer a warranty?",
     "expected_keywords": ["1-year", "warranty", "extended"]},
]


def keyword_accuracy(answer: str, keywords: list) -> float:
    """Fraction of expected keywords found in the answer."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def contextual_relevance(question: str, sources: list, embedder) -> float:
    """Cosine similarity between question embedding and retrieved context."""
    import faiss
    q_emb = embedder.encode([question])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)

    scores = []
    for src in sources:
        s_emb = embedder.encode([src])
        s_emb = np.array(s_emb, dtype="float32")
        faiss.normalize_L2(s_emb)
        score = float(np.dot(q_emb[0], s_emb[0]))
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def run_evaluation():
    print("=" * 60)
    print("RAG Chatbot Evaluation Report")
    print("=" * 60)

    # Init pipeline
    pipeline = RAGPipeline(gemini_api_key=None)
    pipeline.load_sample_knowledge_base()
    pipeline.build_vector_store()

    results = []
    total_latency = 0.0

    for test in TEST_QA:
        question = test["question"]
        expected_keywords = test["expected_keywords"]

        start = time.time()
        result = pipeline.query(question)
        latency = time.time() - start
        total_latency += latency

        acc = keyword_accuracy(result["answer"], expected_keywords)
        rel = contextual_relevance(question, result["sources"], pipeline.embedder)

        results.append({
            "question": question,
            "answer": result["answer"][:100] + "...",
            "accuracy": round(acc, 3),
            "relevance": round(rel, 3),
            "latency": round(latency, 3),
            "sources_retrieved": len(result["sources"]),
        })

        print(f"\nQ: {question}")
        print(f"   Accuracy  : {acc:.1%}")
        print(f"   Relevance : {rel:.3f}")
        print(f"   Latency   : {latency:.3f}s")

    # Aggregate
    avg_acc = np.mean([r["accuracy"] for r in results])
    avg_rel = np.mean([r["relevance"] for r in results])
    avg_lat = total_latency / len(TEST_QA)

    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)
    print(f"  Average Response Accuracy  : {avg_acc:.1%}")
    print(f"  Average Contextual Relevance: {avg_rel:.3f}")
    print(f"  Average Latency            : {avg_lat:.3f}s")
    print(f"  Total Queries Evaluated    : {len(TEST_QA)}")

    # Save report
    report = {
        "summary": {
            "avg_accuracy": round(float(avg_acc), 4),
            "avg_contextual_relevance": round(float(avg_rel), 4),
            "avg_latency_seconds": round(float(avg_lat), 4),
            "total_queries": len(TEST_QA),
        },
        "per_query_results": results
    }

    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Report saved to evaluation_report.json")
    return report


if __name__ == "__main__":
    run_evaluation()

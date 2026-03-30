"""
RAG-Based Customer Service Chatbot
GUVI | HCL Project - Gen AI
"""

import streamlit as st
from rag_pipeline import RAGPipeline
import time

st.set_page_config(
    page_title="RAG Customer Service Bot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f3e5f5;
        margin-right: 20%;
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG-Based Customer Service Chatbot</h1>
    <p>Powered by Google Gemini LLM + FAISS Vector Store</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {"total_queries": 0, "avg_latency": 0.0, "satisfaction": []}

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    gemini_key = st.text_input("Google Gemini API Key", type="password", 
                                help="Get from https://makersuite.google.com/app/apikey")
    
    st.markdown("---")
    st.header("📚 Knowledge Base")
    
    uploaded_file = st.file_uploader(
        "Upload Knowledge Base (CSV/TXT/JSON)",
        type=["csv", "txt", "json"],
        help="Upload your FAQ or support documents"
    )
    
    use_sample = st.checkbox("Use Sample KB (Demo Mode)", value=True)
    
    chunk_size = st.slider("Chunk Size", 100, 500, 200, 50)
    top_k = st.slider("Top-K Results", 1, 5, 3)
    
    if st.button("🚀 Initialize Chatbot", type="primary"):
        with st.spinner("Building vector store..."):
            try:
                pipeline = RAGPipeline(
                    gemini_api_key=gemini_key if gemini_key else None,
                    chunk_size=chunk_size,
                    top_k=top_k
                )
                
                if uploaded_file:
                    pipeline.load_knowledge_base_from_file(uploaded_file)
                elif use_sample:
                    pipeline.load_sample_knowledge_base()
                else:
                    st.error("Please upload a file or enable Demo Mode.")
                    st.stop()
                
                pipeline.build_vector_store()
                st.session_state.rag_pipeline = pipeline
                st.success("✅ Chatbot ready!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Metrics
    st.markdown("---")
    st.header("📊 Session Metrics")
    metrics = st.session_state.metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", metrics["total_queries"])
    with col2:
        st.metric("Avg Latency", f"{metrics['avg_latency']:.2f}s")
    
    if metrics["satisfaction"]:
        avg_sat = sum(metrics["satisfaction"]) / len(metrics["satisfaction"])
        st.metric("Avg Satisfaction", f"{avg_sat:.1f}/5")

# Main chat area
col1, col2 = st.columns([3, 1])

with col1:
    # Chat history
    st.subheader("💬 Chat")
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <b>👤 You:</b> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <b>🤖 Bot:</b> {msg["content"]}
                    <br><small>⏱ {msg.get('latency', 0):.2f}s | Docs retrieved: {msg.get('docs_used', 0)}</small>
                </div>
                """, unsafe_allow_html=True)
                
                if msg.get("sources"):
                    with st.expander("📄 Retrieved Context"):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(f"**Chunk {i}:** {src[:300]}...")
    
    # Input
    if st.session_state.rag_pipeline:
        user_input = st.chat_input("Ask anything about our products or services...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Thinking..."):
                start = time.time()
                result = st.session_state.rag_pipeline.query(user_input)
                latency = time.time() - start
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "docs_used": len(result["sources"]),
                "latency": latency
            })
            
            # Update metrics
            metrics = st.session_state.metrics
            metrics["total_queries"] += 1
            total_lat = metrics["avg_latency"] * (metrics["total_queries"] - 1) + latency
            metrics["avg_latency"] = total_lat / metrics["total_queries"]
            
            st.rerun()
    else:
        st.info("👈 Configure and initialize the chatbot from the sidebar to start chatting.")

with col2:
    st.subheader("⭐ Rate Response")
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        rating = st.slider("Satisfaction", 1, 5, 3, key="rating_slider")
        if st.button("Submit Rating"):
            st.session_state.metrics["satisfaction"].append(rating)
            st.success("Thanks for the feedback!")
    
    st.markdown("---")
    st.subheader("💡 Sample Questions")
    sample_questions = [
        "How do I return a product?",
        "What is the refund policy?",
        "How to track my order?",
        "How do I reset my password?",
        "What payment methods are accepted?",
        "How to contact support?"
    ]
    for q in sample_questions:
        if st.button(q, key=f"btn_{q[:20]}"):
            if st.session_state.rag_pipeline:
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    start = time.time()
                    result = st.session_state.rag_pipeline.query(q)
                    latency = time.time() - start
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "docs_used": len(result["sources"]),
                    "latency": latency
                })
                st.rerun()

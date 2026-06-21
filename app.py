import os
import numpy as np
import streamlit as st
import chromadb
import plotly.express as px
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

st.set_page_config(page_title="Steps AI Support Copilot", page_icon="🤖", layout="wide")
st.title("🤖 Steps AI Support Copilot")
st.caption("Production-Grade Knowledge Assistant with Active Tensor Visualization")

@st.cache_resource
def initialize_rag_pipeline():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("steps_ai_knowledge_base")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    system_prompt = (
        "You are a strict, highly technical Senior Solutions Engineer at Steps AI. "
        "Your primary directive is to NEVER hallucinate. "
        "You are strictly restricted to answering questions using ONLY the provided context. "
        "1. CAREERS & JOBS: If a user asks about job openings, you must strictly reference the 'Careers' section of the company overview. The only human roles currently open are 'AI-first operators across marketing and sales'. "
        "CRITICAL: Do NOT confuse the AI Agent products (e.g., 'Support Agent', 'Sales Assistant', 'AI Copilot') with human job openings. "
        "If a user asks about engineering roles, reply exactly with: 'According to our public directory, we are currently only hiring operators across marketing and sales, but we are always open to connecting with technical talent.' "
        "2. DO NOT invent email addresses. "
        "3. If the answer cannot be confidently derived from the retrieved documentation, you must stop generating and reply EXACTLY with: "
        "'I cannot find that specific detail in our public knowledge base. Let me loop in a human support agent to assist you further.' "
        "Keep your answers extremely concise, professional, and formatted in Markdown."
    )
    
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        system_prompt=system_prompt,
        verbose=True
    )
    return chat_engine, embed_model

try:
    chat_engine, embed_model = initialize_rag_pipeline()
except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# --- THE PYTORCH FLEX: Mathematical Salience Visualizer ---
def render_attention_matrix(query, source_nodes):
    if not source_nodes:
        return
    
    # 1. Generate the Query Tensor (Q)
    query_embedding = np.array(embed_model.get_query_embedding(query))
    
    node_labels = []
    similarities = []
    
    # 2. Extract the Key Tensors (K) and compute dot product
    for i, node in enumerate(source_nodes):
        node_embedding = np.array(node.embedding) if node.embedding else np.array(embed_model.get_text_embedding(node.text))
        
        # Calculate Cosine Similarity (Q * K^T)
        sim = np.dot(query_embedding, node_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
        similarities.append(sim)
        
        # Format the label for the UI
        title = node.metadata.get('title', f'Chunk {i}')
        node_labels.append(f"{title[:25]}...")
        
    # 3. Apply Softmax to get normalized attention distribution
    sim_tensor = np.array(similarities)
    exp_sim = np.exp(sim_tensor * 10) # Temperature scaling for visual contrast
    attention_weights = exp_sim / np.sum(exp_sim)
    
    # 4. Render the Heatmap
    fig = px.bar(
        x=attention_weights, 
        y=node_labels, 
        orientation='h',
        title="Cross-Attention Weights (Query vs. Retrieved Keys)",
        labels={'x': 'Softmax Attention Score', 'y': 'Document Vectors'},
        color=attention_weights,
        color_continuous_scale='Inferno'
    )
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    
    with st.expander("📊 View Mathematical Retrieval Analytics", expanded=True):
        st.markdown("**( $Attention(Q, K) = \\text{softmax}(QK^T)V$ )**")
        st.plotly_chart(fig, use_container_width=True)
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System Online. Ready for architectural and integration queries."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Query the Steps AI Knowledge Base..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Execute the query
        response = chat_engine.stream_chat(user_input)
        
        full_response = ""
        for token in response.response_gen:
            full_response += token
            response_placeholder.markdown(full_response + "▌")
            
        response_placeholder.markdown(full_response)
        
        # Inject the mathematical proof
        if hasattr(response, 'source_nodes'):
            render_attention_matrix(user_input, response.source_nodes)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
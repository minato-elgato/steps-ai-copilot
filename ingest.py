import os
import chromadb
import frontmatter
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def process_and_ingest():
    # --- NEW CODE: Override the OpenAI default with a free local model ---
    print("Initializing local HuggingFace embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # ---------------------------------------------------------------------

    # 1. Initialize the persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("steps_ai_knowledge_base")
    
    # 2. Bind ChromaDB to LlamaIndex's Storage Context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = []
    DATA_DIRECTORY = "./data"

    # 3. Parse files manually to handle the YAML perfectly
    for filename in os.listdir(DATA_DIRECTORY):
        if filename.endswith(".md"):
            filepath = os.path.join(DATA_DIRECTORY, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)

            # Map YAML directly into the LlamaIndex Document metadata
            doc = Document(
                text=post.content,
                metadata={
                    "title": post.get("title", filename),
                    "url": post.get("url", "https://stepsai.co")
                }
            )
            documents.append(doc)

    print(f"Parsed {len(documents)} source files. Splitting into nodes...")

    # 4. Use LlamaIndex's native Markdown parser to create Nodes
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    # 5. Embed and index the nodes directly into ChromaDB
    print("Calculating vector embeddings... (This might take ~30 seconds on the first run to download model weights)")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context
    )
    
    print(f"✅ Phase 1 Complete! Embedded {len(nodes)} Markdown nodes into ./chroma_db")

if __name__ == "__main__":
    process_and_ingest()
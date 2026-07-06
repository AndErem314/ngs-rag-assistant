#!/usr/bin/env python3
"""Re-ingest ChromaDB with bge-m3 embeddings."""
import chromadb, ollama, time, sys
from chromadb.config import Settings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
BGE_MODEL = "bge-m3:latest"
BATCH_SIZE = 20

print("Step 1: Reading existing ChromaDB documents...")
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
col = client.get_collection("ngs_docs")

data = col.get(include=["documents", "metadatas"])
ids = data["ids"]
documents = data["documents"]
metadatas = data["metadatas"]

print(f"  Found {len(ids)} chunks from {len(set(m['source'] for m in metadatas))} sources")

print("\nStep 2: Re-embedding with bge-m3:latest...")
ollama_client = ollama.Client(host="http://localhost:11434")

new_embeddings = []
total = len(documents)
for i in range(0, total, BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)...", end=" ", flush=True)
    
    try:
        response = ollama_client.embed(model=BGE_MODEL, input=batch)
        embeddings = response["embeddings"]
        new_embeddings.extend(embeddings)
        print(f"OK ({len(embeddings)} vectors)")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    time.sleep(0.05)

print(f"\n  Generated {len(new_embeddings)} embeddings")
print(f"  Vector dimension: {len(new_embeddings[0]) if new_embeddings else 'N/A'}")

print("\nStep 3: Clearing and re-populating ChromaDB...")
col.delete(ids=ids)
print(f"  Cleared {len(ids)} existing chunks")

col.add(ids=ids, embeddings=new_embeddings, documents=documents, metadatas=metadatas)
print(f"  Re-added {len(ids)} chunks with bge-m3 embeddings")

verify = col.get(include=["documents", "metadatas"])
print(f"\nStep 4: Verification - {len(verify['ids'])} chunks in collection")

# Test query
test_embed = ollama_client.embed(model=BGE_MODEL, input="Nextera XT bead ratio for small fragments")
results = col.query(query_embeddings=[test_embed["embeddings"][0]], n_results=3, include=["documents", "distances"])
print(f"\nStep 5: Query test")
for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
    snippet = doc[:120].replace('\n', ' ')
    print(f"  #{i+1} dist={dist:.4f}: {snippet}...")

print("\nDone.")
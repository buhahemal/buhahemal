# 1. Transcribe videos (Whisper)
import whisper
model = whisper.load_model("base")
result = model.transcribe("video.mp4")


# 2. Segment transcript
segments = result['segments']  # Each has 'start', 'end', 'text'


# 3. Generate embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
model_st = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model_st.encode([seg['text'] for seg in segments])


# 4. Index with FAISS
import faiss
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))


# 5. Query API (FastAPI)
from fastapi import FastAPI
app = FastAPI()


@app.get('/search')
def search(query: str):
    query_emb = model_st.encode([query])
    D, I = index.search(np.array(query_emb), k=1)
    idx = I[0][0]
    return {"video_id": "video.mp4", "start": segments[idx]['start'], "end": segments[idx]['end'], "text": segments[idx]['text']}

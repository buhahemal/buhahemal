# 1. Transcribe videos (Whisper)
import whisper
model = whisper.load_model("base")
result = model.transcribe("video.mp4")
# result['segments'] contains text and timestamps


# 2. Index with Whoosh
from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.index import create_in
from whoosh.qparser import QueryParser


schema = Schema(video_id=ID(stored=True), start=NUMERIC(stored=True), end=NUMERIC(stored=True), content=TEXT)
ix = create_in("indexdir", schema)
writer = ix.writer()
for seg in result['segments']:
    writer.add_document(video_id="video.mp4", start=seg['start'], end=seg['end'], content=seg['text'])
writer.commit()


# 3. Search API (FastAPI)
from fastapi import FastAPI
app = FastAPI()


@app.get('/search')
def search(query: str):
    with ix.searcher() as searcher:
        qp = QueryParser("content", schema=ix.schema)
        q = qp.parse(query)
        results = searcher.search(q, limit=1)
        if results:
            return dict(results[0])
        return {"error": "No match found"}

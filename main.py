from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from pypdf import PdfReader
from pydantic import BaseModel, Field
from typing import Annotated
from groq import Groq
from dotenv import load_dotenv
import chromadb
import os, psutil, time

load_dotenv()

app = FastAPI()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME")
model = SentenceTransformer("all-MiniLM-L6-v2")
dbclient = chromadb.PersistentClient(path = './chroma_store')


# create dir for files if needed
os.makedirs('uploads', exist_ok = True)

# defining the basemodels for requests and responses
class QueryRequest(BaseModel):
    user: str
    query: Annotated[str, Field(min_length=1)]
    doc_id: str | None = None

class QueryResponse(BaseModel):
    ans: str

@app.get('/health')
def check_status():
    return {"status": "ok"}

# helper functions
def route_query(query: str) -> str:
    model_resp = groq_client.chat.completions.create(
         model = MODEL_NAME,
        messages = [{'role': 'system', 'content':'you are a decision router. given a user query, decide if answering it requires retrieving information from uploaded documents. say YES if the query asks about specific documents, files, summaries, or domain-specific content. say NO if it is general knowledge. respond with only YES or NO.'},
        {'role':'user','content':query}]
)
    return model_resp.choices[0].message.content.strip().upper()


# retrieve_and_answer() function
def retrieve_and_answer(query: str, groq_client, doc_id: str | None = None, n_results: int = 3) -> str:
    """
    Creating the retrieval function to get the context about the files referenced.
    """
    collection = dbclient.get_or_create_collection("documents")
    q_embed = embed_chunks([query])[0]
    n_results = 5 # say top 5 results
    if doc_id:
        res= collection.query(query_embeddings = [q_embed], n_results = n_results, where = {'doc_id':doc_id}) 
    else:
        res= collection.query(query_embeddings = [q_embed], n_results = n_results) 
    print(res['metadatas'])
    context = " ".join(res['documents'][0])
    response = groq_client.chat.completions.create(
        model = MODEL_NAME, messages = [
            {'role':'system', 'content':f'you are a helpful assistant who will use this context to answer the queries : {context}'},
            {'role':'user','content':query}
        ]
    )
    return response.choices[0].message.content.strip()


def answer_directly(query: str) -> str:
    response = groq_client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {'role':'system', 'content': 'you are a helpful assistant who answers in the simplest way'},
            {'role':'user', 'content': query}
        ]
    )
    return response.choices[0].message.content


# query function
@app.post('/ask')
def ask_question(request: QueryRequest) -> QueryResponse:
    decision = route_query(request.query)
    print(f'Query:{request.query} --> decision: {decision}') # debugging
    if 'YES' in decision:
        answer = retrieve_and_answer(request.query, groq_client, request.doc_id)
    elif 'NO' in decision:
        answer = answer_directly(request.query)
    else:
        answer = answer_directly(request.query) # this is the default fallback

    return QueryResponse(ans=answer)


# uploading endpoints
@app.post('/upload')
async def upload_document(file: UploadFile):
    contents = await file.read()
    file_path = f'uploads/{file.filename}'
    with open(file_path, 'wb') as f:
        f.write(contents)
    text = parse_pdf(file_path)
    if not text.strip():
        raise HTTPException(status_code = 400, detail = "PDF is not read right or its empty")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    collection = store_in_chromadb(chunks, embeddings, doc_id = Path(file.filename).stem )
    return {'filename': file.filename, 'status': True, 'chunks': collection.count()}


"""
now we work on creating the file parsing and chunking
"""

def parse_pdf(file_path: str) -> str:
    """
    This function utilises the PdfReader library to parse the pages of the given pdf
    """
    if not file_path:
        return "Error parsing pdf"
    reader = PdfReader(file_path)
    pages = [page.extract_text() for page in reader.pages if page.extract_text() != None]
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Chunk the text with a simple method of taking consecutive text, 
    store all the chunks in a list
    """
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start: start + chunk_size]
        chunks.append(chunk)
        start = start + chunk_size - chunk_overlap
    return chunks

# function to develop embeddings - enter the **Sentence Transformer**
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Creating the embeddings of the chunks using MiniLM model.
    """
    embeddings = model.encode(chunks)
    return embeddings.tolist()

# storing inside the vector store
def store_in_chromadb(chunks: list[str], embeddings: list[list[str]], doc_id: str):
    """
    Stores the chunks and embeddings inside the collection we create
    """
    collection =  dbclient.get_or_create_collection('documents')
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadata = [
    {"doc_id": doc_id} for i in range(len(chunks))
    ]
    collection.upsert(documents = chunks, embeddings = embeddings, metadatas = metadata, ids = ids)
    return collection


'''

if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    start = time.time()
    mem_before = process.memory_info().rss / (1024**2)

    text = parse_pdf("/mnt/c/Users/DELL/Downloads/Abstract_v2.pdf")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    collection = store_in_chromadb(chunks, embeddings, "test_doc")
    print("stored successfully")
    print(collection.count())
    answer = retrieve_and_answer("summarize the document", client)
    print(answer)
    mem_after = process.memory_info().rss / (1024 **2)
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Memory delta: {mem_after - mem_before:.2f} MB")
'''

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from pydantic import BaseModel, Field
from typing import Annotated
from groq import Groq
from dotenv import load_dotenv
import chromadb
import os, psutil, time

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME")
model = SentenceTransformer("all-MiniLM-L6-v2")
dbclient = chromadb.PersistentClient(path = './chroma_store')


# create dir for files if needed
os.makedirs('uploads', exist_ok = True)

# defining the basemodels for requests and responses
class QueryRequest(BaseModel):
    user: str
    query: Annotated[str, Field(min_length=1)]

class QueryResponse(BaseModel):
    ans: str

@app.get('/health')
def check_status():
    return {"status": "ok"}

# helper functions
def route_query(query: str) -> str:
    model_resp = client.chat.completions.create(
         model = MODEL_NAME,
        messages = [{'role': 'system', 'content':'you are a decision router. you will look at the query and decide if you need to look up the documents. give only either YES/NO'},
        {'role':'user','content':query}]
)
    return model_resp.choices[0].message.content.strip().upper()

def retrieve_and_answer(query: str) -> str:
    return "RAG yet to be implemented"

def answer_directly(query: str) -> str:
    response = client.chat.completions.create(
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
        answer = retrieve_and_answer(request.query)
    elif 'NO' in decision:
        answer = answer_directly(request.query)
    else:
        answer = answer_directly(request.query) # this is the default fallback

    return QueryResponse(ans=answer)


# uploading endpoints
@app.post('/upload')
async def upload_document(file: UploadFile):
    contents = await file.read()
    with open(f'uploads/{file.filename}','wb') as f:
        f.write(contents)
    return {'filename': file.filename, 'status': 'uploaded'}


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
    collection.add(documents = chunks, embeddings = embeddings, ids = ids)
    return collection


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
    mem_after = process.memory_info().rss / (1024 **2)
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Memory delta: {mem_after - mem_before:.2f} MB")

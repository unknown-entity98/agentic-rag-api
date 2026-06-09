from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from pydantic import BaseModel, Field
from typing import Annotated
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME")

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

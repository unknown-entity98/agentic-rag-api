
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class QueryRequest(BaseModel):
    user: str
    query: Annotated[str, Field(min_length=1)]

class QueryResponse(BaseModel):
    ans: str

@app.get('/health')
def check_status():
    return {"status": "ok"}

@app.post('/ask')
def ask_question(request: QueryRequest) -> QueryResponse:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content":request.query} 
        ]
    )
    answer = response.choices[0].message.content
    return QueryResponse(ans=answer)

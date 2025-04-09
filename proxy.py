# 用来转发数据
import argparse
import uvicorn
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()
host = "10.120.20.225"
port = 9000

class Message(BaseModel):
    query: list
    query_type: list
    query_args: dict

@app.post("/query")
def proxy(message: Message):
    """
    A decorator to proxy a function call.
    """
    # message = message.model_dump("json")
    message = {"query": message.query, "query_type": message.query_type, "query_args": message.query_args}
    return requests.post(f"http://{host}:{port}/query", json=message).json()

if __name__ == "__main__":
    uvicorn.run(app, port=30027, host="0.0.0.0")    

# 用来转发数据
import argparse
import uvicorn
import requests
from fastapi import FastAPI

app = FastAPI()
host = "10.120.20.225"
port = 9000

@app.get("/query")
def proxy(message):
    """
    A decorator to proxy a function call.
    """
    return requests.post(f"http://{host}:{port}/query", json=message)

if __name__ == "__main__":
    uvicorn.run(app, port=30027, host="0.0.0.0")    

from mathSolver import math_query
from utils.searchr1.retriever import search
from classifier import QueryClassifier
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import uvicorn
import requests
from args import EvaluateParams

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
    ]
)

query_classifier = QueryClassifier(model_path="/data/hyhping/BAAI/bge-m3",
                                        svm_path="outputs/model", 
                                   model_dir="./outputs/model", 
                                   device=["cuda:4"])
app = FastAPI()

class Query(BaseModel):
    query: str
    query_type: str
    query_args: dict

@app.post("/query")
async def query(Query: Query):
    """
    Receive a query and respond with the result.
    
    The query is expected to be a JSON object with a 'query' field and a 'query_type' field.
    The 'query_type' field can be one of 'math', 'search', or 'classifier'.
    If 'query_type' is 'math', the query is solved using the math solver.
    If 'query_type' is 'search', the query is searched using the search engine.
    If 'query_type' is 'classifier', the query is classified using the query classifier.
    If 'query_type' is not specified, the query is classified using the query classifier in default.
    
    For math query, the result will contains 2 parts:
    - result[0] is the chain of thoughts
    - result[1] is the final answer
    For search query, the result will all the searching results:
    """
    logging.info(f"Received query: {Query}")
    query = Query.query
    query_type = Query.query_type
    query_args = Query.query_args
    if query_type == "classifier":
        query_type = query_classifier(query)
        logging.info(f"Query type: {query_type}")
    elif query_type not in ["math", "search"]:
        query_type = query_classifier(query)
        logging.info(f"Query type: {query_type}")
    if query_args is None:
        logging.warning("query_args is None, using template values")
        if query_type == "search":
            query_args = {
                "topk": 3,
                "return_scores": True
            }
        else:
            query_args = {
                "prompt_type": "tool-integrated",
                "model_name_or_path": "/data/hyhping/Qwen/Qwen2.5-Math-7B-Instruct"
            }
    if query_type == "math":
        args = EvaluateParams()
        args.prompt_type = query_args["prompt_type"]
        args.model_name_or_path = query_args["model_name_or_path"]
        result = math_query(query, args)
        return {"result": result[0], 'answer': result[1]}
    elif query_type == "search":
        args = Query.query_args
        result = search(query, topk=args["topk"], return_scores=args["return_scores"])
        return {"result": result["passages"], "scores": result["scores"]} if args["return_scores"] else {"result": result["passages"]}
    else:
        logging.info(f"Unknown query type: {query_type}")
        return {"result": "Unknown query type"}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
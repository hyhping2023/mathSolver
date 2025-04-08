from mathSolver import math_query
from utils.searchr1.retriever import search
from classifier import QueryClassifier
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import uvicorn
from typing import *
from utils.qwen.parser import set_seed
import multiprocessing as mp
import requests
# from vllm import LLM
set_seed(0)
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

def vllm_query(query:list, prompt_type:str="tool-integrated", max_tokens_per_call:int=4096, 
               temperature=0, n_sampling=1, top_p=1):
    args = {
        "prompt_type": prompt_type,
        "temperature": temperature,
        "max_tokens_per_call": max_tokens_per_call,
        "n_sampling": n_sampling,
        'top_p': top_p
    }
    message = {
        "query": query,
        "query_args": args
    }
    response = requests.post("http://localhost:8001/query", json=message).json()
    return response


class Query(BaseModel):
    query: list[str]
    query_type: list[str]
    query_args: dict

@app.post("/query")
def query(Query: Query):
    """
    Receive a query and respond with the result.
    
    The query is expected to be a JSON object with a 'query' field and a 'query_type' field.
    The 'query_type' field can be one of 'math', 'search', or 'classifier'.
    If 'query_type' is 'math', the query is solved using the math solver.
    If 'query_type' is 'search', the query is searched using the search engine.
    If 'query_type' is 'classifier', the query is classified using the query classifier.
    If 'query_type' is not specified, the query is classified using the query classifier in default.
    If the query_type only has one type, it will be used for all queries.

    For math query, the result will contains 2 parts:
    - 'result' is the chain of thoughts or the corresponding code
    - 'answer' is the final answer
    For search query, the result will all the searching results:
    - 'result' is the result of the query.
    - 'scores' is the scores of the query, which is a list.
    """
    logging.info(f"Received query: {Query}")
    query = Query.query
    query_type = Query.query_type
    query_args = Query.query_args
    assert len(query) == len(query_type) or len(query_type) == 1, "query and query_type should have the same length"
    if len(query_type) == 1:
        query_type = query_type * len(query)
    math_tasks = []
    math_index = []
    search_tasks = []
    search_index = []
    if query_args is None:
        query_args ={
                "topk": 3,
                "return_scores": True,
                "prompt_type": "tool-integrated",
            }
        logging.warning("query_args is None, using template values")
    for i, (q, t) in enumerate(zip(query, query_type)):
        if t == "classifier":
            query_type = query_classifier(q)
        elif t not in ["math", "search"]:
            query_type = query_classifier(q)
        else:
            query_type = t
        logging.info(f"Query type: {query_type}")
        if query_type == "math":
            math_tasks.append(q)
            math_index.append(i)
        else:
            search_tasks.append(q)
            search_index.append(i)
    # pool = mp.Pool(processes=64)
    # tasks = [pool.apply_async(math_query, args=(task, query_args)) for task in math_tasks]
    # pool.close()
    # pool.join()
    # math_results = [task.get() for task in tasks]
    math_results = vllm_query(math_tasks, prompt_type=query_args["prompt_type"])
    search_results = search(search_tasks, topk=query_args["topk"], return_scores=query_args["return_scores"])
    # merge the results according to the index
    results = [None] * len(query)
    for i, result in enumerate(math_results):
        results[math_index[i]] = {"result": result[0]+f" After confirmation using Python, the final answer is {result[1]}.", 'answer': result[1]}
    for i in range(len(search_results['passages'])):
        results[search_index[i]] = {"result": search_results["passages"][i], "scores": search_results["scores"][i]} if query_args["return_scores"] else {"result": search_results["passages"][i]}
    return results
    # if query_type == "math":
    #     args = EvaluateParams()
    #     args.prompt_type = query_args["prompt_type"]
    #     args.model_name_or_path = query_args["model_name_or_path"]
    #     result = math_query(query, args)
    #     return {"result": result[0]+f" After confirmation using Python, the final answer is {result[1]}.", 'answer': result[1]}
    # elif query_type == "search":
    #     args = Query.query_args
    #     result = search(query, topk=args["topk"], return_scores=args["return_scores"])
    #     return {"result": result["passages"], "scores": result["scores"]} if args["return_scores"] else {"result": result["passages"]}
    # else:
    #     logging.info(f"Unknown query type: {query_type}")
    #     return {"result": "Unknown query type"}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
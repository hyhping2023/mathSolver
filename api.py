import requests
from typing import *

def send_query(query:list, query_type:Union[str, List[str]]=['classifier'], query_args = None):
    """
    Send a query to the server and get the result.
    
    Args:
        query (str): The query to send to the server.
        query_type (str, optional): The type of the query. Defaults to "".
        query_args (dict, optional): The arguments for the query. Defaults to None.
    
    For query_args, the following keys are expected:
        if your query typr is 'math', the query_args should be:
            query_args = {
                "prompt_type": "tool-integrated", or "cot"
                "model_name_or_path": "path/to/your/model"
            }
        if your query type is 'search', the query_args should be:
            query_args = {
                "topk": int,
                "return_scores": bool
            }
        if your query type is 'classifier', the query_args should be:
            you should specify all above, which is:
            query_args = {
                "prompt_type": "tool-integrated", or "cot"
                "model_name_or_path": "path/to/your/model",
                "topk": int,
                "return_scores": bool
            }
    
    Returns:
        dict: The result of the query.
            for math query, the result will contains:
                'result': The chain of thoughts.
                'answer': The final answer.
            for seach query, the result will contains:
                'result': The result of the query.
                'scores': The scores of the query.
    """
    if isinstance(query_type, str):
        query_type = [query_type] * len(query)
    message = {"query": query, "query_type": query_type, "query_args": query_args}
    return requests.post("http://0.0.0.0:9000/query", json=message).json()
    # return requests.post("http://10.120.16.175:30027/query", json=message).json()


if __name__ == "__main__":
    import time, json
    start_time = time.time()
    # test_template = "data/template.jsonl"
    # questions = []
    # with open(test_template, "r") as f:
    #     for line in f:
    #         test_data = json.loads(line)
    #         questions.append(test_data["question"])
    # queries = questions

    queries = ["What is the capital of France?", "If there are 10 eggs in a basket, and there are twice as many eggs in a second basket, how many eggs are in both baskets put together?"] * 10
    
    query_types = ["search", "math"] * 10
    query_types = ['classifier']
    query_args = {
        "prompt_type": "tool-integrated",
        "topk": 5,
        "return_scores": True
    }
    results = send_query(queries, query_types, query_args)
    for result in results:
        print(result) 
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
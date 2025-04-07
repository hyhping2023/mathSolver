import requests
import logging

def send_query(query, query_type = "", query_args = None):
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
        'result': The result of the query.
        'scores': The scores of the query.
    """
    message = {"query": query, "query_type": query_type, "query_args": query_args}
    return requests.post("http://0.0.0.0:9000/query", json=message).json()

if __name__ == "__main__":
    query = "What is the integral of x^2 from 0 to 1?"
    query_type = ""
    result = send_query(query, query_type, {"prompt_type": "tool-integrated", "model_name_or_path": "/data/hyhping/Qwen/Qwen2.5-Math-7B-Instruct/"})
    print(result)
    query = "What is the capital of France?"
    query_type = ""
    result = send_query(query, query_type, {"topk": 5, "return_scores": False})
    print(result)
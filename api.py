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
        str: The result from the server.
    """
    message = {"query": query, "query_type": query_type, "query_args": query_args}
    return requests.post("http://10.120.20.225:8000/query", json=message).json()['result']

if __name__ == "__main__":
    query = "What is the integral of x^2 from 0 to 1?"
    query_type = ""
    result = send_query(query, query_type)
    print(result)
    query = "What is the capital of France?"
    query_type = ""
    result = send_query(query, query_type)
    print(result)
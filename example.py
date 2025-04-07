import requests

def send_query(query, query_type = ""):
    message = {"query": query, "query_type": query_type}
    return requests.post("http://127.0.0.1:8000/query", json=message).json()['result']

if __name__ == "__main__":
    query = "What is the integral of x^2 from 0 to 1?"
    query_type = ""
    result = send_query(query, query_type)
    print(result)
    query = "What is the capital of France?"
    query_type = ""
    result = send_query(query, query_type)
    print(result)
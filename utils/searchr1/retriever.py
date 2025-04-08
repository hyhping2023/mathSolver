import requests
from typing import Union

def search(query: list, topk: int = 3, return_scores: bool = True):
    """
    Send a query to the local retrieval server and get the topk results.
    
    Args:
        query (str): The query to send to the server.
        topk (int, optional): The number of results to return. Defaults to 3.
        return_scores (bool, optional): Whether to return the scores for the results. Defaults to True.
    
    Returns:
        dict: A dictionary with the topk results. If return_scores is True, the dictionary will contain two keys: "passages" and "scores". Otherwise, it will only contain the "passages" key. The value of "passages" is a string containing the title and text of each passage, separated by a newline. The value of "scores" is a list of the scores for the results, in descending order of relevance.
    """
    payload = {
            "queries": query,
            "topk": topk,
            "return_scores": return_scores
        }
    results = requests.post("http://127.0.0.1:8002/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference
    if return_scores:
        return {"passages": [_passages2string(result) for result in results], "scores": [[doc['score'] for doc in result] for result in results]}
    return {"passages": [_passages2string([result]) for result in results]}
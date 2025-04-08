
###############################

# Note: The following code is for testing vllm_query

###############################


from args import EvaluateParams
import json, requests
test_template = "data/template.jsonl"
questions = []
answers = []
model = []
cot = []
with open(test_template, "r") as f:
    for line in f:
        test_data = json.loads(line)
        questions.append(test_data["question"])
        answers.append(test_data["pred"])
        model.append(test_data["gt"])
        cot.append(test_data["code"][0])
queries = questions
# args = EvaluateParams()
# args.prompt_type = "tool-integrated"
# args.model_name_or_path = "/data/hyhping/Qwen/Qwen2.5-Math-7B-Instruct"
# args.max_tokens_per_call = 2048
args={
    "prompt_type": "cot",
    "temperature": 0,
    "max_tokens_per_call": 4096,
    "n_sampling": 1,
    'top_p': 1
}
results = requests.post("http://localhost:8001/query", json={"query": queries, "query_args": args}).json()
acc = 0
ref = 0
for i in range(len(results)):
    print(results[i])
    if results[i][1] == answers[i][0]:
        ref += 1
    else:
        print("results:\n{}\nCOT:\n{}".format(results[i][0], cot[i]))
        print("answers:\n{}\nmodel:\n{}".format(results[i][1], answers[i][0]))
    if results[i][1] == model[i]:
        acc += 1
    else:
        print(results[i][1], " != ", model[i], " != ", answers[i][0])
print("Accuracy: ", acc / len(results), "Reference: ", ref / len(results))
from utils.qwen.python_executor import PythonExecutor
from utils.qwen.model_utils import load_hf_lm_and_tokenizer, generate_completions
from utils.qwen.utils import save_jsonl, construct_prompt
from utils.qwen.parser import *
from utils.qwen.trajectory import *
from args import EvaluateParams
from openai import OpenAI
import logging

# 配置日志记录器
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),  # 输出到控制台
#         logging.FileHandler('app.log')  # 输出到文件
#     ]
# )

PROMPT = {
    "cot":[
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": None}
        ],
    "tool":[
            {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
            {"role": "user", "content": None}
        ]
}

def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def math_query(query:str, args: EvaluateParams = EvaluateParams(),
                vllm_port = 8001, data_name = ""):
    """
    Use the vllm server to get the solution to the query and execute the program.

    Parameters:
    query (str): The math query to be solved.
    args (EvaluateParams): The parameters for evaluation.
    vllm_port (int): The port of the vllm server.
    data_name (str): The name of the dataset.

    Returns:
    code (str): The code of the solution or the cot of the solution.
    answer (str): The answer of the solution.
    """
    assert isinstance(query, str), "query should be a string"
    assert len(query) > 0, "query should not be empty"
    
    # vllm config
    if not isinstance(vllm_port, str):
        try:
            open_api_base = "http://localhost:{}/v1".format(str(vllm_port))
        except:
            open_api_base = "http://localhost:8001/v1"
    else:
        open_api_base = "http://localhost:{}/v1".format(vllm_port)
    open_api_key = "EMPTY"
    client = OpenAI(
        base_url=open_api_base,
        api_key=open_api_key,
    )

    # message template
    if "tool" in args.prompt_type:
        executer = PythonExecutor(get_answer_from_stdout=True)
        message_template = PROMPT["tool"]
    elif "cot" in args.prompt_type:
        executer = None
        message_template = PROMPT["cot"]
    else:
        logging.warn("Unsupported prompt type, please check the prompt type. Deafault to 'tool'.")
        executer = PythonExecutor(get_answer_from_stdout=True)
        message_template = PROMPT["tool"]
    
    # construct prompt
    full_prompt = message_template
    full_prompt[1]["content"] = query

    # stop words constructing
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    if "cot" in args.prompt_type:
        stop_words.append("\n\nQuestion:")
    elif "tool" in args.prompt_type:
        stop_words.extend(["\n\n---", "```output"])
    else:
        logging.warn("Unsupported prompt type, please check the prompt type. Deafault to 'tool'.")
        stop_words.append("\n\n---", "```output")

    # get result from vllm server and try for some times
    max_func_call = 1 if args.prompt_type == "cot" else 4
    for epoch in range(max_func_call):
        response = client.chat.completions.create(
            messages=full_prompt,
            n=args.n_sampling,
            model=args.model_name_or_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens_per_call,
            top_p=args.top_p,
            stop=stop_words,
        )

        response = response.choices[0].message.content.strip()
        full_prompt[1]["content"] += response # add response to the query prompt
        end_prompt = full_prompt[1]["content"]
        if "boxed" not in response and response.endswith("```"):
            program = extract_program(response)
        else: # "COT" situation or "TIR" finished
            break
        program_result = executer.sync_apply(program)
        result, report = program_result
        exec_result = result if result else report
        exec_result = f"\n```output\n{exec_result}\n```\n"
        full_prompt[1]["content"] += exec_result
        if epoch == max_func_call - 1:
            full_prompt[1]["content"] += "\nReach max function call limit."

    code = end_prompt.split(query)[-1].strip()
    for stop_word in stop_words:
        if stop_word in code:
            code = code.split(stop_word)[0].strip()
    if "tool" in args.prompt_type:
        answer = end_prompt.split(query)[-1].strip()
        for stop_word in stop_words:
            if stop_word in answer:
                answer = answer.split(stop_word)[-1].strip()
        code = code.split("```")[0].strip()
    else: # In COT situation, the 'code' is actually the thinking content
        answer = code
    answer = extract_answer(answer, data_name)
    return (code, answer)
    
    
if __name__ == "__main__":
    set_seed(0)
    question = "A car travels 60 miles in 1 hour. How far will it travel in 3 hours?"
    args = EvaluateParams()
    args.model_name_or_path = "/data/hyhping/Qwen/Qwen2.5-Math-7B-Instruct/"
    args.max_tokens_per_call = 2048
    # args.prompt_type = "cot"
    import json
    test_template = "/home/asc1/hyhping/Qwen2.5-Math/evaluation/outputs/data/hyhping/Qwen/Qwen2.5-Math-7B-Instruct/math_eval/aime24/test_tool-integrated_-1_seed0_t0.0_s0_e-1.jsonl"
    
    import multiprocessing as mp
    pool = mp.Pool(processes=64)
    tasks = []
    reference_answer = []
    reference_cot = []
    correct_answer = []
    with open(test_template, "r") as f:
        num = 0
        for line in f:
            test_data = json.loads(line)
            question = test_data["question"]
            # math_query(question, args)
            tasks.append(pool.apply_async(math_query, args=(question, args, )))
            reference_answer.append(test_data["pred"][0])
            reference_cot.append(test_data['code'][0])
            correct_answer.append(test_data["gt"])
            # num += 1
            # if num>100:
            #     break
    pool.close()
    pool.join()
    acc = 0
    ref = 0
    for i,result in enumerate(tasks):
        code, answer = result.get()
        if answer != reference_answer[i]:
            print(answer, reference_answer[i], correct_answer[i])
        else:
            ref += 1
        if answer == correct_answer[i]:
            acc += 1
    print("acc: ", acc/len(tasks))
    print("ref: ", ref/len(tasks))
            # print("current code: ", code)
            # print("reference code: ", reference_cot[i])
            # print("Question: ", question)
            # print("Thought: ", thought)
            # print("reference answer: ", test_data["pred"])
            # print("Answer: ", answer)

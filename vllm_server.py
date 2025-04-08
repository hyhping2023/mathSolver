from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
from utils.qwen.python_executor import PythonExecutor
from utils.qwen.parser import *
from utils.qwen.trajectory import *
from args import EvaluateParams, PROMPT_BATCH
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from utils.qwen.parser import set_seed

set_seed(0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
    ]
)

llm = LLM(
    model="/data/hyhping/Qwen/Qwen2.5-Math-7B-Instruct",
    tensor_parallel_size=4,
    pipeline_parallel_size=1,
    trust_remote_code=True,
)

app = FastAPI()

class Query(BaseModel):
    query: list[str]
    query_args: dict

@app.post("/query")
def math_batch_query(query: Query):
    # init python executor
    examples = query.query
    args = query.query_args
    prompt_type = args['prompt_type']
    
    if "tool" in prompt_type:
        executor = PythonExecutor(get_answer_from_stdout=True)
        message_template = PROMPT_BATCH["tool"]
    elif "cot" in prompt_type:
        message_template = PROMPT_BATCH["cot"]
    else:
        logging.warn("Unsupported prompt type, please check the prompt type. Deafault to 'tool'.")
        executor = PythonExecutor(get_answer_from_stdout=True)
        message_template = PROMPT_BATCH["tool"]
        prompt_type = "tool-integrated"

    samples = []
    for idx, example in enumerate(tqdm(examples, total=len(examples))):
        samples.append({
            "idx": idx,
            "prompt": example.join(message_template),
            })

    samples *= args['n_sampling']

    remain_prompts = samples.copy()
    end_prompts = []

    max_func_call = 1 if prompt_type == 'cot' else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    if "cot" in prompt_type:
        stop_words.append("\n\nQuestion:")
    elif "tool" in prompt_type:
        stop_words.extend(["\n\n---", "```output"])
    else:
        logging.warn("Unsupported prompt type, please check the prompt type. Deafault to 'tool'.")
        stop_words.append("\n\n---", "```output")

    # start inference
    for epoch in range(max_func_call):
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [sample["prompt"] for sample in current_prompts]
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=args['temperature'],
                top_p=args["top_p"],
                max_tokens=args["max_tokens_per_call"],
                n=1,
                stop=stop_words
                )
        )

        outputs = sorted(
            outputs, key=lambda x: int(x.request_id)
        )  # sort outputs by request_id
        outputs = [output.outputs[0].text for output in outputs]

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for item, output in zip(current_prompts, outputs):
            i, query = item['idx'], item['prompt']
            output = output.rstrip()
            query += output
            if "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append({"idx": i, "prompt": query})
                remain_codes.append(program)
            else:
                end_prompts.append({"idx": i, "prompt": query})
                continue
        # execute the remain prompts
        if not 'cot' in prompt_type:
            remain_results = executor.batch_apply(remain_codes)
            for k in range(len(remain_prompts)):
                i, query = remain_prompts[k]['idx'], remain_prompts[k]['prompt']
                res, report = remain_results[k]
                exec_result = res if res else report
                exec_result = f"\n```output\n{exec_result}\n```\n"
                query += exec_result
                # not end
                if epoch == max_func_call - 1:
                    query += "\nReach max function call limit."
                remain_prompts[k]['prompt'] = query

    # unsolved samples
    logging.info(f"Unsolved samples: {len(remain_prompts)}")
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x['idx'])

    # remove input_prompt from end_prompt
    results = []
    assert len(samples) == len(end_prompts)
    for i in range(len(samples)):
        end_prompt = end_prompts[i]['prompt']
        query = samples[i]['prompt']
        code = end_prompt.split(query)[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        if "tool" in prompt_type:
            answer = end_prompt.split(query)[-1].strip()
            for stop_word in stop_words:
                if stop_word in answer:
                    answer = answer.split(stop_word)[-1].strip()
            # code = code.split("```")[0].strip()
        else: # In COT situation, the 'code' is actually the thinking content
            answer = code
        answer = strip_string(extract_answer(answer, ""))
        results.append({
            "idx": i,
            "code": code,
            "answer": answer,
            })
    results = sorted(results, key=lambda x: x['idx'])
    return [[result['code'], result['answer']] for result in results]

if __name__ == "__main__":
    # queries = [
    #     "What is 2 + 2?",
    #     "What is the capital of France?",
    #     "What is the square root of 16?",
    #     "What is the derivative of x^2?",
    #     "What is the integral of x^2?",
    #     "What is the value of pi?",
    #     "What is the value of e?",
    #     "Please calculate the factorial of 5.",
    # ]
    uvicorn.run(app, host="0.0.0.0", port=8001)
    # evaluate()


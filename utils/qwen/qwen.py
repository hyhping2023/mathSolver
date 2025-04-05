from transformers import AutoModelForCausalLM, AutoTokenizer
from math_eval import prepare_data, parse_question, parse_ground_truth, is_multi_choice
from tqdm import tqdm
import argparse, json
from hyhping.mathSolver.utils.qwen.utils import set_seed, save_jsonl
from parser import extract_answer, choice_answer_clean
from evaluate import evaluate
from model_utils import load_hf_lm_and_tokenizer
from vllm import LLM, SamplingParams
from openai import OpenAI

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="aime24,amc23", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args

def Qwen_chat(model, tokenizer, messages):
    device = model.device
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def Qwen_chat_vllm(model, tokenizer, messages, params):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # generate outputs
    outputs = model.generate([text], params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
    return response

def Qwen_chat_vllm_api(idx, model_name, prompt, args):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    if "cot" in args.prompt_type:
        messages = PROMPT["cot"]
    elif "tool" in args.prompt_type:
        messages = PROMPT["tool"]

    messages[1]['content'] = prompt

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=2048,
    )

    return {"idx": idx, "response": chat_response.choices[0].message.content}

def main(args):
    data_names = args.data_names.split(",")
    model_name = "/data/hyhping/Qwen/Qwen2.5-Math-7B-Instruct/"
    if "cot" in args.prompt_type:
        messages = PROMPT["cot"]
    elif "tool" in args.prompt_type:
        messages = PROMPT["tool"]
    else:
        raise NotImplementedError

    if args.use_vllm:
        pass
        # model = LLM(model_name, tensor_parallel_size=4)
        # params = SamplingParams(
        #     temperature=args.temperature,
        #     top_p=args.top_p,
        #     max_tokens=2048,
        # )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for data_name in data_names:
        examples, processed_samples, out_file = prepare_data(data_name, args)
        if len(examples) > 0:
            print(examples[0])
        samples = []
        for example in tqdm(examples, total=len(examples)):
            idx = example["idx"]
            # parse question and answer
            example["question"] = parse_question(example, data_name)
            if example["question"] == "":
                continue
            gt_cot, gt_ans = parse_ground_truth(example, data_name)
            example["gt_ans"] = gt_ans

            sample = {
                "idx": idx,
                "question": example["question"],
                "gt_cot": gt_cot,
                "gt": gt_ans,
            }

            # add remain fields
            for key in [
                "level",
                "type",
                "unit",
                "solution_type",
                "choices",
                "solution",
                "ques_type",
                "ans_type",
                "answer_type",
                "dataset",
                "subfield",
                "filed",
                "theorem",
                "answer",
            ]:
                if key in example:
                    sample[key] = example[key]
            samples.append(sample)

        # repeat n times
        input_prompts = [
            sample["question"] for sample in samples for _ in range(args.n_sampling)
        ]

        input_prompts = [(i, prompt) for i, prompt in enumerate(input_prompts)]
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
        if args.prompt_type in ["cot"]:
            stop_words.append("\n\nQuestion:")
        if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
            stop_words.extend(["\n\n---", "```output"])
        elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
            stop_words.extend(["Instruction", "Response"])
        elif "jiuzhang" in args.prompt_type:
            stop_words.append("\n\n## Question")
        elif "numina" in args.prompt_type:
            stop_words.append("\n### Problem")
        elif "pure" in args.prompt_type:
            stop_words.append("\n\n\n")
        
        return_samples = []
        queue = []
        import multiprocessing
        pool = multiprocessing.Pool(128)
        for i, input_prompt in tqdm(input_prompts, total=len(input_prompts)):
            messages[1]['content'] = input_prompt
            if args.use_vllm:
                # response = Qwen_chat_vllm(model, tokenizer, message, params)
                # print(Qwen_chat_vllm_api(i, model_name, message, args))
                queue.append(pool.apply_async(Qwen_chat_vllm_api, 
                                              args=(i, model_name, input_prompt, args, )))
            else:
                response = Qwen_chat(model, tokenizer, messages)
                return_samples.append({"idx": i, "response": response})
        if args.use_vllm:
            pool.close()
            pool.join()
            for res in queue:
                result = res.get()
                return_samples.append(result)
        return_samples = sorted(return_samples, key=lambda x: x["idx"])
        codes = []
        results = []
        for i in range(len(input_prompts)):
            end_prompt = return_samples[i]['response']
            code = end_prompt.strip()
            answer = code
            for stop_word in stop_words:
                if stop_word in code:
                    code = code.split(stop_word)[0].strip()
            codes.append(code)
            if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
                answer = answer.split("```output")[-1]
                pred = extract_answer(answer, data_name)
            else:
                pred = extract_answer(code, data_name)
            results.append(pred)

        all_samples = []
        for i, sample in enumerate(samples):
            code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
            result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
            preds = [item for item in result]
            for j in range(len(preds)):
                if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                ]:
                    preds[j] = choice_answer_clean(code[j])
                elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                    # remove any non-choice char
                    preds[j] = "".join(
                        [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                    )

            sample.update({"code": code, "pred": preds, "report": None})
            all_samples.append(sample)

        all_samples.extend(processed_samples)
        all_samples, result_json = evaluate(
            samples=all_samples,
            data_name=data_name,
            prompt_type=args.prompt_type,
            execute=True,
        )
    
        if len(processed_samples) < len(all_samples) and args.save_outputs:
            save_jsonl(all_samples, out_file)

        with open(
            out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
        ) as f:
            json.dump(result_json, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
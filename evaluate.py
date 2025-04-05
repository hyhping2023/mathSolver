import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.qwen.evaluate import evaluate
from utils.qwen.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.qwen.parser import *
from utils.qwen.trajectory import *
from utils.qwen.data_loader import load_data
from args import EvaluateParams
from mathSolver import PROMPT

'''
{dataset name:
    [ --> storng all the samples
        {
            idx: 0
            content: "result"
        }
    ]
}
'''

def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def prepare_data(data_name:str, args:EvaluateParams = EvaluateParams()):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file

def get_input(data_name:str = "gsm8k,math,aime24,amc23", args:EvaluateParams = EvaluateParams()):
    data_name = data_name.split(",")
    data_name = [name.strip() for name in data_name]
    data = [(name, prepare_data(name, args)) for name in data_name]
    final_samples = {}
    for data_name, d in data:
        samples = []
        for example in tqdm(d[0], total=len(d[0])):
            idx = example["idx"]

            # parse question and answer
            example["question"] = parse_question(example, data_name)
            if example["question"] == "":
                continue
            gt_cot, gt_ans = parse_ground_truth(example, data_name)
            example["gt_ans"] = gt_ans
            if args.prompt_type == "cot":
                full_prompt = PROMPT['cot']
            else:
                full_prompt = PROMPT['tool']
            full_prompt[1]['content'] = example["question"]

            if idx == args.start:
                print(full_prompt)

            sample = {
                "idx": idx,
                "question": example["question"],
                "gt_cot": gt_cot,
                "gt": gt_ans,
                "prompt": full_prompt,
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
        final_samples[data_name] = samples
        for _ in range(args.n_sampling-1):
            final_samples[data_name].extend(samples)
    return final_samples

def evalate_output(input_data:list, output_data:list, args:EvaluateParams = EvaluateParams()):
    assert len(input_data.keys()) == len(output_data.keys()), "input and output dataset numbers mismatch"
    input_data = {k:sorted(v, key=lambda x: x["idx"]) for k, v in input_data.items()}
    output_data = {k:sorted(v, key=lambda x: x["idx"]) for k, v in output_data.items()}
    print("input data:", len(input_data['aime24']))
    print("output data:", len(output_data['aime24']))
    correct = 0
    all_samples = []
    for data_name, results in input_data.items():
        for i in range(len(results)):
            result = output_data[data_name][i * args.n_sampling : (i + 1) * args.n_sampling]
            print(result[0])
            preds = [item['content'] for item in result]
            model = input_data[data_name][i]
            for j in range(len(preds)):
                if model["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                ]:
                    preds[j] = choice_answer_clean(preds[j])
                elif is_multi_choice(model["gt"]) and not is_multi_choice(preds[j]):
                    # remove any non-choice char
                    preds[j] = "".join(
                        [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                    )
            model.pop("prompt")
            model.update({"pred": preds})
            all_samples.append(model)
        all_samples, result_json = evaluate(
            samples=all_samples,
            data_name=data_name,
            prompt_type=args.prompt_type,
            execute=True
        )
    return all_samples

if __name__ == "__main__":
    args = EvaluateParams()
    args.n_sampling = 1
    final_samples = get_input("aime24", args)
    print(len(final_samples['aime24']))
    test = {"aime24":[{"idx":i, "content":"902"} for i in range(30, 0, -1)] }
    all_samples = evalate_output(final_samples, test, args)


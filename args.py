class EvaluateParams:
    def __init__(self, 
            data_names:str = "gsm8k,math", 
            data_dir:str = "utils/qwen/data", 
            model_name_or_path:str = "Qwen/Qwen2.5-Math-7B-Instruct", 
            output_dir:str = "./output", 
            prompt_type:str = "tool-integrated", 
            split:str = "test", 
            num_test_sample:int = -1, 
            seed:int = 0, start:int = 0, end:int = -1, 
            temperature:float = 0, 
            n_sampling:int = 1, 
            top_p:float = 1, 
            max_tokens_per_call:int = 2048, 
            shuffle:bool = False, 
            save_outputs:bool = False,
            overwrite:bool = False, 
            num_shots:int = 0, 
            apply_chat_template:bool = False,
            adapt_few_shot:bool = False,):
        self.data_names = data_names
        self.data_dir = data_dir
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.prompt_type = prompt_type
        self.split = split
        self.num_test_sample = num_test_sample
        self.seed = seed
        self.start = start
        self.end = end
        self.temperature = temperature
        self.n_sampling = n_sampling
        self.top_p = top_p
        self.max_tokens_per_call = max_tokens_per_call
        self.shuffle = shuffle
        self.save_outputs = save_outputs
        self.overwrite = overwrite
        self.num_shots = num_shots
        self.apply_chat_template = apply_chat_template
        self.adapt_few_shot = adapt_few_shot
        self.top_p = (
            1 if self.temperature == 0 else self.top_p
        )  # top_p must be 1 when using greedy sampling (vllm)


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

PROMPT_BATCH ={
    "cot":[
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n"
        ],
    "tool":[
        "<|im_start|>system\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n"
        ],
}
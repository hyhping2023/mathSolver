# 将jsonl里的所有问题合并为一个pd.DataFrame

import pandas as pd

jsonl_file = "/home/asc1/hyhping/Qwen2.5-Math/evaluation/outputs/data/hyhping/Qwen/Qwen2.5-Math-7B-Instruct/math_eval/aime24/test_tool-integrated_-1_seed0_t0.0_s0_e-1.jsonl"
name = jsonl_file.split("/")[-2]
output_file = f"data/{name}.parquet"
df = pd.DataFrame()
with open(jsonl_file, "r") as f:
    for line in f:
        data = pd.read_json(line, lines=True)
        data = data[["question"]]
        df = pd.concat([df, data], ignore_index=True)
df = df.reset_index(drop=True)
df = df.dropna(subset=["question"])
#保存
df.to_parquet(output_file, index=False)
print(f"数据已保存到 {output_file}")
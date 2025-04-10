# #!/bin/bash

# # 启动 VLLM 服务的脚本
# # 请根据你的环境修改以下参数

# # 模型路径（替换为你本地的模型路径）
# # 如果接收到一个参数，则使用该参数作为模型路径
# if [ $# -eq 1 ]; then
#     MODEL_PATH=$1
# else
#     echo "No model path provided. Using default path."
#     # 如果没有提供模型路径，则使用默认路径
#     # 默认模型路径（请根据实际情况修改）
#     # 默认模型路径
#     MODEL_PATH="/path/to/your/model"
# fi


# # 启动服务
# echo "Starting VLLM service..."
# echo "Model path: $MODEL_PATH"

# # 检查 CUDA 是否可用
# if ! command -v nvidia-smi &> /dev/null; then
#     echo "CUDA is not installed or not available. Exiting."
#     exit 1
# fi

# # 检查 VLLM 是否安装
# if ! command -v vllm &> /dev/null; then
#     echo "VLLM is not installed. Please install it first."
#     exit 1
# fi
# # 检查模型路径是否存在
# if [ ! -d "$MODEL_PATH" ]; then
#     echo "Model path does not exist: $MODEL_PATH"
#     exit 1
# fi

# # 启动 VLLM 服务
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
# TP=4
# MEMORY_UTILIZATION=0.6
# PORT=8001
# vllm serve $MODEL_PATH -tp $TP --gpu-memory-utilization $MEMORY_UTILIZATION --port $PORT

# echo "VLLM service started successfully"

TOKENIZERS_PARALLELISM=false python vllm_server.py
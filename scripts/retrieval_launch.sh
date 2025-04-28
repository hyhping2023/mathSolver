file_path=/data2/share/hanxu/hyh/dataset
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
# retriever=intfloat/e5-base-v2
retriever=/data2/share/hanxu/hyh/retriever

CUDA_VISIBLE_DEVICES=6,7 python utils/searchr1/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever

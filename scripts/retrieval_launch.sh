file_path=/data/hyhping/dataset/
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
# retriever=intfloat/e5-base-v2
retriever=/data/hyhping/retriever

CUDA_VISIBLE_DEVICES=4,5 python utils/searchr1/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever

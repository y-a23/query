
corpus_file=/nfsdata/yiao/medRAG/mixed_corpus/merged_corpus.jsonl # jsonl
# corpus_file=/nfsdata/yiao/medRAG/textbooks/chunk/Anatomy_Gray.jsonl
save_dir=/nfsdata/yiao/medRAG/mixed_corpus/index/merged_corpus_contriever.index
retriever_name=contriever # this is for indexing naming
retriever_model=/nfsdata/yiao/model/contriever

# 设置缓存目录，避免默认缓存目录空间不足的问题
export HF_HOME=/nfsdata/yiao/hf_cache
export TRANSFORMERS_CACHE=/nfsdata/yiao/hf_cache
export DATASETS_CACHE=/nfsdata/yiao/hf_cache

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 8192 \
    --batch_size 1024 \
    --pooling_method mean \
    --faiss_type Flat \
    --faiss_gpu \
    --save_embedding

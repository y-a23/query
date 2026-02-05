
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer, util
import os
import faiss
import json
import torch
import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel

corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    # "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "merged_chunk": ["merged_chunk"],
    "all_chunk": ["pubmed", "textbooks", "wikipedia"]
    # "MedText": ["textbooks", "statpearls"],
    # "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
}

retriever_names = {
    "BM25": ["bm25"],
    "Contriever": ["/nfsdata/yiao/model/contriever"],
    "SPECTER": ["allenai/specter"],
    "MedCPT": ["ncbi/MedCPT-Query-Encoder"],
    "RRF-2": ["bm25", "ncbi/MedCPT-Query-Encoder"],
    "RRF-4": ["bm25", "/nfsdata/yiao/model/contriever", "/nfsdata/yiao/model/specter2_base", "/nfsdata/yiao/model/MedCPT-Query-Encoder"]
}

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

class CustomizeSentenceTransformer(SentenceTransformer): # change the default pooling "MEAN" to "CLS"

    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        """
        Creates a simple Transformer + CLS Pooling model and returns the modules
        """
        print("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(model_name_or_path))
        token = kwargs.get('token', None)
        cache_folder = kwargs.get('cache_folder', None)
        revision = kwargs.get('revision', None)
        trust_remote_code = kwargs.get('trust_remote_code', False)
        if 'token' in kwargs or 'cache_folder' in kwargs or 'revision' in kwargs or 'trust_remote_code' in kwargs:
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
                tokenizer_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
            )
        else:
            transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]

# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings
# def embed(chunk_dir, index_dir, model_name, **kwarg):
    
#     save_dir = os.path.join(index_dir, "embedding")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)

#     fnames = sorted([fname for fname in os.listdir(chunk_dir) if fname.endswith(".jsonl")])

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     with torch.no_grad():
#         for fname in tqdm.tqdm(fnames):
#             fpath = os.path.join(chunk_dir, fname)
#             save_path = os.path.join(save_dir, fname.replace(".jsonl", ".npy"))
#             embed_chunks = None
#             if os.path.exists(save_path):
#                 continue
#             if open(fpath).read().strip() == "":
#                 continue
#             texts = [json.loads(item) for item in open(fpath).read().strip().split('\n')]
#             if "specter" in model_name.lower():
#                 texts = [model.tokenizer.sep_token.join([item["title"], item["content"]]) for item in texts]
#             elif "contriever" in model_name.lower():
#                 texts = [". ".join([item["title"], item["content"]]).replace('..', '.').replace("?.", "?") for item in texts]
#             elif "medcpt" in model_name.lower():
#                 texts = [[item["title"], item["content"]] for item in texts]
#             else:
#                 texts = [concat(item["title"], item["content"]) for item in texts]

#             import pdb; pdb.set_trace()
#             inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
#             outputs = model(**inputs)
#             embed_chunks = mean_pooling(outputs[0], inputs['attention_mask'])
#             import pdb; pdb.set_trace()
#             np.save(save_path, embed_chunks)

#     inputs = model.tokenizer([""], truncation=True, padding=True, return_tensors="pt")
#     outputs = model(**inputs)
#     embed_chunks = mean_pooling(outputs[0], inputs['attention_mask'])
#     return embed_chunks.shape[-1]
def embed(chunk_dir, index_dir, model_name, **kwarg):

    save_dir = os.path.join(index_dir, "embedding")
    
    if "contriever" in model_name:
        model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = CustomizeSentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    fnames = sorted([fname for fname in os.listdir(chunk_dir) if fname.endswith(".jsonl")])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames):
            fpath = os.path.join(chunk_dir, fname)
            save_path = os.path.join(save_dir, fname.replace(".jsonl", ".npy"))
            if os.path.exists(save_path):
                continue
            if open(fpath).read().strip() == "":
                continue
            texts = [json.loads(item) for item in open(fpath).read().strip().split('\n')]
            if "specter" in model_name.lower():
                texts = [model.tokenizer.sep_token.join([item["title"], item["content"]]) for item in texts]
            elif "contriever" in model_name.lower():
                texts = [". ".join([item["title"], item["content"]]).replace('..', '.').replace("?.", "?") for item in texts]
            elif "medcpt" in model_name.lower():
                texts = [[item["title"], item["content"]] for item in texts]
            else:
                texts = [concat(item["title"], item["content"]) for item in texts]
            embed_chunks = model.encode(texts, batch_size=2048, **kwarg)
            np.save(save_path, embed_chunks)
        embed_chunks = model.encode([""], **kwarg)
    return embed_chunks.shape[-1]

def construct_index(index_dir, model_name, h_dim=768, HNSW=False, M=32, use_gpu=False):
    use_gpu = False
    with open(os.path.join(index_dir, "metadatas.jsonl"), 'w') as f:
        f.write("")
    
    if HNSW:
        M = M
        if "specter" in model_name.lower():
            index = faiss.IndexHNSWFlat(h_dim, M)
        else:
            index = faiss.IndexHNSWFlat(h_dim, M)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        if "specter" in model_name.lower():
            index = faiss.IndexFlatL2(h_dim)
        else:
            index = faiss.IndexFlatIP(h_dim)
    # 收集所有嵌入向量和元数据
    all_embeddings = []
    all_metadata = []
    
    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(index_dir, "embedding")))):
        curr_embed = np.load(os.path.join(index_dir, "embedding", fname))
        # 确保向量是float32类型
        if curr_embed.dtype != np.float32:
            curr_embed = curr_embed.astype(np.float32)
        all_embeddings.append(curr_embed)
        # 为当前文件的所有嵌入创建元数据
        current_metadata = [json.dumps({'index': i, 'source': fname.replace(".npy", "")}) 
                           for i in range(len(curr_embed))]
        all_metadata.extend(current_metadata)

    # 如果启用了GPU
    if use_gpu and faiss.get_num_gpus() > 0:
        # 配置GPU选项
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True  # 使用半精度浮点数以节省内存
        co.shard = True       # 将索引分片到多个GPU
        gpu_index = faiss.index_cpu_to_all_gpus(index, co)
        
        # 将所有嵌入连接成一个大矩阵并添加到GPU索引
        if all_embeddings:
            all_embeddings_matrix = np.vstack(all_embeddings)
            gpu_index.add(all_embeddings_matrix)
        
        # 将GPU索引转换回CPU索引以便保存
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(index_dir, "embedding")))):
            curr_embed = np.load(os.path.join(index_dir, "embedding", fname))
            index.add(curr_embed)
            with open(os.path.join(index_dir, "metadatas.jsonl"), 'a+') as f:
                f.write("\n".join([json.dumps({'index': i, 'source': fname.replace(".npy", "")}) for i in range(len(curr_embed))]) + '\n')
    if HNSW==False:
        faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    else:
        faiss.write_index(index, os.path.join(index_dir, "faiss_hnsw.index"))
    return index


class Retriever: 

    def __init__(self, retriever_name="ncbi/MedCPT-Query-Encoder", corpus_name="textbooks", db_dir="./corpus", HNSW=False, **kwarg):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.use_gpu = kwarg.get("use_gpu", False)
        self.db_dir = db_dir
        if not os.path.exists(self.db_dir):
            print("no corpus found, please clone the corpus from Huggingface")
            raise Exception("no corpus found")
        
        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, "chunk")
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index", self.retriever_name.split('/')[-1].replace("Query-Encoder", "Article-Encoder"))

        if "bm25" in self.retriever_name.lower():
            from pyserini.search.lucene import LuceneSearcher
            self.metadatas = None
            self.embedding_function = None
            if os.path.exists(self.index_dir):
                self.index = LuceneSearcher(os.path.join(self.index_dir))
            else:
                os.system("python -m pyserini.index.lucene --collection JsonCollection --input {:s} --index {:s} --generator DefaultLuceneDocumentGenerator --threads 16".format(self.chunk_dir, self.index_dir))
                self.index = LuceneSearcher(os.path.join(self.index_dir))
        else:
            if HNSW == False and os.path.exists(os.path.join(self.index_dir, "faiss.index")):
                print("Loading faiss index from {}".format(self.index_dir))
                self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
                self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]
            elif HNSW == True and os.path.exists(os.path.join(self.index_dir, "faiss_hnsw.index")):
                print("Loading faiss index from {}".format(self.index_dir))
                self.index = faiss.read_index(os.path.join(self.index_dir, "faiss_hnsw.index"))
                self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]
            else:
                print("[In progress] Embedding the {:s} corpus with the {:s} retriever...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))
                h_dim = embed(chunk_dir=self.chunk_dir, index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), **kwarg)

                print("[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format(h_dim))
                self.index = construct_index(index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), h_dim=h_dim, HNSW=HNSW, use_gpu=self.use_gpu)
                print("[Finished] Corpus indexing finished!")
                self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]            
            if "contriever" in self.retriever_name.lower():
                self.embedding_function = SentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.embedding_function = CustomizeSentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_function.eval()

    def get_relevant_documents(self, question, k=32, id_only=False, **kwarg):
        assert type(question) == str
        question = [question]

        if "bm25" in self.retriever_name.lower():
            res_ = [[]]
            hits = self.index.search(question[0], k=k)
            res_[0].append(np.array([h.score for h in hits]))
            ids = [h.docid for h in hits]
            indices = [{"source": '_'.join(h.docid.split('_')[:-1]), "index": eval(h.docid.split('_')[-1])} for h in hits]
        else:
            with torch.no_grad():
                import time
                start_time = time.time()
                query_embed = self.embedding_function.encode(question, **kwarg)
                end_time = time.time()
                print(f"Embedding time: {end_time - start_time:.4f} seconds")
            res_ = self.index.search(query_embed, k=k)
            ids = ['_'.join([self.metadatas[i]["source"], str(self.metadatas[i]["index"])]) for i in res_[1][0]]
            indices = [self.metadatas[i] for i in res_[1][0]]

        scores = res_[0][0].tolist()
        
        if id_only:
            return [{"id":i} for i in ids], scores
        else:
            return self.idx2txt(indices), scores

    def idx2txt(self, indices): # return List of Dict of str
        '''
        Input: List of Dict( {"source": str, "index": int} )
        Output: List of str
        '''
        return [json.loads(open(os.path.join(self.chunk_dir, i["source"]+".jsonl")).read().strip().split('\n')[i["index"]]) for i in indices]

class RetrievalSystem:

    def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", HNSW=False, cache=False):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names
        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                self.retrievers[-1].append(Retriever(retriever, corpus, db_dir, HNSW=HNSW))
        self.cache = cache
        if self.cache:
            self.docExt = DocExtracter(cache=True, corpus_name=self.corpus_name, db_dir=db_dir)
        else:
            self.docExt = None
    
    def retrieve(self, question, k=32, rrf_k=100, id_only=False):
        '''
            Given questions, return the relevant snippets from the corpus
        '''
        assert type(question) == str

        output_id_only = id_only
        if self.cache:
            id_only = True

        texts = []
        scores = []

        if "RRF" in self.retriever_name:
            k_ = max(k * 2, 100)
        else:
            k_ = k
        for i in range(len(retriever_names[self.retriever_name])):
            texts.append([])
            scores.append([])
            for j in range(len(corpus_names[self.corpus_name])):
                t, s = self.retrievers[i][j].get_relevant_documents(question, k=k_, id_only=id_only)
                texts[-1].append(t)
                scores[-1].append(s)
        texts, scores = self.merge(texts, scores, k=k, rrf_k=rrf_k)
        if self.cache:
            texts = self.docExt.extract(texts)
        return texts, scores

    def merge(self, texts, scores, k=32, rrf_k=100):
        '''
            Merge the texts and scores from different retrievers
        '''
        RRF_dict = {}
        for i in range(len(retriever_names[self.retriever_name])):
            texts_all, scores_all = None, None
            for j in range(len(corpus_names[self.corpus_name])):
                if texts_all is None:
                    texts_all = texts[i][j]
                    scores_all = scores[i][j]
                else:
                    texts_all = texts_all + texts[i][j]
                    scores_all = scores_all + scores[i][j]
            if "specter" in retriever_names[self.retriever_name][i].lower():
                sorted_index = np.array(scores_all).argsort()
            else:
                sorted_index = np.array(scores_all).argsort()[::-1]
            texts[i] = [texts_all[i] for i in sorted_index]
            scores[i] = [scores_all[i] for i in sorted_index]
            for j, item in enumerate(texts[i]):
                if item["id"] in RRF_dict:
                    RRF_dict[item["id"]]["score"] += 1 / (rrf_k + j + 1)
                    RRF_dict[item["id"]]["count"] += 1
                else:
                    RRF_dict[item["id"]] = {
                        "id": item["id"],
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "score": 1 / (rrf_k + j + 1),
                        "count": 1
                        }
        RRF_list = sorted(RRF_dict.items(), key=lambda x: x[1]["score"], reverse=True)
        if len(texts) == 1:
            texts = texts[0][:k]
            scores = scores[0][:k]
        else:
            texts = [dict((key, item[1][key]) for key in ("id", "title", "content")) for item in RRF_list[:k]]
            scores = [item[1]["score"] for item in RRF_list[:k]]
        return texts, scores
    

class DocExtracter:
    
    def __init__(self, db_dir="./corpus", cache=False, corpus_name="MedCorp"):
        self.db_dir = db_dir
        self.cache = cache
        print("Initializing the document extracter...")
        for corpus in corpus_names[corpus_name]:
            if not os.path.exists(os.path.join(self.db_dir, corpus, "chunk")):
                print("no corpus chunk directory")
                raise Exception("no corpus chunk directory")
        if self.cache:
            if os.path.exists(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"]))):
                self.dict = json.load(open(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"]))))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(self.db_dir, corpus, "chunk")))):
                        if open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip() == "":
                            continue
                        for i, line in enumerate(open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip().split('\n')):
                            item = json.loads(line)
                            _ = item.pop("contents", None)
                            # assert item["id"] not in self.dict
                            self.dict[item["id"]] = item
                with open(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"])), 'w') as f:
                    json.dump(self.dict, f)
        else:
            if os.path.exists(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"]))):
                self.dict = json.load(open(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"]))))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(self.db_dir, corpus, "chunk")))):
                        if open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip() == "":
                            continue
                        for i, line in enumerate(open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip().split('\n')):
                            item = json.loads(line)
                            # assert item["id"] not in self.dict
                            self.dict[item["id"]] = {"fpath": os.path.join(corpus, "chunk", fname), "index": i}
                with open(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"])), 'w') as f:
                    json.dump(self.dict, f, indent=4)
        print("Initialization finished!")
    
    def extract(self, ids):
        if self.cache:
            output = []
            for i in ids:
                item = self.dict[i] if type(i) == str else self.dict[i["id"]]
                output.append(item)
        else:
            output = []
            for i in ids:
                item = self.dict[i] if type(i) == str else self.dict[i["id"]]
                output.append(json.loads(open(os.path.join(self.db_dir, item["fpath"])).read().strip().split('\n')[item["index"]]))
        return output

import argparse
def main():
    parser = argparse.ArgumentParser(description = "Creating index.")
    parser.add_argument('--corpus_name', type=str)
    parser.add_argument('--retriever_names', type=str)
    args = parser.parse_args()
    retriever_names = args.retriever_names
    corpus_names = args.corpus_name
    db_dir = "/nfsdata/yiao/medRAG"
    cache = db_dir
    retrieval_system = RetrievalSystem(retriever_names, corpus_names, db_dir, HNSW=True)
    retrieved_snippets, scores = retrieval_system.retrieve("These may include abdominal pain, achiness in the jaw or back, nausea, shortness of breath, or simply fatigue.", k=32)
    import pdb; pdb.set_trace()
    print(retrieved_snippets)

if __name__ == "__main__":
    main()
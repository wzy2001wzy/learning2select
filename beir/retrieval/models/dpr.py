from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from typing import Union, List, Dict, Tuple
from tqdm.autonotebook import trange
import torch


class DPR:
    def __init__(self, model_path: Union[str, Tuple] = None, **kwargs):
        # Query tokenizer and model
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_path[0])
        self.q_model = DPRQuestionEncoder.from_pretrained(model_path[0])
        self.q_model.cuda()
        self.q_model.eval()

        # Context tokenizer and model
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path[1])
        self.ctx_model = DPRContextEncoder.from_pretrained(model_path[1])
        self.ctx_model.cuda()
        self.ctx_model.eval()

    # def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> torch.Tensor:
    #     query_embeddings = []
    #     with torch.no_grad():
    #         for start_idx in trange(0, len(queries), batch_size):
    #             encoded = self.q_tokenizer(queries[start_idx:start_idx + batch_size], truncation=True, padding=True,
    #                                        max_length=512, return_tensors='pt')
    #             model_out = self.q_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
    #             query_embeddings += model_out.pooler_output
    #
    #     return torch.stack(query_embeddings)
    #
    # def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> torch.Tensor:
    #
    #     corpus_embeddings = []
    #     with torch.no_grad():
    #         for start_idx in trange(0, len(corpus), batch_size):
    #             titles = [row['title'] for row in corpus[start_idx:start_idx + batch_size]]
    #             texts = [row['text'] for row in corpus[start_idx:start_idx + batch_size]]
    #             encoded = self.ctx_tokenizer(titles, texts, truncation='longest_first', max_length=512, padding=True,
    #                                          return_tensors='pt')
    #             model_out = self.ctx_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
    #             corpus_embeddings += model_out.pooler_output.detach()
    #
    #     return torch.stack(corpus_embeddings)


def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> torch.Tensor:
    num_queries = len(queries)
    # 预分配CPU tensor
    query_embeddings = torch.zeros(num_queries, self.embedding_dim, dtype=torch.float32)

    with torch.no_grad():
        for start_idx in trange(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            batch_queries = queries[start_idx:end_idx]

            encoded = self.q_tokenizer(
                batch_queries,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )

            model_out = self.q_model(
                encoded['input_ids'].cuda(),
                attention_mask=encoded['attention_mask'].cuda()
            )

            # 立即转移到CPU并存储
            query_embeddings[start_idx:end_idx] = model_out.pooler_output.cpu()

            # 清理GPU内存
            del model_out, encoded
            torch.cuda.empty_cache()

    return query_embeddings


def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> torch.Tensor:
    num_docs = len(corpus)
    # 预分配CPU tensor
    corpus_embeddings = torch.zeros(num_docs, self.embedding_dim, dtype=torch.float32)

    with torch.no_grad():
        for start_idx in trange(0, num_docs, batch_size):
            end_idx = min(start_idx + batch_size, num_docs)
            batch_corpus = corpus[start_idx:end_idx]

            titles = [row['title'] for row in batch_corpus]
            texts = [row['text'] for row in batch_corpus]

            encoded = self.ctx_tokenizer(
                titles, texts,
                truncation='longest_first',
                max_length=512,
                padding=True,
                return_tensors='pt'
            )

            model_out = self.ctx_model(
                encoded['input_ids'].cuda(),
                attention_mask=encoded['attention_mask'].cuda()
            )

            # 立即转移到CPU并存储
            corpus_embeddings[start_idx:end_idx] = model_out.pooler_output.cpu()

            # 清理GPU内存
            del model_out, encoded
            torch.cuda.empty_cache()

    return corpus_embeddings

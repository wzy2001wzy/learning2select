import os

import numpy as np
import torch.multiprocessing as mp
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel, AutoConfig
import logging

from typing import Union, List, Dict, Tuple
from tqdm.autonotebook import trange
import torch
import queue

logger = logging.getLogger(__name__)
from peft import PeftModel
from biMis import MistralBiForMNTP


class FMIS:
    def __init__(self, model_name, checkpoint_path, option=1, debug=False, template=None, pooling='last',
                 max_tokens=1000, bi_att=False, dtype=torch.bfloat16):

        # Query tokenizer and model
        if bi_att is False:
            model_class = AutoModelForCausalLM
        else:
            assert 'mistral' in model_name or 'Mistral' in model_name
            model_class = MistralBiForMNTP

        if 'mntp' in model_name:
            self.q_tokenizer = AutoTokenizer.from_pretrained('/gs/home/by2206029/wuzy/Mistral-7B-Instruct-v0.2')
        else:
            self.q_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if checkpoint_path is not None:
            if 'mntp' in model_name:
                tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")
                config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                                                    trust_remote_code=True)
                model = AutoModel.from_pretrained(
                    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                    trust_remote_code=True,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                )
                model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")
                model = model.merge_and_unload()
                self.q_model = PeftModel.from_pretrained(model,
                                                         "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised")
            else:
                self.q_model = PeftModel.from_pretrained(
                    model_class.from_pretrained(model_name,
                                                # device_map='auto',
                                                torch_dtype=dtype,
                                                output_hidden_states=True,
                                                trust_remote_code=True).cuda().eval(),
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                )
        else:
            self.q_model = model_class.from_pretrained(model_name,
                                                       # device_map='auto',
                                                       torch_dtype=dtype,
                                                       output_hidden_states=True,
                                                       trust_remote_code=True).cuda().eval()

        print("load model success!!!!!!!")

        # Context tokenizer and model
        self.ctx_tokenizer = self.q_tokenizer
        self.ctx_model = self.q_model

        self.q_tokenizer.pad_token_id = 0
        self.ctx_tokenizer.pad_token_id = 0
        self.q_tokenizer.padding_side = "left"
        self.ctx_tokenizer.padding_side = "left"

        self.option = option
        self.debug = debug
        self.template = template
        self.pooling = pooling
        self.max_tokens = max_tokens
        # print(model_name)
        # print('pooling', self.pooling)
        # print('template', self.template)
        # print('max_tokens', self.max_tokens)
        # print('pooling', self.pooling)
        # print('bi_att:', bi_att)
        # print('dtype:', dtype)

    def encode(self, sentences: List[str], template="*sent_0*", batch_size=8, max_tokens=1000
               ) -> torch.Tensor:
        sublists = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        if self.debug is True:
            sublists = sublists[:10]
        assert max_tokens <= 1000
        # max_tokens = min(max_tokens, 1000)
        all_embeddings = None
        if self.template is not None:
            template = self.template
        # print("###########", self.pooling, self.template)
        for sublist in tqdm(sublists):
            tokenized_sentences = [self.q_tokenizer.tokenize(sentence) for sentence in sublist]
            first_max_tokens_batch = [tokens[:max_tokens] for tokens in tokenized_sentences]
            reconstructed_texts = [self.q_tokenizer.convert_tokens_to_string(tokens) for tokens in
                                   first_max_tokens_batch]
            # print(reconstructed_texts)

            inputs = self.q_tokenizer(
                [template.replace('*sent_0*', i).replace('_', ' ').strip() for i in reconstructed_texts],
                padding=True, return_tensors="pt")
            inputs = {k: v.to(self.q_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.q_model(**inputs, output_hidden_states=True,
                                       return_dict=True).hidden_states[-1]
                if self.pooling == 'last':
                    embeddings = outputs[:, -1, :].detach()
                elif self.pooling == 'avg':
                    mask = inputs['input_ids'] != 0
                    # 利用掩码获取有效的输出，扩展mask的形状以匹配outputs的最后一个维度
                    # mask形状为 [batch_size, seq_len]，扩展后形状为 [batch_size, seq_len, 1]
                    masked_outputs = outputs * mask.unsqueeze(-1).float()
                    # 计算每个batch中有效output的个数，形状为 [batch_size, 1]
                    valid_outputs_count = mask.sum(dim=1, keepdim=True).float()
                    # 避免除以零的情况，将0替换为1（这种情况下输出可以设为零向量，因为没有有效值）
                    valid_outputs_count = valid_outputs_count.masked_fill(valid_outputs_count == 0, 1)
                    # 计算有效输出的平均值，形状为 [batch_size, hid_dim]
                    embeddings = masked_outputs.sum(dim=1) / valid_outputs_count
                elif self.pooling == 'weight_avg':
                    mask = inputs['input_ids'] != 0
                    # 生成每个batch的有效位置的索引
                    indices = torch.cumsum(mask, dim=1) * mask
                    # 计算每个batch中有效output的个数
                    valid_outputs_count = mask.sum(dim=1)  # 形状为 [batch_size]
                    # 根据有效位置的数量计算每个有效位置的权重
                    weights = indices.float() / (valid_outputs_count * (valid_outputs_count + 1) / 2).unsqueeze(1)
                    # 计算加权平均
                    # 应用掩码和权重
                    weighted_outputs = outputs[0] * weights.unsqueeze(-1)  # 扩展权重维度以匹配outputs
                    # 计算加权和
                    embeddings = weighted_outputs.sum(dim=1)

                del outputs

                embeddings = embeddings.cpu()
                if all_embeddings is None:
                    all_embeddings = embeddings
                else:
                    all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)

            del inputs
        torch.cuda.empty_cache()
        return all_embeddings

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> torch.Tensor:
        if self.template is not None:
            template = self.template
        else:
            template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
        queries_emb = self.encode(queries, batch_size=batch_size, template=template,
                                  max_tokens=self.max_tokens)
        # np.save(self.save_q, queries_emb.detach().cpu().numpy())
        return queries_emb

    def encode_corpus(self, corpus, batch_size: int = 8, **kwargs) -> torch.Tensor:

        assert self.option == 1
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][
                    i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + doc["text"]).strip() if "title" in doc else doc["text"].strip() for
                         doc in corpus]

        new_list = sentences
        corpus_emb = None

        if self.option == 1 or self.option == 0:
            if self.template is not None:
                template = self.template
            else:
                template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
            corpus_emb = self.encode(new_list, batch_size=batch_size,
                                     template=template,
                                     max_tokens=self.max_tokens)

        return corpus_emb

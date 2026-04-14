import os

import numpy as np
import torch.multiprocessing as mp
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import logging

from typing import Union, List, Dict, Tuple
from tqdm.autonotebook import trange
import torch
import queue

logger = logging.getLogger(__name__)
from peft import PeftModel


class FLM:
    def __init__(self, model_name, checkpoint_path, option=1, debug=False, template=None, template_gen=None, jw=1,
                 wd=768, cs=-1,
                 **kwargs):

        # Query tokenizer and model

        self.q_tokenizer = LlamaTokenizer.from_pretrained(model_name)

        if checkpoint_path is not None:
            self.q_model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(model_name,
                                                     # device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     output_hidden_states=True,
                                                     trust_remote_code=True).cuda().eval(),
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            )
        else:
            self.q_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                                # device_map='auto',
                                                                torch_dtype=torch.float16,
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
        self.template_gen = template_gen
        self.jw = jw
        self.wd = wd
        self.cs = cs


    def encode(self, sentences: List[str], template="*sent_0*", batch_size=8, max_tokens=2000
               ) -> torch.Tensor:
        sublists = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        if self.debug is True:
            sublists = sublists[:10]
        max_tokens = min(max_tokens, 1000)
        all_embeddings = None
        if self.template is not None:
            template = self.template
        if self.jw < 0:
            template = "*sent_0*"
        for sublist in tqdm(sublists):
            inputs = self.q_tokenizer([template.replace('*sent_0*', i).replace('_', ' ').strip() for i in sublist],
                                      padding=True, return_tensors="pt", max_length=max_tokens, truncation=True)
            inputs = {k: v.to(self.q_model.device) for k, v in inputs.items()}
            with torch.no_grad():

                embeddings = self.q_model(**inputs, output_hidden_states=True,
                                          return_dict=True).hidden_states[self.cs][:, -1, :].detach()

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
                                  max_tokens=2000)
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
                                     max_tokens=2000)

        return corpus_emb

import os

import numpy as np
import torch.multiprocessing as mp
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from typing import Union, List, Dict, Tuple
from tqdm.autonotebook import trange
import torch
import queue

logger = logging.getLogger(__name__)
from peft import PeftModel


class FOPT:
    def __init__(self, model_name, checkpoint_path, option=1, debug=False, template=None,
                 **kwargs):

        # Query tokenizer and model

        self.q_tokenizer = AutoTokenizer.from_pretrained(model_name)

        if checkpoint_path is not None:
            self.q_model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(model_name,
                                                     # device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     output_hidden_states=True,
                                                     trust_remote_code=True),
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            )
        else:
            self.q_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                                # device_map='auto',
                                                                torch_dtype=torch.float16,
                                                                output_hidden_states=True,
                                                                trust_remote_code=True)

        # Context tokenizer and model
        self.ctx_tokenizer = self.q_tokenizer
        self.ctx_model = self.q_model
        self.ctx_model.cuda().eval()

        self.q_tokenizer.pad_token_id = 0
        self.ctx_tokenizer.pad_token_id = 0
        self.q_tokenizer.padding_side = "left"
        self.ctx_tokenizer.padding_side = "left"

        self.option = option
        self.debug = debug
        self.template = template

    def encode(self, sentences: List[str], template="*sent_0*", batch_size=8, max_tokens=2000
               ) -> torch.Tensor:
        sublists = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        if self.debug is True:
            sublists = sublists[:10]
        max_tokens = min(max_tokens, 1000)
        all_embeddings = None
        if self.template is not None:
            template = self.template

        for sublist in tqdm(sublists):
            inputs = self.q_tokenizer([template.replace('*sent_0*', i).replace('_', ' ').strip() for i in sublist],
                                      padding=True, return_tensors="pt", max_length=max_tokens, truncation=True)
            inputs = {k: v.to(self.q_model.device) for k, v in inputs.items()}
            with torch.no_grad():

                embeddings = self.q_model(**inputs, output_hidden_states=True,
                                          return_dict=True).hidden_states[-1][:, -1, :].detach()

                embeddings = embeddings.cpu()
                if all_embeddings is None:
                    all_embeddings = embeddings
                else:
                    all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)

            del inputs
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

        if self.option == 0:
            if type(corpus) is dict:
                sentences = [
                    (corpus["title"][i]).strip() if "title" in corpus else corpus["text"][
                        i].strip() for i in range(len(corpus['text']))]
            else:
                sentences = [(doc["title"]).strip() if "title" in doc else doc["text"].strip() for
                             doc in corpus]
        else:
            if type(corpus) is dict:
                sentences = [
                    (corpus["title"][i] + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][
                        i].strip() for i in range(len(corpus['text']))]
            else:
                sentences = [(doc["title"] + doc["text"]).strip() if "title" in doc else doc["text"].strip() for
                             doc in corpus]

        new_list = sentences
        corpus_emb = None

        if self.debug is True:
            new_list = new_list[5000:5005]
        if self.option == 1 or self.option == 0:
            if self.template is not None:
                template = self.template
            else:
                template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
            corpus_emb = self.encode(new_list, batch_size=batch_size,
                                     template=template,
                                     max_tokens=2000)


        return corpus_emb

        # def encode_corpus_parallel(self, corpus: Union[List[Dict[str, str]], Dataset], pool: Dict[str, str],
        #                            batch_size: int = 8, chunk_id: int = None, max_tokens=1000, template="*sent_0*",
        #                            option=1, **kwargs):
        #     if self.option == 0:
        #         if type(corpus) is dict:
        #             sentences = [
        #                 (corpus["title"][i]).strip() if "title" in corpus else corpus["text"][
        #                     i].strip() for i in range(len(corpus['text']))]
        #         else:
        #             sentences = [(doc["title"]).strip() if "title" in doc else doc["text"].strip() for
        #                          doc in corpus]
        #     else:
        #         if type(corpus) is dict:
        #             sentences = [
        #                 (corpus["title"][i] + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][
        #                     i].strip() for i in range(len(corpus['text']))]
        #         else:
        #             sentences = [(doc["title"] + doc["text"]).strip() if "title" in doc else doc["text"].strip() for
        #                          doc in corpus]
        #     sublists = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        #     if self.debug is True:
        #         sublists = sublists[:10]
        #     if self.option == 1 or self.option == 0:
        #         template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
        #         for sublist in sublists:
        #
        #             tokenized_texts = [
        #                 self.ctx_tokenizer.tokenize(template.replace('*sent_0*', i).replace('_', ' ').strip())
        #                 [:max_tokens] for i in sublist]
        #
        #             # 将截断后的 token lists 转换回文本
        #             truncated_texts = [self.ctx_tokenizer.convert_tokens_to_string(tokens) for tokens in tokenized_texts]
        #             inputs = self.ctx_tokenizer(truncated_texts, padding=True, return_tensors="pt")
        #
        #             if chunk_id is not None and chunk_id >= len(pool['processes']):
        #                 output_queue = pool['output']
        #                 output_queue.get()
        #
        #             input_queue = pool['input']
        #             input_queue.put([chunk_id, batch_size, inputs])
        #
        #     if self.option == 2 :
        #         template = 'The_summary_of_the_text_:_"*sent_0*"_is:'
        #         for sublist in sublists:
        #             tokenized_texts = [
        #                 self.ctx_tokenizer.tokenize(template.replace('*sent_0*', i).replace('_', ' ').strip())
        #                 [:max_tokens] for i in sublist]
        #
        #             # 将截断后的 token lists 转换回文本
        #             truncated_texts = [self.ctx_tokenizer.convert_tokens_to_string(tokens) for tokens in tokenized_texts]
        #             if chunk_id is not None and chunk_id >= len(pool['processes']):
        #                 output_queue = pool['output']
        #                 output_queue.get()
        #
        #             input_queue = pool['input']
        #             input_queue.put([chunk_id, batch_size, truncated_texts])

        # def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        #     logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))
        #
        #     ctx = mp.get_context('spawn')
        #     input_queue = ctx.Queue()
        #     output_queue = ctx.Queue()
        #     processes = []
        #
        #     for process_id, device_name in enumerate(target_devices):
        #         p = ctx.Process(target=SentenceTransformer._encode_multi_process_worker,
        #                         args=(process_id, device_name, self.ctx_model, input_queue, output_queue), daemon=True)
        #         p.start()
        #         processes.append(p)
        #
        #     return {'input': input_queue, 'output': output_queue, 'processes': processes}
        #
        # def stop_multi_process_pool(self, pool: Dict[str, object]):
        #     output_queue = pool['output']
        #     [output_queue.get() for _ in range(len(pool['processes']))]
        #     for p in pool['processes']:
        #         p.terminate()
        #
        #     for p in pool['processes']:
        #         p.join()
        #         p.close()
        #
        #     pool['input'].close()
        #     pool['output'].close()

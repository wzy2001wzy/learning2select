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


class OPT:
    def __init__(self, model_name, option=1, debug=False, template=None, template_gen=None, jw=1, wd=768, cs=-1,
                 **kwargs):

        # Query tokenizer and model
        self.q_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.q_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Context tokenizer and model
        self.ctx_tokenizer = self.q_tokenizer
        self.ctx_model = self.q_model
        self.ctx_model.cuda()

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
        max_tokens = min(max_tokens, 1900)
        all_embeddings = None
        if self.template is not None:
            template = self.template
        # if self.jw < 0:
        #     template = "*sent_0*"

        for sublist in sublists:
            new_texts = [template.replace('*sent_0*', i[:max_tokens]).replace('_', ' ').strip() for i in sublist]

            inputs = self.q_tokenizer(new_texts, padding=True, return_tensors="pt", truncation=True,
                                      max_length=2000)

            inputs = {k: v.to(self.q_model.device) for k, v in inputs.items()}
            with torch.no_grad():

                embeddings = self.q_model(**inputs, output_hidden_states=True,
                                          return_dict=True).hidden_states[self.cs][:, -1, :].detach()

                # embeddings = embeddings.cpu()
                if all_embeddings is None:
                    all_embeddings = embeddings.clone()
                else:
                    all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)

            del inputs
            del embeddings
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

    def gen(self, old_text: List[str], template_gen="*sent_0*", batch_size=8, max_tokens=2000,
            **kwargs) -> List[str]:
        for i, s in enumerate(old_text):
            if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
            s = s.replace('"', '\'')
            if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
            old_text[i] = s.strip()

        sublists = [old_text[i:i + batch_size] for i in range(0, len(old_text), batch_size)]
        if self.debug is True:
            sublists = sublists[:1]
        if self.template is not None:
            template_gen = self.template_gen

        generated_sentences = []
        sent_len = []

        for sublist in tqdm(sublists):
            tokenized_texts = [
                self.ctx_tokenizer.tokenize(template_gen.replace('*sent_0*', i).replace('_', ' ').strip())
                [:max_tokens] for i in sublist]

            # print(tokenized_texts)

            # 将截断后的 token lists 转换回文本
            truncated_texts = [self.ctx_tokenizer.convert_tokens_to_string(tokens) for tokens in
                               tokenized_texts]

            for t in truncated_texts:
                # print(t)
                # print("")
                sent_len.append(len(t))

            # print(truncated_texts)
            # 使用 tokenizer 将截断后的文本转换为模型的输入格式
            inputs = self.ctx_tokenizer(truncated_texts, padding=True, return_tensors="pt")

            inputs = {k: v.to(self.q_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = self.q_model.generate(**inputs, max_new_tokens=50)  # max_length 可以根据需要调整

                # 将生成的 token IDs 解码为文本
                for gen_id in generated_ids:
                    generated_sentences.append(self.ctx_tokenizer.decode(gen_id, skip_special_tokens=True))

            del inputs
            # torch.cuda.empty_cache()
        new_corpus = []

        for i in range(len(generated_sentences)):
            ss = generated_sentences[i][sent_len[i]:]
            if len(ss.split('::')) > 1:
                ss = ss.split('::')[1]
            else:
                ss = ss.split('::')[0]
            new_corpus.append(ss)

        # for i in range(5):
        #     print("#############")
        #     print(old_text[i])
        #     print("$$$$$$$$$$$")
        #     print(new_corpus[i])
        return new_corpus

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

        if self.option == 2 or self.option == 3:
            template_gen = ''
            if self.option == 2:
                template_gen = 'The_summary_of_the_text_:_"*sent_0*"_is_in_one_sentence::'
            if self.option == 3:
                template_gen = 'use_a_question_to_search_the_text_:_"*sent_0*"::'
            if self.template_gen is not None:
                template_gen = self.template_gen
            new_corpus = self.gen(new_list, template=template_gen, batch_size=batch_size, max_tokens=2000)
            # print(new_corpus)
            if self.template is not None:
                template = self.template
            else:
                template = 'This_sentence_:_"*sent_0*"_means_in_one_word:"'
            corpus_emb = self.encode(new_corpus, batch_size=batch_size,
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

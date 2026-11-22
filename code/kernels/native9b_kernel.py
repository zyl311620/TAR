"""Exact-permutation native9B rerun client; no API or response repair."""
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.error import HTTPError

from permutation_decoder import PermutationTokenVocabulary, PROTOCOL
from llm_ranking import parse_permutation


class Native9BClient:
    def __init__(self, model, tokenizer, folder):
        self.model,self.tokenizer,self.folder=model,tokenizer,Path(folder)
        eos=model.generation_config.eos_token_id
        self.vocabulary=PermutationTokenVocabulary(tokenizer,[eos] if isinstance(eos,int) else eos)
        self.last_was_cached=False

    def request(self,messages,namespace='main',max_tokens=8192):
        import torch
        count=len(json.loads(messages[-1]['content'].split('\n\nCandidates:\n',1)[1]))
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.tokenizer(text,return_tensors='pt');length=int(inputs['input_ids'].shape[1])
        if length+2048>32768:
            exc=HTTPError('native://Qwen3.5-9B',400,'context length',None,io.BytesIO(b'context length'))
            exc.paper_error_body='context length';raise exc
        key=hashlib.sha256(json.dumps({'messages':messages,'namespace':namespace,'protocol':PROTOCOL},
                                     ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        inputs={k:v.cuda() for k,v in inputs.items()};started=time.monotonic()
        with torch.inference_mode():
            output=self.model.generate(**inputs,do_sample=False,max_new_tokens=2048,
                prefix_allowed_tokens_fn=self.vocabulary.constraint(count,length))
        content=self.tokenizer.decode(output[0,length:],skip_special_tokens=True)
        raw={'model':'Qwen3.5-9B-native','choices':[{'message':{'content':content}}],
             'usage':{'prompt_tokens':length,'completion_tokens':int(output.shape[1]-length)}}
        from reproduce import save
        save(self.folder/f'{key}.json',{'request':messages,'response':raw,'decoder_protocol':PROTOCOL,
             'response_repair':False,'external_api_calls':0,'seconds':time.monotonic()-started})
        try:parse_permutation(content,count)
        except (ValueError,TypeError,KeyError) as exc:
            raise RuntimeError('Invalid constrained response retained; no retry or repair') from exc
        return raw,key

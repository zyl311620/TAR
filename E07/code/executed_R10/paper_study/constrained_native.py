"""Versioned local inference; original caches never supply constrained responses."""
import json,time
from urllib.error import HTTPError
import io
from .common import STUDY,identity,load,save,now
from .inference_hold import require_new_llm_inference
from .llm_ranking import parse_permutation
from .permutation_decoder import PermutationTokenVocabulary,PROTOCOL

REVISION=STUDY/'revisions/native_format_v3_20261106'

def request_identity(signature,messages,namespace):
    return identity({'model':signature,'messages':messages,'namespace':namespace,
        'thinking':False,'max_tokens':2048,'do_sample':False,'decoder_protocol':PROTOCOL})

class ConstrainedNativeClient:
    def __init__(self,client):
        self.client=client
        eos=client.network.generation_config.eos_token_id
        self.vocabulary=PermutationTokenVocabulary(client.tokenizer,[eos] if isinstance(eos,int) else eos)
        self.last_was_cached=False
        self.scope=None

    def request(self,messages,namespace='main',max_tokens=8192):
        if self.scope not in ('rev2rev-en','rev2rev-ja'):raise PermissionError('Only the two human-authorized pairs')
        key=request_identity(self.client.signature,messages,namespace)
        path=REVISION/'cache'/f'{key}.json'
        candidates=json.loads(messages[-1]['content'].split('\n\nCandidates:\n',1)[1])
        count=len(candidates)
        if [x['id'] for x in candidates]!=list(range(1,count+1)):raise ValueError('Candidate ordinal IDs changed')
        if path.exists():
            item=load(path)
            if (item['identity']!=key or item['decoder_protocol']!=PROTOCOL or item['request']!=messages
                    or item['model_signature']!=self.client.signature or item['external_api_calls']!=0):
                raise ValueError('Constrained request cache identity changed')
            parse_permutation(item['response']['choices'][0]['message']['content'],count)
            self.last_was_cached=True
            return item['response'],key
        require_new_llm_inference(STUDY,backend='native9b',experiment='E08',row_id='E08.010',task=self.scope)
        text=self.client.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.client.tokenizer(text,return_tensors='pt')
        prompt_length=int(inputs['input_ids'].shape[1])
        if prompt_length+2048>32768:
            exc=HTTPError('native://Qwen3.5-9B',400,'context length',None,io.BytesIO(b'context length'))
            exc.paper_error_body='context length';raise exc
        inputs={k:v.cuda() for k,v in inputs.items()}
        started=time.monotonic()
        with self.client.torch.inference_mode():
            output=self.client.network.generate(**inputs,do_sample=False,max_new_tokens=2048,
                prefix_allowed_tokens_fn=self.vocabulary.constraint(count,prompt_length))
        content=self.client.tokenizer.decode(output[0,prompt_length:],skip_special_tokens=True)
        elapsed=time.monotonic()-started
        raw={'model':self.client.model,'choices':[{'message':{'content':content}}],
            'usage':{'prompt_tokens':prompt_length,'completion_tokens':int(output.shape[1]-prompt_length)}}
        item={'at':now(),'identity':key,'request':messages,'response':raw,'namespace':namespace,
            'model_signature':self.client.signature,'decoder_protocol':PROTOCOL,'documents':count,
            'generation_seconds':elapsed,'external_api_calls':0,'response_repair':False,
            'decoding':{'thinking':False,'do_sample':False,'max_new_tokens':2048}}
        save(path,item)
        try:parse_permutation(content,count)
        except (ValueError,KeyError,TypeError) as exc:
            # A decoder bug must stop, not disappear into recursive recovery.
            raise RuntimeError('Constrained format failure retained; no automatic fallback') from exc
        self.last_was_cached=False
        print('CONSTRAINED_NATIVE_REQUEST '+json.dumps({'at':now(),'task':self.scope,'documents':count,
            'seconds':round(elapsed,3),'identity':key,'format_valid':True}),flush=True)
        return raw,key

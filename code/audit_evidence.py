"""Offline checks of release scope, request identities and R10 checkpoints."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reproduce import load,offline,save,sha


def identity(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                    separators=(',',':')).encode('utf-8')).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests-dir',type=Path,required=True,
                        help='Authorized local raw request evidence; not publicly distributed')
    args=parser.parse_args()
    if not args.requests_dir.is_dir():
        parser.error('--requests-dir must point to existing, authorized local evidence')
    offline()
    if [p.name for p in sorted(ROOT.glob('E[0-9][0-9]'))]!=[f'E{i:02d}' for i in range(1,11)]:
        raise ValueError('Release scope differs from manuscript E01--E10')
    expected=load(ROOT/'release_manifest.json')
    raw_count=0;new_count=0;models={};new_cache={}
    forbidden=[]
    key_pattern=re.compile(r'sk-[A-Za-z0-9_.-]{18,}')
    for path in args.requests_dir.glob('*.json.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as stream:text=stream.read()
        if key_pattern.search(text):forbidden.append(path.name)
        try:
            item=json.loads(text)
        except ValueError as exc:
            raise ValueError('Invalid JSON response evidence: '+path.name) from exc
        raw_count+=1
        if item.get('decoder_protocol')=='native-exact-permutation-json-v1-20261106':
            new_count+=1
            if item.get('external_api_calls')!=0 or item.get('response_repair') is not False:
                raise ValueError('New R10 response identity or no-repair evidence differs')
            if item['decoding']!={'thinking':False,'do_sample':False,'max_new_tokens':2048}:
                raise ValueError('New R10 decoding payload changed')
            signature=identity({'model':item['model_signature'],'messages':item['request'],
                'namespace':item['namespace'],'thinking':False,'max_tokens':2048,'do_sample':False,
                'decoder_protocol':item['decoder_protocol']})
            if signature!=item['identity'] or path.name!=signature+'.json.gz':
                raise ValueError('Constrained request cache key does not reproduce')
            candidates=json.loads(item['request'][-1]['content'].split('\n\nCandidates:\n',1)[1])
            if [d['id'] for d in candidates]!=list(range(1,len(candidates)+1)):
                raise ValueError('R10 candidate ordinal identity changed')
            content=item['response']['choices'][0]['message']['content']
            value=json.loads(content)
            ranking=value['ranking']
            if any(type(i) is not int for i in ranking) or sorted(ranking)!=list(range(1,len(candidates)+1)):
                raise ValueError('New R10 response is not an exact complete permutation')
            if item['response']['model']!='Qwen3.5-9B-native':
                raise ValueError('Unexpected R10 response model')
            new_cache[signature]={'model_signature':item['model_signature'],'documents':len(candidates)}
        model=item.get('response',{}).get('model','not-recorded')
        models[model]=models.get(model,0)+1
    if forbidden:raise ValueError('Credential-like values found in request evidence files: '+repr(forbidden))
    # Do not print a suspicious string. Only the affected path can be reported.
    for base in ('code','E03/code','E08/code'):
        for path in (ROOT/base).rglob('*.py'):
            if key_pattern.search(path.read_text(encoding='utf-8-sig')):
                # The vendored evaluator initializer may contain an explicit non-secret placeholder.
                value=path.read_text(encoding='utf-8-sig').replace('sk-recare-no-llm-stub','')
                if key_pattern.search(value):raise ValueError('Credential-like value in code: '+str(path.relative_to(ROOT)))
    checked={};used=set()
    for entry in load(ROOT/'E07/results.json'):
        if entry.get('decoder_protocol')!='native-exact-permutation-json-v1-20261106':continue
        task=entry['task'];run=ROOT/entry['rankings'][0]['path']
        final={r['qid']:r['ranked'] for r in (json.loads(line) for line in run.read_text(encoding='utf-8').splitlines())}
        source_sha=entry['original_provenance']['candidate_source_sha256']
        source=ROOT/f'_shared/rankings/{source_sha}.jsonl'
        if sha(source)!=source_sha:raise ValueError('R10 C200 input ranking checksum differs')
        candidates={r['qid']:r['ranked'] for r in
                    (json.loads(line) for line in source.read_text(encoding='utf-8').splitlines())}
        paths=sorted((ROOT/f'E07/evidence/R10/{task}/queries').glob('*.json'))
        if len(paths)!=len(final):raise ValueError('R10 full checkpoint coverage differs')
        seen=set();calls=0
        for path in paths:
            item=load(path);q=item['qid'];seen.add(q)
            pool,ranked=item['candidate_ids'],item['ranked']
            if (pool!=candidates[q] or item['model_signature']!=entry['original_provenance']['model_signature'] or
                    item['decoder_protocol']!=entry['decoder_protocol'] or
                    item['runtime_id']!=entry['original_provenance']['runtime_id'] or
                    final[q]!=ranked or len(ranked)!=len(set(ranked)) or
                    set(ranked[:100])!=set(pool[:100]) or ranked[100:]!=pool[100:200]):
                raise ValueError('R10 checkpoint/final ranking or frozen-pool invariant differs')
            for call in item['calls']:
                key=call['request_identity'];proof=new_cache[key]
                if proof['model_signature']!=item['model_signature'] or proof['documents']!=call['documents']:
                    raise ValueError('R10 checkpoint/raw response identity differs')
                used.add(key);calls+=1
        if seen!=set(final):raise ValueError('Duplicate/missing R10 checkpoint qids')
        checked[task]={'queries':len(paths),'referenced_windows':calls,'ranked_sha256':sha(run)}
    if set(new_cache)!=used or new_count!=8520:
        raise ValueError('R10 request count or checkpoint request coverage differs')
    if raw_count!=expected['raw_request_files']:raise ValueError('Request file count differs')
    if load(ROOT/'audit/missing_request_evidence.json'):
        raise ValueError('Some query checkpoint request references lack raw evidence')
    result={'scope':'manuscript E01--E10','raw_request_files':raw_count,'response_model_counts':models,
            'R10_requests':new_count,'R10_checkpoint_checks':checked,
            'credentials_found':False,'external_requests_this_audit':0,
            'note':'Checks identity, complete permutations and fixed pools; not legal relevance or reasoning correctness.'}
    save(ROOT/'outputs/evidence_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

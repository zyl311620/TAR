"""Manual server-side reruns of the native stages used in the manuscript.

These commands never contact a model API. They write a NEW work directory and
do not replace packaged predictions. See README.md for the frozen conditions.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/vendor'))


def records(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('dense', 'rrf', 'rerank', 'bm25', 'rm3', 'native9b'))
    parser.add_argument('--data', type=Path, help='Released ReCaRe directory, including corpora')
    parser.add_argument('--model', type=Path, help='Already downloaded local model directory')
    parser.add_argument('--encoder', choices=('bge-m3', 'me5-base', 'jina-v3', 'jina-v4',
                                             'qwen3-embedding-8b', 'jina-v5-small'), default='jina-v4')
    parser.add_argument('--reranker', default='qwen3-reranker-8b')
    parser.add_argument('--task', choices=('rat2rev-en','rev2rev-en','rat2rev-ja','rev2rev-ja'), required=True)
    parser.add_argument('--split', choices=('test','validation'), default='test')
    parser.add_argument('--queries', type=Path, help='Optional _id/text query JSONL (for saved views)')
    parser.add_argument('--input', type=Path, nargs='+', help='Ranking sources; first RRF branch is BM25')
    parser.add_argument('--weights', type=float, nargs='+')
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--source-depth', type=int, default=500)
    parser.add_argument('--candidate-depth', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--k1', type=float, default=.9)
    parser.add_argument('--b', type=float, default=.4)
    parser.add_argument('--jars', type=Path, help='Folder with the three Lucene9.12.1 jars; BM25 only')
    parser.add_argument('--rm3-config', type=Path, help='Frozen rm3_selected.TASK.json; RM3 only')
    parser.add_argument('--work', type=Path, required=True, help='New server work directory')
    args = parser.parse_args()
    if os.name == 'nt':
        parser.error('Native computation is server-only. Use reproduce.py on Windows.')
    work = args.work.resolve()
    if work == ROOT or ROOT in work.parents or work.exists():
        parser.error('--work must be a new directory outside the release')
    # Explicitly block all network connections. Model files must already exist.
    sys.path.insert(0, str(ROOT))
    from reproduce import offline, save, sha
    offline()
    work.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.environ['TAR_NATIVE_WORK'] = str(work)
    local_data = args.data.resolve() if args.data else Path(os.environ.get('TAR_DATA_DIR', ROOT / 'data'))
    os.environ['TAR_DATA'] = str(local_data)
    sys.path.insert(0, str(ROOT / 'code/kernels'))
    from fusion_kernel import rrf
    weights=None
    if args.model is not None:
        from dense_kernel import model_identity
        weights=model_identity(args.model.resolve())
    output = []
    task, split = args.task, args.split
    query_file = args.queries or local_data/f'queries/{task}/{split}.jsonl'
    if args.stage == 'rrf':
        if not args.input or not args.weights or len(args.input) != len(args.weights):
            parser.error('rrf needs --input and matching --weights')
        branches = [{r['qid']:r['ranked'][:args.source_depth] for r in records(p)} for p in args.input]
        if any(set(b) != set(branches[0]) for b in branches):
            raise ValueError('RRF source query sets differ')
        output = [{'qid':q,'ranked':rrf([b[q] for b in branches],args.weights,k=args.k,
                                      count=args.candidate_depth)} for q in branches[0]]
    elif args.stage in ('bm25','rm3'):
        if args.data is None or args.jars is None:
            parser.error('bm25 needs --data and --jars')
        import shutil
        jar_dir = work/'framework/vendor_ja/lucene';jar_dir.mkdir(parents=True)
        for p in args.jars.glob('*.jar'):shutil.copy2(p,jar_dir/p.name)
        target=work/f'inputs/queries/{task}/{split}.jsonl';target.parent.mkdir(parents=True)
        shutil.copy2(query_file,target)
        from bm25_kernel import lucene, build_index, search
        cls,jars=lucene();analyzer,folder,sig=build_index(cls,task.rsplit('-',1)[1],jars)
        if args.stage=='rm3':
            if args.rm3_config is None:parser.error('rm3 needs --rm3-config')
            from rm3_kernel import run_setting
            from reproduce import load
            config=load(args.rm3_config)['selected']
            directory=cls('org.apache.lucene.store.FSDirectory').open(cls('java.nio.file.Paths').get(str(folder/'index')))
            reader=cls('org.apache.lucene.index.DirectoryReader').open(directory)
            searcher=cls('org.apache.lucene.search.IndexSearcher')(reader)
            searcher.setSimilarity(cls('org.apache.lucene.search.similarities.BM25Similarity')(args.k1,args.b))
            try:folder,_=run_setting(cls,analyzer,searcher,reader,sig,task,split,config,{})
            finally:reader.close();directory.close()
        else:folder,_=search(cls,analyzer,folder,task,split,args.k1,args.b,sig)
        output=records(folder/'ranked.jsonl')
    else:
        if args.data is None or args.model is None:
            parser.error('This stage needs --data and --model')
        import numpy as np
        corpus=records(args.data/f'corpus-{task.rsplit("-",1)[1]}/corpus.jsonl')
        ids=[r['_id'] for r in corpus];texts={r['_id']:r['text'] for r in corpus}
        queries={r['_id']:r['text'] for r in records(query_file)}
        if args.stage == 'dense':
            from dense_kernel import Encoder,SPECS
            spec=SPECS[args.encoder];encoder=Encoder(spec,args.model)
            bank=None
            # Save vectors on the server only. No vectors are included in the release.
            for start in range(0,len(corpus),args.batch_size):
                values=encoder.encode([r['text'] for r in corpus[start:start+args.batch_size]])
                if bank is None:
                    bank=np.lib.format.open_memmap(work/'corpus.npy',mode='w+',dtype=np.float32,
                                                   shape=(len(corpus),values.shape[1]))
                bank[start:start+len(values)]=values
            bank.flush()
            from dense_rank_kernel import rank_vectors
            qids=list(queries)
            for start in range(0,len(qids),args.batch_size):
                batch=qids[start:start+args.batch_size]
                ranked=rank_vectors(encoder.encode([queries[q] for q in batch],query=True),bank,ids,batch,task)
                output.extend({'qid':q,'ranked':r} for q,r in zip(batch,ranked))
        else:
            if not args.input or len(args.input)!=1:
                parser.error('rerank/native9b needs one --input C200 ranking')
            pools=records(args.input[0])
            if {r['qid'] for r in pools}!=set(queries):raise ValueError('Pool/query coverage mismatch')
            if args.stage == 'rerank':
                from recare_baselines import reranker
                from cross_kernel import jina_load,jina_rank
                scorer=(jina_load(args.model) if args.reranker=='jina-reranker-v3.5' else
                        reranker._load(replace(reranker.MODELS[args.reranker],hf_id=str(args.model))))
                if args.reranker.startswith('qwen3-reranker'):
                    from rerank_optimization import optimize
                    optimize(scorer,args.reranker)
                for item in pools:
                    pool=item['ranked'][:args.candidate_depth];query=queries[item['qid']]
                    if args.reranker=='jina-reranker-v3.5':
                        order,_=jina_rank(scorer,query,[texts[d] for d in pool]);ranked=[pool[i] for i in order]
                    else:
                        values=scorer.score_pairs([(query,texts[d]) for d in pool])
                        ranked=[d for _,d in sorted(zip(values,pool),key=lambda x:(x[0],x[1]),reverse=True)]
                    output.append({'qid':item['qid'],'ranked':ranked[:200]})
            else:
                if task not in ('rev2rev-en','rev2rev-ja') or args.model.name!='Qwen3.5-9B':
                    parser.error('Only the two R10 Rev Qwen3.5-9B constrained conditions are supported')
                import torch
                from transformers import AutoTokenizer,AutoModelForImageTextToText
                from native9b_kernel import Native9BClient
                from adaptive_kernel import rerank_query
                tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
                model=AutoModelForImageTextToText.from_pretrained(args.model,torch_dtype=torch.bfloat16,
                        attn_implementation='sdpa',local_files_only=True).eval().cuda()
                client=Native9BClient(model,tokenizer,work/'requests')
                for item in pools:
                    singleton=[]
                    ranked,calls=rerank_query(client,task,queries[item['qid']],item['ranked'],texts,
                        criterion='T2',native_singleton_identity=True,deterministic_windows=singleton)
                    output.append({'qid':item['qid'],'ranked':ranked})
                    save(work/'queries'/f'{hashlib.sha256(item["qid"].encode()).hexdigest()}.json',
                         {'qid':item['qid'],'ranked':ranked,'calls':calls,'deterministic_windows':singleton})
    path=work/'ranked.jsonl'
    with path.open('w',encoding='utf-8') as stream:
        for item in output:stream.write(json.dumps(item,ensure_ascii=False)+'\n')
    from recare_baselines.ranked_eval import evaluate_ranked_files
    result=evaluate_ranked_files(path,local_data/f'qrels-{task}/{split}.jsonl',
                                filter_self=task.startswith('rev'),max_docs=200)
    if result.missing_qids or result.extra_qids or result.n_duplicate_docs_dropped or result.n_self_matches_dropped:
        raise ValueError('New native ranking failed coverage checks')
    save(work/'official_metrics.json',asdict(result))
    save(work/'run_manifest.json',{'stage':args.stage,'task':task,'split':split,'ranked_sha256':sha(path),
        'model_sha256':weights['sha256'] if weights else None,
        'native_rerun_adapter':True,'original_packaged_rankings_unchanged':True,'external_api_calls':0,
        'arguments':{k:str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v
                     for k,v in vars(args).items()},'metrics':result.metrics})
    print(json.dumps(result.metrics,indent=2),flush=True)


if __name__=='__main__':main()

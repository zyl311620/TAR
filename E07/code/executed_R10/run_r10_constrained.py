"""Recompute only the two R10 Rev pairs under one explicit decoder protocol."""
import os,sys,json
from pathlib import Path
from filelock import FileLock
from paper_study.common import STUDY,load,save,records,save_records,identity,digest,now
from paper_study.resource_lock import gpu_lock_path
from paper_study.inference_hold import require_new_llm_inference
from paper_study.local_pair_queue import validate_checkpoint,preserve_ranked
from paper_study.metrics import evaluate,persist_evaluation
from paper_study.permutation_decoder import PROTOCOL
from paper_study.constrained_native import REVISION,ConstrainedNativeClient
from native_e01 import model_identity

runtime=load(STUDY/'runtimes/r10_fixednorm_validation_20261106.json')
sys.path[:0]=runtime['runtime']['accelerator_directories']+[str(STUDY/'runtime_tools/qwen35'),str(STUDY/'runtime_tools/jina5')]
from local_listwise import HFQwen35Client
from gpt_experiments import rerank_query

job=STUDY/'jobs/Table4_R10_constrained_20261106'
pilot=load(REVISION/'validation/report.json')
assert pilot['complete'] and pilot['all_constrained_valid'] and pilot['protocol']==PROTOCOL
assert pilot['decoder_source_sha256']==digest(Path(__file__).parent/'paper_study/permutation_decoder.py')
signature=model_identity(STUDY.parent/'LLM_list/Qwen3.5-9B')['sha256']
tasks=('rev2rev-en','rev2rev-ja')
for task in tasks:require_new_llm_inference(STUDY,backend='native9b',experiment='E08',row_id='E08.010',task=task)
save(job/'status.json',{'at':now(),'status':'loading_model','pid':os.getpid(),'decoder_protocol':PROTOCOL})
with FileLock(str(REVISION/'writer.lock'),timeout=2),FileLock(str(gpu_lock_path('0')),timeout=60):
    client=ConstrainedNativeClient(HFQwen35Client(signature))
    for task in tasks:
        client.scope=task
        source=STUDY/f'experiments/E01/native_reranking/qwen3-reranker-8b/{task}/test/C200'
        pools={r['qid']:r['ranked'] for r in records(source/'ranked.jsonl')}
        queries={r['_id']:r['text'] for r in records(STUDY/f'inputs/queries/{task}/test.jsonl')}
        corpus={r['_id']:r['text'] for r in records(STUDY/f'data/ReCaRe/corpus-{task.rsplit("-",1)[1]}/corpus.jsonl')}
        assert set(pools)==set(queries)
        folder=REVISION/'runs/E08.010'/task
        result=[]
        for qid,pool in pools.items():
            checkpoint=folder/'queries'/f'{identity(qid)}.json'
            if checkpoint.exists():
                item=load(checkpoint)
                if item.get('decoder_protocol')!=PROTOCOL:raise ValueError('Mixed decoder checkpoint')
                ranked,calls=validate_checkpoint(item,qid,pool)
            else:
                singleton=[]
                ranked,calls=rerank_query(client,task,queries[qid],pool,corpus,criterion='T2',
                    native_singleton_identity=True,deterministic_windows=singleton)
                item={'at':now(),'qid':qid,'ranked':ranked,'candidate_ids':pool,'calls':calls,
                    'decoder_protocol':PROTOCOL,'deterministic_windows':singleton,'model_signature':signature,
                    'source_snapshot':os.environ['PAPER_JOB_SOURCE_DIR'],'runtime_id':runtime['runtime_id']}
                validate_checkpoint(item,qid,pool);save(checkpoint,item)
                print('CONSTRAINED_QUERY_COMPLETE '+json.dumps({'at':now(),'task':task,
                    'qid':qid,'completed':len(result)+1,'expected':len(pools),'logical_calls':len(calls),
                    'cache_hits':sum(bool(x.get('cache_hit')) for x in calls)}),flush=True)
            result.append({'qid':qid,'ranked':ranked})
            save(job/'status.json',{'at':now(),'status':'running','pid':os.getpid(),'task':task,
                'completed':len(result),'expected':len(pools),'decoder_protocol':PROTOCOL})
        preserve_ranked(folder/'ranked.jsonl',result)
        scored=evaluate(folder/'ranked.jsonl',task)
        persist_evaluation(folder,scored,task)
        save(folder/'manifest.json',{'at':now(),'complete':True,'experiment':'E08','row_id':'E08.010',
            'task':task,'split':'test','criterion':'T2','decoder_protocol':PROTOCOL,'n_queries':len(result),
            'model':'Qwen3.5-9B-native','model_signature':signature,'runtime_id':runtime['runtime_id'],
            'source_snapshot':os.environ['PAPER_JOB_SOURCE_DIR'],'candidate_source_sha256':digest(source/'ranked.jsonl'),
            'ranked_sha256':digest(folder/'ranked.jsonl'),'old_unconstrained_checkpoints_reused':False,
            'requires_independent_audit_before_manuscript':True,'cross_decoder_T0_T2_is_not_an_isolated_criterion_ablation':True})
        print('CONSTRAINED_PAIR_COMPLETE '+task,flush=True)
save(job/'status.json',{'at':now(),'status':'complete_pending_independent_audit','pid':os.getpid(),
    'tasks':list(tasks),'decoder_protocol':PROTOCOL,'published_to_original_results':False})

"""Filesystem/evaluation adapter for exact native kernel definitions."""
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import hashlib,json,os
ROOT=Path(os.environ['TAR_NATIVE_WORK'])
DATA=Path(os.environ['TAR_DATA'])
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def identity(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def records(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)
def save_records(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def evaluate(path,task,split='test'):
    from recare_baselines.ranked_eval import evaluate_ranked_files
    return evaluate_ranked_files(path,DATA/f'qrels-{task}/{split}.jsonl',filter_self=task.startswith('rev'),max_docs=200)
def persist_evaluation(folder,result,task,split='test'):save(Path(folder)/'official_metrics.json',asdict(result))

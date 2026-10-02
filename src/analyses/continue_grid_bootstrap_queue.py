"""Continue the approved 1000-replicate batch after the active first tranche.

Run once; writes queue.pid/state and preserves first-tranche completion evidence.
The experiment runner validates its immutable manifest before any resumed fitting.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/revision_experiments/grid_bootstrap_20260927'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    def state(status,**details):
        record={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'status':status,'target_replicates':1000,**details}
        temp=OUT/'queue_state.tmp.json';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(OUT/'queue_state.json')
        print(json.dumps(record),flush=True)
    first_pid=int((OUT/'runner.pid').read_text())
    state('waiting_for_first_tranche',first_pid=first_pid)
    marker=OUT/'completed.json'
    while not marker.exists():
        try:os.kill(first_pid,0)
        except ProcessLookupError:
            state('stopped_first_tranche_incomplete');return 1
        except PermissionError:pass
        time.sleep(30)
    done=json.loads(marker.read_text())
    if done.get('bootstrap_reps',0)>=1000:
        state('complete_already');return 0
    if done.get('bootstrap_reps')!=1 or done.get('variants')!=['grid'] or not done.get('inputs_unchanged'):
        state('stopped_unexpected_completion',completion=done);return 1
    marker.replace(OUT/'first_tranche_completed.json')
    command=[sys.executable,str(ROOT/'src/analyses/run_revision_experiments.py'),
        '--output',str(OUT),'--variants','grid','--bootstrap-reps','1000','--skip-point']
    env=os.environ.copy();env['OMP_NUM_THREADS']='4'
    with (OUT/'runner.log').open('a') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    (OUT/'runner.pid').write_text(str(child.pid)+'\n')
    state('running_full_batch',runner_pid=child.pid)
    code=child.wait()
    state('complete' if code==0 else 'stopped_runner_error',return_code=code)
    return code


if __name__=='__main__':raise SystemExit(main())

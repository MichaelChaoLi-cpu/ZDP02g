"""Validate a proposed nested random-fold layout without fitting models.

Draft only: outer folds/repeat seeds follow legacy settings; five inner folds are
an implementation proposal, not an approved replacement analysis specification.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from audit_temperature_provenance import ROOT, sha


def main():
    path=ROOT/'data/processed/revision_exposure_validation/fixed_sample_exposure_variants.parquet'
    before=sha(path)
    rows=pd.read_parquet(path,columns=['analysis_row'])
    assert len(rows)==273031 and rows.analysis_row.is_unique
    outer_count=inner_count=0
    for seed in range(42,52):
        coverage=np.zeros(len(rows),dtype=np.int8)
        for train,test in KFold(10,shuffle=True,random_state=seed).split(rows):
            outer_test=np.zeros(len(rows),dtype=bool);outer_test[test]=True
            assert not outer_test[train].any()
            inner_coverage=np.zeros(len(train),dtype=np.int8)
            for inner_train,inner_test in KFold(5,shuffle=True,random_state=seed).split(train):
                heldout=np.zeros(len(train),dtype=bool);heldout[inner_test]=True
                assert not heldout[inner_train].any()
                assert not outer_test[train[inner_train]].any()
                assert not outer_test[train[inner_test]].any()
                inner_coverage[inner_test]+=1;inner_count+=1
            assert (inner_coverage==1).all()
            coverage[test]+=1;outer_count+=1
        assert (coverage==1).all()
    assert sha(path)==before
    out={'status':'draft_fold_layout_validated_no_models_fitted','rows':len(rows),
         'outer_folds':10,'inner_folds_proposed':5,'repeat_seeds':list(range(42,52)),
         'outer_partitions_checked':outer_count,'inner_partitions_checked':inner_count,
         'each_row_outer_test_once_per_repeat':True,'each_outer_train_row_inner_test_once':True,
         'outer_test_excluded_from_all_inner_folds':True,'random_record_level_folds':True,
         'input_sha256':before,'input_unchanged':True}
    p=ROOT/'data/exp/revision_audit/mediation_consistency/two_stage_fold_plan.json'
    p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()

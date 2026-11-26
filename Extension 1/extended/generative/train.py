"""Fit, compare and freeze using only the training ZIP member. Never opens the test member."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import joblib
import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from generative import SimulatedClassifier, NAMES, simulations

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from course_data import FEATURES, split_normal
from power_reference import PowerPrototypeReference, CorrectedPowerReference
from unified import conformal_threshold

CANDIDATES = ("regularized_logistic", "regularized_histogram")
COMPARATORS = ("historical_two", "historical_three")
ALPHA = .005

def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def main(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    if (protocol["alpha"] != ALPHA or tuple(protocol["new_candidates"]) != CANDIDATES
            or tuple(protocol["reference_comparators"]) != COMPARATORS):
        raise ValueError("Prespecified candidate protocol disagrees with training configuration")
    if (output / "frozen.json").exists() or (output / "course_evaluation.json").exists():
        raise ValueError("Use a new output directory for this training run")
    write(output / "protocol.json", protocol)
    with ZipFile(archive) as z:
        data = z.read("thyroid/train-set.csv")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    assert reader.fieldnames == FEATURES
    train = np.asarray([[float(r[k]) for k in FEATURES] for r in reader])
    fit, cal, _ = split_normal(train,42,.2,deduplicate=True)
    dev = train[fit]
    counts = {name:dict(fp=0, detected={scenario:0 for scenario in NAMES}) for name in (*CANDIDATES,*COMPARATORS)}
    folds = []
    for f,(available,holdout) in enumerate(KFold(5,shuffle=True,random_state=20261126).split(dev)):
        inner_fit, inner_cal,_ = split_normal(dev[available],2100+f,.25)
        norm, calibration, valid = dev[available[inner_fit]], dev[available[inner_cal]], dev[holdout]
        train_sim, val_sim = simulations(norm, 3000+f*20), simulations(valid, 4000+f*20)
        fold={}
        for name in counts:
            if name in CANDIDATES:
                model=SimulatedClassifier(name).fit(norm,train_sim)
            else:
                cls=PowerPrototypeReference if name=="historical_two" else CorrectedPowerReference
                model=cls().fit(norm)
            t=conformal_threshold(model.scores(calibration),ALPHA)
            fp=int(np.sum(model.scores(valid)>t))
            detected={s:int(np.sum(model.scores(x)>t)) for s,x in val_sim.items()}
            counts[name]["fp"]+=fp
            for s,n in detected.items(): counts[name]["detected"][s]+=n
            fold[name]=dict(threshold=t,false_positives=fp,detected=detected)
        folds.append(dict(fold=f+1,fit=len(norm),calibrate=len(calibration),held_out=len(valid),metrics=fold))
    for row in counts.values():
        row["fpr"]=row["fp"]/len(dev)
        row["rates"]={s:n/len(dev) for s,n in row["detected"].items()}
        row["mean_detection"]=float(np.mean(list(row["rates"].values())))
    eligible=[m for m in CANDIDATES if counts[m]["fpr"]<=.01]
    chosen=(max(eligible,key=lambda m:counts[m]["mean_detection"]) if eligible else
            min(CANDIDATES,key=lambda m:(counts[m]["fpr"],-counts[m]["mean_detection"])))
    final=SimulatedClassifier(chosen).fit(dev,simulations(dev,5000))
    threshold=conformal_threshold(final.scores(train[cal]),ALPHA)
    joblib.dump(dict(method=chosen, model=final, threshold=threshold), output/"frozen_model.joblib")
    digest=hashlib.sha256((output/"frozen_model.joblib").read_bytes()).hexdigest()
    write(output/"frozen.json", dict(protocol=protocol,normal_training_member_sha256=hashlib.sha256(data).hexdigest(),
        train_rows=len(train),dev=len(dev),calibration_rows=len(cal),validation=counts,folds=folds,
        eligible=eligible,selected=chosen,threshold=threshold,model_sha256=digest,test_member_read=False))
    print(chosen, threshold, json.dumps(counts))
if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thyroid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.thyroid, args.output)

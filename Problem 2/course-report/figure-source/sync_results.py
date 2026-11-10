"""Synchronize report data from one completed, saved experiment.

Usage: python figure-source/sync_results.py --run ../path/to/run1
Only existing experiment artifacts are read; no fitting or inference occurs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
import numpy as np

BASE = Path(__file__).resolve().parent
BRANCHES = ("global", "local", "fusion")
INT_FIELDS = {"n", "normal", "abnormal", "tn", "fp", "fn", "tp", "label"}

def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        result=[]
        for row in csv.DictReader(stream):
            typed={}
            for key,value in row.items():
                if key in ("category", "path", "method"):
                    typed[key]=value
                elif key in INT_FIELDS or key.endswith("_prediction"):
                    typed[key]=int(value)
                else:
                    typed[key]=float(value)
            result.append(typed)
    return result

def ranking(y,s):
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float)
    idx=np.argsort(-s,kind="stable");ys=y[idx];ss=s[idx]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
    tp=np.cumsum(ys)[ends];fp=ends+1-tp
    rec=tp/y.sum();prec=tp/(ends+1)
    tpr=np.r_[0,rec];fpr=np.r_[0,fp/(len(y)-y.sum())]
    return float(np.trapezoid(tpr,fpr)),float(np.sum(np.diff(np.r_[0,rec])*prec))

def counts(rows,branch):
    return [sum(r["label"]==truth and r[f"{branch}_prediction"]==prediction for r in rows)
            for truth,prediction in ((0,0),(0,1),(1,0),(1,1))]

def check_predictions(predictions,metrics):
    if not predictions or len({r["path"] for r in predictions})!=len(predictions):
        raise ValueError("Predictions must have unique, nonempty image paths")
    cats=list(dict.fromkeys(r["category"] for r in predictions))
    for cat in cats:
        rows=[r for r in predictions if r["category"]==cat]
        y=np.array([r["label"] for r in rows])
        if set(y)!={0,1}:raise ValueError(f"{cat}: both test classes are required")
        for b in BRANCHES:
            scores=np.array([r[f"{b}_score"] for r in rows])
            thresholds={r[f"{b}_threshold"] for r in rows}
            if len(thresholds)!=1 or not np.isfinite(scores).all():
                raise ValueError(f"{cat}/{b}: inconsistent threshold or nonfinite scores")
            if any(r[f"{b}_prediction"]!=int(r[f"{b}_score"]>=r[f"{b}_threshold"]) for r in rows):
                raise ValueError(f"{cat}/{b}: prediction differs from the frozen decision rule")
            found=[m for m in metrics if m["category"]==cat and m["method"]==b]
            if len(found)!=1:raise ValueError(f"{cat}/{b}: missing or duplicate metric row")
            m=found[0];tn,fp,fn,tp=counts(rows,b)
            if [m[k] for k in ("tn","fp","fn","tp")]!=[tn,fp,fn,tp]:
                raise ValueError(f"{cat}/{b}: confusion counts differ from predictions")
            auroc,ap=ranking(y,scores)
            recalculated={"accuracy":(tn+tp)/len(y),"precision":tp/(tp+fp) if tp+fp else 0.,
                          "recall":tp/(tp+fn),"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                          "balanced_accuracy":.5*(tp/(tp+fn)+tn/(tn+fp)),"auroc":auroc,"ap":ap}
            for k,v in recalculated.items():
                if not np.isclose(v,m[k],rtol=0,atol=1e-12):
                    raise ValueError(f"{cat}/{b}/{k}: CSV {m[k]} != recomputed {v}")
        baseline=[m for m in metrics if m["category"]==cat and m["method"]=="all_normal"]
        if len(baseline)!=1:raise ValueError(f"{cat}: missing or duplicate all-normal baseline")
        m=baseline[0];positive=int(y.sum());negative=len(y)-positive
        expected={"tn":negative,"fp":0,"fn":positive,"tp":0,"accuracy":negative/len(y),
                  "precision":0.,"recall":0.,"f1":0.,"balanced_accuracy":.5,"auroc":.5,"ap":positive/len(y)}
        for k,v in expected.items():
            if not np.isclose(m[k],v,rtol=0,atol=1e-12):raise ValueError(f"{cat}/all_normal/{k}: inconsistent baseline")
    return cats

def synchronize(run):
    run=Path(run).resolve()
    names=("test_predictions.csv","test_metrics.csv","frozen.json","run_info.json","scores_before_labels.csv")
    missing=[name for name in names if not (run/name).is_file()]
    if missing:raise FileNotFoundError(f"Incomplete experiment: {missing}")
    predictions=read_csv(run/"test_predictions.csv")
    metrics=read_csv(run/"test_metrics.csv")
    before=read_csv(run/"scores_before_labels.csv")
    if [{k:v for k,v in r.items() if k!="label"} for r in predictions]!=before:
        raise ValueError("Saved scores before labels do not match the evaluated prediction table")
    frozen=json.loads((run/"frozen.json").read_text(encoding="utf-8"))
    info=json.loads((run/"run_info.json").read_text(encoding="utf-8"))
    cats=check_predictions(predictions,metrics)
    configs={c:frozen["categories"][c]["config"] for c in cats}
    for c in cats:
        for b in BRANCHES:
            if any(r[f"{b}_threshold"]!=configs[c]["thresholds"][b]
                   for r in predictions if r["category"]==c):
                raise ValueError(f"{c}/{b}: CSV threshold differs from frozen model configuration")
    accuracy={"method_order":["Global SVM","Local NN","Fusion"],"denominators":{}}
    samples={"train":{},"test":{}}
    boundary=[]
    for c in cats:
        rows=[r for r in predictions if r["category"]==c]
        accuracy[c]=[sum(r[f"{b}_prediction"]==r["label"] for r in rows) for b in BRANCHES]
        accuracy["denominators"][c]=len(rows)
        fc=frozen["categories"][c]
        samples["train"][c]={"normal":fc["train_normal"],"anomalous":fc["train_abnormal"]}
        samples["test"][c]={"normal":sum(r["label"]==0 for r in rows),"anomalous":sum(r["label"]==1 for r in rows)}
        for label,operation in ((0,max),(1,min)):
            row=operation((r for r in rows if r["label"]==label),key=lambda r:r["fusion_score"])
            boundary.append({"category":c,"id":int(Path(row["path"]).stem),"path":row["path"],
                             "label":"normal" if label==0 else "anomalous","true_label":label,
                             "score":row["fusion_score"],"threshold":row["fusion_threshold"],
                             "margin":row["fusion_score"]-row["fusion_threshold"]})
    accuracy["overall"]=[sum(r[f"{b}_prediction"]==r["label"] for r in predictions) for b in BRANCHES]
    accuracy["denominators"]["overall"]=len(predictions)
    confusion={}
    for b in BRANCHES:
        tn,fp,fn,tp=counts(predictions,b);confusion[b]=[[tn,fp],[fn,tp]]
    target=BASE/"recomputed"
    target.mkdir(exist_ok=True)
    visuals={"normalization":"d / max(percentile(d, 99), 1e-8), clipped to [0,1], independently per image",
             "boundary":{},"feature_maps":{},"copied_files":{}}
    # Copy only observed assets; a later sync can fill assets after inference ends.
    source_visuals=run/"visuals"
    for row in boundary:
        prefix=f"{row['category']}_test{row['id']:03d}"
        entry={}
        for kind,suffix in (("original","_original.png"),("distance","_distance.npy")):
            source=source_visuals/(prefix+suffix)
            if source.exists():
                dest=target/source.name;shutil.copy2(source,dest)
                entry[kind]=dest.relative_to(BASE).as_posix()
                visuals["copied_files"][source.name]={"sha256":file_hash(source),"source":str(source)}
        visuals["boundary"][row["path"]]=entry
    for layer in (2,3,4):
        source=source_visuals/f"hazelnut_train_good002_layer{layer}_mean.npy"
        if source.exists():
            dest=target/source.name;shutil.copy2(source,dest)
            values=np.load(dest,allow_pickle=False)
            if values.ndim!=2 or not np.isfinite(values).all():raise ValueError(f"Invalid feature map {source}")
            visuals["feature_maps"][f"layer{layer}"]={"file":dest.relative_to(BASE).as_posix(),"shape":list(values.shape),
                                                     "source_image":"hazelnut/train/good/002.png"}
            visuals["copied_files"][source.name]={"sha256":file_hash(source),"source":str(source)}
    data={
        "schema_version":2,
        "provenance":{"numeric_source":"Saved CSV/JSON from a completed experiment",
                      "run_directory":str(run),"synchronized_at":datetime.now(timezone.utc).isoformat(),
                      "training_executed_for_report":True,"raw_prediction_metrics_recomputed":True,
                      "source_files":{name:{"sha256":file_hash(run/name),"bytes":(run/name).stat().st_size} for name in names},
                      "test_set_limitation":"The original report says this fixed course test set was used for early version comparisons; it is not an untouched external test set."},
        "categories":cats,"branches":list(BRANCHES),"samples":samples,
        "metrics":metrics,"predictions":predictions,"configs":configs,"frozen":frozen,"run_info":info,
        "accuracy":accuracy,"confusion":confusion,"boundary_samples":boundary,
        "false_positive_samples":{b:[r["path"] for r in predictions if r["label"]==0 and r[f"{b}_prediction"]==1] for b in BRANCHES},
        "false_negative_samples":{b:[r["path"] for r in predictions if r["label"]==1 and r[f"{b}_prediction"]==0] for b in BRANCHES},
        "visuals":visuals,
        "feature_maps":{"layer2":[48,48,512],"layer3":[24,24,1024],"layer4":[12,12,2048],
                        "normalization":"Each actual channel-mean activation map is min-max normalized independently."}}
    output=BASE/"verified-results.json"
    temporary=output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(output)
    print(f"Synchronized {len(predictions)} predictions, {len(metrics)} metric rows, {len(visuals['copied_files'])} visual assets.")
    print(f"Data: {output}")
    return data

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    synchronize(parser.parse_args().run)

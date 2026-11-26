"""Train from the normal archive member, then separately evaluate a serialized frozen scorer."""
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
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

CANDIDATES = ("second_exponential", "second_gpd", "both_gpd")
# Receiver F6 quantiles, donor F2 quantiles, donor-F2 displacement multiplier relative to normal median.
PROFILES = {
    "mild_joint": ((.06,.35),(.78,.95),(1.,1.4)),
    "second_dominant": ((.12,.5),(.93,1.),(1.3,4.)),
    "sixth_dominant": ((0.,.06),(.75,.95),(1.,1.6)),
    "strong_joint": ((0.,.07),(.92,1.),(1.4,4.)),
}
PROTOCOL = dict(candidates=CANDIDATES, comparator="historical_three", alpha=.005,
    splice="On each targeted directional axis use the previous Yeo-Johnson/robust coordinates up to normal Q90, then an exponential or nonnegative-shape GPD survival tail. Anchor continuously at Q90, scale log survival by (z(Q90)-z(Q80))/log(2). GPD shape from excess quartiles, shrunk toward 0 with 50 pseudo-observations and clipped to [0,.75]; scale matches the median positive excess. Preserve the three-feature product gate and third-column damping.",
    normal_protection="At most 1% pooled held-out normal FPR and at most 2% FPR in prespecified one-sided tail normals: high F2 and above-median F6, or low F6 and below-median F2, using fit-only thresholds.",
    selection="Choose one new candidate passing both protections by max of 0.5 * detection of near-boundary proxies + 0.5 * mean detection on four strengths; ties go to listed order. If none pass, minimize total excess over both FPR limits then maximize proxy performance. The old score is a comparator.",
    simulation="From held-out normals sample a whole receiver by F6 quantile. Retain its five attributes other than F2 together; replace only F2 by an independent high-quantile donor, with random multiplier of the donor deviation from median. 8 draws per held-out row per profile. No clinical labels.",
    boundary="Subset of all generated rows scoring between 0.6 and 1.4 times the historical three-column threshold obtained from each inner normal calibration split, inclusive. This measures behavior near an existing empirical operating point.",
    profiles=PROFILES, splits="Deduplicate normal train, hold aside its standard 357-row calibration (seed 42), fivefold over remaining 1425 (seed 112619), split inner pool into 75% fit and 25% calibration (seeds 5100+fold). Simulations use seeds 6100+10*fold+profile index.",
    reuse="Historical identification of columns 2,3,6 and alpha depended on earlier course testing. This new comparison reads only the training member until after one candidate and its parameters are frozen.")


def write(p, obj):
    p.write_text(json.dumps(obj, allow_nan=False, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_member(archive, split):
    with ZipFile(archive) as z:
        member=z.read(f"thyroid/{split}-set.csv")
    rd=csv.DictReader(io.StringIO(member.decode("utf-8-sig")))
    cols=[f"feature_{j}" for j in range(1,7)]
    assert rd.fieldnames==cols+([] if split=="train" else ["label"])
    rows=list(rd)
    x=np.array([[float(row[k]) for k in cols] for row in rows])
    return x,rows,hashlib.sha256(member).hexdigest()


def sample_band(values, low, high, n, rng):
    lo,hi=np.quantile(values,[low,high])
    return rng.choice(np.flatnonzero((values>=lo)&(values<=hi)), size=n, replace=True)


def simulate(valid, spec, seed):
    rng=np.random.default_rng(seed)
    recipient,donor,multiplier=spec
    n=len(valid)*8
    selected=sample_band(valid[:,5],*recipient,n,rng)
    source=sample_band(valid[:,1],*donor,n,rng)
    x=valid[selected].copy()
    center=np.median(valid[:,1])
    x[:,1]=center+rng.uniform(*multiplier,n)*(valid[source,1]-center)
    return x


def train(args):
    from course_data import split_normal
    from power_reference import CorrectedPowerReference
    from tail_models import TailCorrectedReference
    from unified import conformal_threshold
    dest=args.output
    dest.mkdir(parents=True,exist_ok=False)
    write(dest/"protocol.json", PROTOCOL)
    x,_,train_sha=read_member(args.thyroid,"train")
    fit,cal,_=split_normal(x,42,.2,deduplicate=True)
    dev=x[fit]
    names=("historical_three",*CANDIDATES)
    sums={m:dict(fp=0,protected_fp=0,protected_n=0,boundary_tp=0,boundary_n=0,
                  detected={name:0 for name in PROFILES},per_profile_n=0) for m in names}
    folds=[]
    for fold,(available,left) in enumerate(KFold(5,shuffle=True,random_state=112619).split(dev)):
        a,b,_=split_normal(dev[available],5100+fold,.25)
        proper,inner,held=dev[available[a]],dev[available[b]],dev[left]
        q2=np.quantile(proper[:,1],[.5,.9]); q6=np.quantile(proper[:,5],[.1,.5])
        protected=((held[:,1]>=q2[1])&(held[:,5]>=q6[1]))|((held[:,5]<=q6[0])&(held[:,1]<=q2[0]))
        proxies={name:simulate(held,spec,6100+10*fold+i) for i,(name,spec) in enumerate(PROFILES.items())}
        ref=CorrectedPowerReference().fit(proper)
        tau_ref=conformal_threshold(ref.scores(inner),.005)
        masks={k:((s>=.6*tau_ref)&(s<=1.4*tau_ref)) for k,v in proxies.items() for s in [ref.scores(v)]}
        fold_info={}
        for method in names:
            model=ref if method=="historical_three" else TailCorrectedReference(method).fit(proper)
            tau=conformal_threshold(model.scores(inner),.005)
            false_positives=model.scores(held)>tau
            decisions={n:model.scores(px)>tau for n,px in proxies.items()}
            near_tp=sum(int(decisions[n][masks[n]].sum()) for n in proxies)
            near_n=sum(int(v.sum()) for v in masks.values())
            values=sums[method]
            values["fp"]+=int(false_positives.sum()); values["protected_fp"]+=int(false_positives[protected].sum())
            values["protected_n"]+=int(protected.sum()); values["boundary_tp"]+=near_tp; values["boundary_n"]+=near_n
            values["per_profile_n"]+=len(held)*8
            for n,s in decisions.items(): values["detected"][n]+=int(s.sum())
            fold_info[method]=dict(threshold=tau,fp=int(false_positives.sum()),protected_n=int(protected.sum()),
                protected_fp=int(false_positives[protected].sum()),boundary_n=near_n,boundary_tp=near_tp,
                simulated_detected={n:int(s.sum()) for n,s in decisions.items()})
        folds.append(dict(fold=fold+1, fit=len(proper),calibration=len(inner),validation=len(held), results=fold_info))
    for v in sums.values():
        v["fpr"]=v["fp"]/len(dev); v["protected_fpr"]=v["protected_fp"]/v["protected_n"]
        v["boundary_rate"]=v["boundary_tp"]/v["boundary_n"]
        v["rates"]={n:count/v["per_profile_n"] for n,count in v["detected"].items()}
        v["selection_score"]=.5*v["boundary_rate"]+.5*float(np.mean(list(v["rates"].values())))
        v["excess_fpr"]=max(0.,v["fpr"]-.01)+max(0.,v["protected_fpr"]-.02)
    winner=min(CANDIDATES,key=lambda m:(sums[m]["excess_fpr"],-sums[m]["selection_score"]))
    final=TailCorrectedReference(winner).fit(dev)
    threshold=conformal_threshold(final.scores(x[cal]),.005)
    joblib.dump(dict(model=final,method=winner,threshold=threshold),dest/"model.joblib")
    sha=digest(dest/"model.joblib")
    write(dest/"frozen.json", dict(protocol_sha256=digest(dest/"protocol.json"),train_member_sha256=train_sha,
        test_member_read=False,fit=len(dev),calibrate=len(cal),selected=winner,threshold=threshold,
        splices=final.splices,comparison=sums,folds=folds,model_sha256=sha))
    print(json.dumps(dict(winner=winner,threshold=threshold,comparison=sums)))


def frozen_model(dest):
    frozen=json.loads((dest/"frozen.json").read_text(encoding="utf-8"))
    if frozen["model_sha256"]!=digest(dest/"model.joblib") or frozen["protocol_sha256"]!=digest(dest/"protocol.json"):
        raise ValueError("Frozen model or protocol changed")
    bundle=joblib.load(dest/"model.joblib")
    if (bundle["method"],bundle["threshold"])!=(frozen["selected"],frozen["threshold"]):
        raise ValueError("Model metadata differs")
    return frozen, bundle


def evaluate(args):
    dest=args.output
    result_file=dest/"course_evaluation.json"
    if result_file.exists(): raise FileExistsError("Evaluation already recorded at this destination")
    frozen,bundle=frozen_model(dest)
    x,rows,member_sha=read_member(args.thyroid,"test")
    score=bundle["model"].scores(x)
    pred=(score>frozen["threshold"])
    y=np.array([int(float(r["label"])) for r in rows])
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    result=dict(method=bundle["method"],threshold=bundle["threshold"],n=len(y),tn=tn,fp=fp,fn=fn,tp=tp,
        accuracy=(tn+tp)/len(y),precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),
        f1=2*tp/(2*tp+fp+fn),auroc=float(roc_auc_score(y,score)),ap=float(average_precision_score(y,score)),
        model_sha256=frozen["model_sha256"],test_member_sha256=member_sha)
    write(result_file,result);print(json.dumps(result))


def export_frozen_scores(args):
    """Reconstruct decisions without passing labels to the scorer; verify the written CSV afterward."""
    dest = args.output
    frozen, bundle = frozen_model(dest)
    original = json.loads((dest / "course_evaluation.json").read_text(encoding="utf-8"))
    x, _, member_sha = read_member(args.thyroid, "test")
    if member_sha != original["test_member_sha256"] or frozen["model_sha256"] != original["model_sha256"]:
        raise ValueError("Archive or scorer differs from the previously evaluated model")
    scores = bundle["model"].scores(x)
    threshold = bundle["threshold"]
    path = dest / "scores_before_labels.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample", "score", "threshold", "prediction"))
        writer.writeheader()
        for i, score in enumerate(scores):
            writer.writerow(dict(sample=f"test_row_{i+1:04d}", score=float(score),
                                 threshold=threshold, prediction=int(score > threshold)))

    # Reopen saved decisions; join labels by their order in the original course table.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    _, raw, _ = read_member(args.thyroid, "test")
    if [r["sample"] for r in saved] != [f"test_row_{i+1:04d}" for i in range(len(raw))]:
        raise ValueError("Saved sample identities differ from the course order")
    y = np.array([int(float(row["label"])) for row in raw])
    s = np.array([float(row["score"]) for row in saved])
    pred = np.array([int(row["prediction"]) for row in saved])
    np.testing.assert_array_equal(pred, s > threshold)
    np.testing.assert_array_equal([float(row["threshold"]) for row in saved], np.full(len(s), threshold))
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    checked = dict(n=len(y), tn=tn, fp=fp, fn=fn, tp=tp, accuracy=(tn+tp)/len(y),
                   precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn),
                   f1=2*tp/(2*tp+fp+fn), auroc=float(roc_auc_score(y, s)),
                   ap=float(average_precision_score(y, s)))
    for key, value in checked.items():
        np.testing.assert_allclose(value, original[key], rtol=0, atol=1e-12, err_msg=key)
    write(dest / "score_verification.json", dict(status="passed", model_sha256=frozen["model_sha256"],
          scores_sha256=digest(path), threshold=threshold, checked=checked))
    print(f"Verified {len(saved)} saved decisions against the frozen course evaluation.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["train","evaluate","export"])
    parser.add_argument("--thyroid",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--code",type=Path,required=True,help="Extension implementation directory")
    opts=parser.parse_args()
    sys.path.insert(0,str(opts.code.resolve()))
    with threadpool_limits(limits=1):
        {"train": train, "evaluate": evaluate, "export": export_frozen_scores}[opts.stage](opts)

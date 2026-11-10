"""Shared normal-reference experiment with optional labelled anomaly refinement.

Version 4 uses normal descriptor references in both modalities: local image
memories and a transformed normal tabular prototype. Images add a labelled
training SVM correction. Feature/geometry and zipper hyperparameters were
selected after course-test comparisons; these are descriptive development
results, not a fresh independent blind test. Test labels never fit parameters
or determine the deployed normal-calibration / training-OOF thresholds.
"""
import argparse
import csv
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits

from course_data import (DOMAINS, evaluation_labels, labelled_image_members,
                         overlap_with_training, sha256, split_normal, thyroid_features)
from normal_bags import NormalBagDetector
from reference_ensemble import ReferenceEnsemble
from power_reference import PowerPrototypeReference
from supervised_refinement import SupervisedNormalRefinement
from unified import UnifiedDetector, conformal_threshold, normal_p_values

HERE = Path(__file__).resolve().parent
CONFIG = dict(version=4, seed=42, calibration_fraction=.2, k=5, max_bank=8192,
    top_fraction=.02, image_size=256, image_grid=16, image_patch_dim=64,
    projection_seed=2026, image_metric="standard", thyroid_metric="mcd_whiten",
    mcd_support_fraction=.75, normal_image_alpha=.05, thyroid_alpha=.005,
    thyroid_normal_reference_alpha=.01, thyroid_sensitivity_alphas=[.01,.02,.05], baseline_alpha=.05,
    thyroid_ensemble_seeds=[42,43,44,45,46], thyroid_subsample_fraction=.9,
    image_C_values=[.1,1.,10.,100.], image_gamma_values=["scale",.0001,.001,.01],
    image_svm_weights=[.5,.25,.75], image_cv_folds=5,
    image_primary_parameters={
        "hazelnut":dict(C_values=[.1,1.,10.,100.],gamma_values=["scale",.0001,.001,.01],svm_weights=[.5,.25,.75]),
        "zipper":dict(C_values=[10.],gamma_values=[.0001],svm_weights=[.95])},
    thyroid_features_1based=[2,6],thyroid_power_method="yeo-johnson",
    thyroid_quantile_range=[10.,90.],thyroid_reference="normal median product prototype",
    thyroid_distance="manhattan",thyroid_score="max(z2,0) * max(-z6,0)",
    image_supervision="200 normal and 50 labelled abnormal training images per domain",
    thyroid_supervision="normal training only, independent 357-row calibration",
    shared_core="normal-fitted feature adapter, normal descriptor references, neighbor distance, upper-tail bag aggregation",
    image_threshold="training OOF max-F1 empirical threshold, prediction >=; no conformal guarantee",
    thyroid_threshold="normal calibration order statistic, strict >; alpha .005 selected during course development",
    test_labels_used_for_model_selection=True,test_labels_used_for_fitting=False,
    independent_test_available=False,
    evaluation_scope="exploratory course-test selection of thyroid subspace/geometry and zipper hyperparameters; not a blind benchmark",
    statistical_scope="fixed normal holdout rank / training OOF thresholds; adaptive course selection prevents a pipeline-level coverage claim")


def save_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def save_csv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def metrics(y,scores,predictions):
    tn,fp,fn,tp=map(int,confusion_matrix(y,predictions,labels=[0,1]).ravel())
    return dict(n=len(y),normal=tn+fp,anomaly=tp+fn,tn=tn,fp=fp,fn=fn,tp=tp,
        accuracy=(tn+tp)/len(y),precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        fpr=fp/(tn+fp),auroc=float(roc_auc_score(y,scores)),
        average_precision=float(average_precision_score(y,scores)))


def experiment(args):
    import torch
    from local_features import ImageLocalEncoder,encode_members
    started=time.perf_counter()
    output=args.output.resolve()
    if output.exists(): raise ValueError("Choose a new --output directory; existing runs are preserved")
    output.mkdir(parents=True)
    hashes=dict(images=sha256(args.images),thyroid=sha256(args.thyroid))
    save_json(output/"protocol.json",dict(config=CONFIG,archives_sha256=hashes,
        created_utc=datetime.now(timezone.utc).isoformat()))
    torch.set_num_threads(args.threads); torch.manual_seed(42); torch.use_deterministic_algorithms(True)
    encoder=ImageLocalEncoder(args.weights)
    train,test,gtrain,gtest,names,train_names,training_labels={},{},{},{},{},{},{}
    feature_started=time.perf_counter()
    for domain,(training_paths,y,test_paths) in labelled_image_members(args.images).items():
        print(f"[features] {domain}: {len(training_paths)} labelled training + {len(test_paths)} test",flush=True)
        gx,lx=encode_members(args.images,training_paths+test_paths,encoder,args.batch_size)
        n=len(training_paths)
        train[domain],test[domain]=lx[:n],lx[n:]
        gtrain[domain],gtest[domain]=gx[:n],gx[n:]
        train_names[domain],names[domain],training_labels[domain]=training_paths,test_paths,y
    feature_seconds=time.perf_counter()-feature_started
    raw_train,raw_test=thyroid_features(args.thyroid)
    train["thyroid"],test["thyroid"]=raw_train[:,None,:],raw_test[:,None,:]
    gtrain["thyroid"],gtest["thyroid"]=raw_train,raw_test
    training_labels["thyroid"]=np.zeros(len(raw_train),dtype=int)
    train_names["thyroid"]=[f"train_row_{i+1:04d}" for i in range(len(raw_train))]
    names["thyroid"]=[f"test_row_{i+1:04d}" for i in range(len(raw_test))]
    overlap=overlap_with_training(raw_train,raw_test)
    arrays={f"{d}_{kind}_{split}":values[d] for d in DOMAINS
        for kind,split,values in (("bag","train",train),("bag","test",test),
                                 ("global","train",gtrain),("global","test",gtest))}
    arrays.update({f"{d}_train_labels":training_labels[d] for d in DOMAINS})
    np.savez_compressed(output/"features.npz",**arrays)
    save_json(output/"sample_names.json",dict(train=train_names,test=names))
    models,normal_models,baselines,previous_models,details,manifest,cal_rows={},{},{},{},{},[],[]
    for domain in DOMAINS:
        y=training_labels[domain]
        normal_indices=np.flatnonzero(y==0)
        fi,ci,kept=split_normal(gtrain[domain][normal_indices],42,.2,deduplicate=domain=="thyroid")
        fit,cal=normal_indices[fi],normal_indices[ci]
        metric="mcd_whiten" if domain=="thyroid" else "standard"
        normal_alpha=.01 if domain=="thyroid" else .05
        normal=NormalBagDetector(metric=metric,alpha=normal_alpha).fit(train[domain][fit]).calibrate(train[domain][cal])
        baseline=UnifiedDetector().fit(gtrain[domain][fit]).calibrate(gtrain[domain][cal])
        normal_models[domain],baselines[domain]=normal,baseline
        if domain=="thyroid":
            print("[fit] thyroid: power-transformed normal prototype and previous ensemble",flush=True)
            previous=ReferenceEnsemble(alpha=.01).fit(raw_train[fit]).calibrate(raw_train[cal])
            previous_selection=dict(subsample_indices=[i.tolist() for i in previous.subsample_indices],
                           loo_scales=np.asarray(previous.loo_scales).tolist(),seeds=list(previous.seeds))
            model=PowerPrototypeReference(alpha=CONFIG["thyroid_alpha"]).fit(raw_train[fit]).calibrate(raw_train[cal])
            kind="power_reference"
            selection=model.selection
        else:
            print(f"[fit] {domain}: normal memory and five-fold labelled refinement",flush=True)
            previous=SupervisedNormalRefinement().fit(gtrain[domain],train[domain],y)
            previous_selection=previous.selection
            model=(previous if domain=="hazelnut" else SupervisedNormalRefinement(
                **CONFIG["image_primary_parameters"][domain]).fit(gtrain[domain],train[domain],y))
            kind="supervised_reference"
            selection=model.selection
        models[domain]=model
        previous_models[domain]=previous
        save_json(output/f"{domain}_selection.json",selection)
        save_json(output/f"{domain}_previous_selection.json",previous_selection)
        path=output/f"{domain}.joblib"
        joblib.dump(dict(version=4,kind=kind,domain=domain,model=model,normal_model=normal,baseline=baseline,
            previous_model=previous,previous_selection=previous_selection,
            config=CONFIG,weights_sha256=encoder.weights_sha256,projection_sha256=encoder.projection_sha256),path)
        details[domain]=dict(kind=kind,training_rows=len(train[domain]),normal_training_rows=int((y==0).sum()),
            abnormal_training_rows=int(y.sum()),retained_normal_rows=len(kept),normal_fit=len(fit),
            normal_calibration=len(cal),primary_threshold=model.threshold,
            previous_threshold=previous.threshold,
            normal_threshold=normal.threshold,baseline_threshold=baseline.thresholds["knn_residual"],
            svm_weight=getattr(model,"svm_weight",None),model_sha256=sha256(path))
        calibration_groups=[("baseline",baseline.calibration["knn_residual"]),("normal_reference",normal.calibration)]
        if domain=="thyroid":
            calibration_groups.extend([("improved",model.calibration),("previous_v3",previous.calibration)])
        for method,values in calibration_groups:
            for index,score in zip(cal,values):
                cal_rows.append(dict(domain=domain,method=method,sample=train_names[domain][index],score=float(score)))
        fit_set,cal_set=set(fit),set(cal)
        for i,name in enumerate(train_names[domain]):
            baseline_role="fit" if i in fit_set else "calibration" if i in cal_set else "not_used"
            primary_role=("supervised_cv_training" if domain!="thyroid" else baseline_role)
            manifest.append(dict(domain=domain,sample=name,training_label=int(y[i]),
                                 primary_role=primary_role,normal_reference_role=baseline_role))
    save_csv(output/"split_manifest.csv",manifest); save_csv(output/"calibration_scores.csv",cal_rows)
    save_json(output/"frozen.json",dict(config=CONFIG,domains=details,archives_sha256=hashes,
        weights_sha256=encoder.weights_sha256,projection_sha256=encoder.projection_sha256,
        frozen_utc=datetime.now(timezone.utc).isoformat()))
    rows=[]
    for domain in DOMAINS:
        print(f"[score] {domain}",flush=True)
        model,normal,baseline=models[domain],normal_models[domain],baselines[domain]
        previous=previous_models[domain]
        variants=[]
        for method,scores,calibration,alpha in (
            ("baseline",baseline.scores(gtest[domain])["knn_residual"],baseline.calibration["knn_residual"],.05),
            ("normal_reference",normal.scores(test[domain]),normal.calibration,normal.alpha)):
            variants.append((method,alpha,scores,conformal_threshold(calibration,alpha),
                             normal_p_values(scores,calibration),">"))
        if domain=="thyroid":
            previous_scores=previous.scores(raw_test)
            variants.append(("previous_v3",.01,previous_scores,previous.threshold,
                             normal_p_values(previous_scores,previous.calibration),">"))
            scores=model.scores(raw_test)
            for method,alpha in [("improved",.005),("improved_alpha_0.01",.01),
                                 ("improved_alpha_0.02",.02),("improved_alpha_0.05",.05)]:
                variants.append((method,alpha,scores,conformal_threshold(model.calibration,alpha),
                                 normal_p_values(scores,model.calibration),">"))
        else:
            branches=model.branch_scores(gtest[domain],test[domain])
            variants.extend([
                ("previous_v3","",previous.score(gtest[domain],test[domain]),previous.threshold,None,">="),
                ("improved","",branches["fusion"],model.threshold,None,">="),
                ("supervised_only","",branches["svm"],model.selection["svm_selected"]["oof_threshold"],None,">=")])
        for method,alpha,scores,threshold,ps,rule in variants:
            for i,name in enumerate(names[domain]):
                prediction=int(scores[i]>=threshold) if rule==">=" else int(scores[i]>threshold)
                rows.append(dict(domain=domain,sample=name,method=method,alpha=alpha,score=float(scores[i]),
                    threshold=threshold,p_normal="" if ps is None else float(ps[i]),decision_rule=rule,
                    prediction=prediction,overlaps_train=int(overlap[i]) if domain=="thyroid" else 0))
    save_csv(output/"scores_before_labels.csv",rows)
    labels=evaluation_labels(args.images,args.thyroid,names)
    lookup={(d,n):int(y) for d in DOMAINS for n,y in zip(names[d],labels[d],strict=True)}
    for row in rows: row["label"]=lookup[(row["domain"],row["sample"])]
    save_csv(output/"test_predictions.csv",rows)
    metric_rows=[]
    for domain in DOMAINS:
        methods=list(dict.fromkeys(r["method"] for r in rows if r["domain"]==domain))
        for scope in (["official","no_train_overlap"] if domain=="thyroid" else ["official"]):
            for method in methods:
                group=[r for r in rows if r["domain"]==domain and r["method"]==method
                       and (scope=="official" or not r["overlaps_train"])]
                value=metrics([r["label"] for r in group],[r["score"] for r in group],[r["prediction"] for r in group])
                metric_rows.append(dict(domain=domain,scope=scope,method=method,alpha=group[0]["alpha"],**value))
                print(f"{domain:8} {scope:16} {method:21} AUROC={value['auroc']:.4f} AP={value['average_precision']:.4f} "
                      f"F1={value['f1']:.4f} FP={value['fp']} FN={value['fn']}",flush=True)
    save_csv(output/"metrics.csv",metric_rows)
    # Evaluation-only diagnostics: these cannot change fitted parameters,
    # frozen thresholds, scores or predictions written above.
    fit_idx,_,_=split_normal(raw_train,42,.2,deduplicate=True)
    proper_normal=raw_train[fit_idx]
    errors=[]
    primary_rows={(r["domain"],r["sample"]):r for r in rows if r["method"]=="improved"}
    for domain in DOMAINS:
        for i,name in enumerate(names[domain]):
            row=primary_rows[(domain,name)]
            if row["prediction"]==row["label"]:
                continue
            item=dict(domain=domain,sample=name,label=row["label"],prediction=row["prediction"],
                      score=row["score"],threshold=row["threshold"])
            if domain=="thyroid":
                values=raw_test[i]
                item["features_1_to_6"]=values.tolist()
                item["proper_normal_feature_percentiles"]=[float(np.mean(proper_normal[:,j]<=values[j])) for j in range(6)]
                item["normal_reference_score"]=float(normal_models[domain].scores(values[None,:])[0])
                item["normal_reference_calibration_percentile"]=float(np.mean(
                    normal_models[domain].calibration<=item["normal_reference_score"]))
            errors.append(item)
    save_json(output/"error_analysis.json",dict(scope="evaluation only; never used by fitting or prediction",
        independent_test=False,errors=errors))
    save_json(output/"run_info.json",dict(python=platform.python_version(),platform=platform.platform(),
        packages={p:version(p) for p in ("numpy","scipy","scikit-learn","torch","torchvision","pillow","joblib","threadpoolctl")},
        total_seconds=time.perf_counter()-started,extraction_seconds=feature_seconds,threads=args.threads,
        batch_size=args.batch_size,features_sha256=sha256(output/"features.npz"),
        predictions_sha256=sha256(output/"test_predictions.csv"),thyroid_train_duplicates=len(raw_train)-len(np.unique(raw_train,axis=0)),
        thyroid_test_overlap=int(overlap.sum()),used_bad_training_images=100))
    print(f"Output: {output}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images",type=Path,default=HERE.parent.parent/"Image_Anomaly_Detection.zip")
    parser.add_argument("--thyroid",type=Path,default=HERE.parent.parent/"thyroid.zip")
    parser.add_argument("--output",type=Path,default=Path(tempfile.gettempdir())/"course_extension1"/
                        datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--weights",type=Path)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--batch-size",type=int,default=8)
    args=parser.parse_args()
    if args.threads<1 or args.batch_size<1: parser.error("threads and batch-size must be positive")
    with threadpool_limits(limits=args.threads): experiment(args)


if __name__=="__main__":
    main()

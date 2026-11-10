"""Verify version-4 refined normal-reference runs and their original inputs.

The run directory is always explicit. Model loading is intended only for trusted
local joblib files. No verification output is written into the source directory.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from course_data import DOMAINS, evaluation_labels, labelled_image_members, sha256, thyroid_features
from normal_bags import NormalBagDetector
from power_reference import PowerPrototypeReference
from reference_ensemble import ReferenceEnsemble
from supervised_refinement import SupervisedNormalRefinement

HERE = Path(__file__).resolve().parent
ATOL = 1e-6


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, atol=ATOL):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, f"{message}: shape {a.shape} != {b.shape}")
    require(np.allclose(a, b, rtol=0, atol=atol, equal_nan=False), message)
    if not a.size:
        return 0.
    # Equal infinities are useful for boundary-threshold verification.
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def keyed(rows):
    result = {(r["domain"], r["sample"], r["method"]): r for r in rows}
    require(len(result) == len(rows), "Duplicate domain/sample/method rows")
    return result


def threshold(cal, alpha):
    rank = math.ceil((len(cal) + 1) * (1 - alpha))
    return float(sorted(cal)[rank - 1]) if rank <= len(cal) else float("inf")


def p_values(scores, cal):
    return np.array([(1 + np.count_nonzero(cal >= s)) / (len(cal) + 1) for s in scores])


def split_indices(train, seed, fraction, deduplicate):
    if deduplicate:
        first = {}
        for i, row in enumerate(train):
            first.setdefault(tuple(row), i)
        keep = np.array(sorted(first.values()))
    else:
        keep = np.arange(len(train))
    order = np.random.default_rng(seed).permutation(keep)
    n_cal = math.ceil(len(keep) * fraction)
    return np.sort(order[n_cal:]), np.sort(order[:n_cal]), keep


def bag_parameters(config, domain, seed=None, max_bank=None):
    return dict(metric=config["thyroid_metric"] if domain == "thyroid" else config["image_metric"],
                alpha=config["thyroid_normal_reference_alpha"] if domain == "thyroid" else config["normal_image_alpha"],
                k=config["k"],max_bank=config["max_bank"] if max_bank is None else max_bank,
                top_fraction=config["top_fraction"],seed=config["seed"] if seed is None else seed)


def metric_from_normal(fit_bags, parameters):
    x = np.asarray(fit_bags, dtype=np.float64)
    if x.ndim == 2: x=x[:,None,:]
    metric=parameters["metric"]
    flat = x.reshape(-1, x.shape[-1])
    scaler = RobustScaler() if metric == "mcd_whiten" else StandardScaler()
    z = scaler.fit_transform(flat)
    center, whitening = np.zeros(x.shape[-1]), np.eye(x.shape[-1])
    covariance_array,floor=None,None
    if metric == "mcd_whiten":
        covariance = MinCovDet(support_fraction=.75, random_state=parameters["seed"]).fit(z)
        center = covariance.location_
        covariance_array=covariance.covariance_
        eigenvalues, eigenvectors = np.linalg.eigh(covariance.covariance_)
        floor = max(float(eigenvalues.max()) * 1e-6, 1e-12)
        whitening = eigenvectors / np.sqrt(np.maximum(eigenvalues, floor))[None, :]
    full_reference = (z - center) @ whitening
    if len(full_reference) > parameters["max_bank"]:
        selected = np.sort(np.random.default_rng(parameters["seed"]).choice(len(full_reference), parameters["max_bank"], replace=False))
    else:
        selected = np.arange(len(full_reference))
    return (scaler,center,whitening,full_reference[selected]),dict(shape=x.shape,selected=selected,
        covariance=covariance_array,eigenvalue_floor=floor)


def rebuild_bag_metric(model, fit_bags, parameters):
    require(type(model) is NormalBagDetector, "Local references must use NormalBagDetector")
    require(model.distance=="euclidean" and model.reference_mode=="samples",
            "Legacy and image normal banks must preserve Euclidean sample references")
    reconstructed,details=metric_from_normal(fit_bags,parameters)
    scaler,center,whitening,reference=reconstructed
    xshape=details["shape"]; selected=details["selected"]
    require(tuple(model.fit_shape)==xshape and model.dimension==xshape[-1],"Stored fit shape differs")
    for field,expected in parameters.items():
        require(getattr(model,field)==expected,f"Stored detector parameter differs: {field}")
    require(type(model.scaler) is type(scaler),"Metric scaler class differs")
    fields=("center_","scale_") if parameters["metric"]=="mcd_whiten" else ("mean_","var_","scale_")
    for field in fields:
        close(getattr(model.scaler,field),getattr(scaler,field),f"Fit-only scaler {field}",atol=1e-12)
    if details["covariance"] is None:
        require(model.covariance is None,"Standard metric unexpectedly has covariance")
        require(model.scaler.n_samples_seen_==xshape[0]*xshape[1],"Scaler sample count differs")
    else:
        close(model.covariance,details["covariance"],"Fit-only MCD covariance",atol=1e-10)
        close(model.eigenvalue_floor,details["eigenvalue_floor"],"Whitening floor",atol=1e-12)
    close(model.center,center,"Fit-only metric center",atol=1e-10)
    close(model.whitening,whitening,"Fit-only whitening",atol=1e-10)
    close(model.memory_indices, selected, "Deterministic bank selection", atol=0.)
    close(model.reference_owners, selected // xshape[1], "Bank owner identities", atol=0.)
    close(model.reference, reference, "Fit-only bank vectors", atol=1e-10)
    close(model.neighbors._fit_X, model.reference, "Neighbor index bank", atol=0.)
    require(model.neighbors.n_neighbors == parameters["k"], "Neighbor k differs")
    return reconstructed


def independent_bag_scores(bags, metric, k, fraction):
    """Reconstruct all distances and top pooling without model scoring methods."""
    scaler, center, whitening, reference = metric
    x = np.asarray(bags, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1])
    transformed = (scaler.transform(flat) - center) @ whitening
    search = NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=1).fit(reference)
    descriptor_scores = np.empty(len(flat))
    for start in range(0, len(flat), 512):
        distances = search.kneighbors(transformed[start:start+512], return_distance=True)[0]
        descriptor_scores[start:start+len(distances)] = distances.mean(axis=1) / math.sqrt(x.shape[-1])
    count = max(1, math.ceil(fraction * x.shape[1]))
    bag_scores = np.sort(descriptor_scores.reshape(x.shape[:2]), axis=1)[:, -count:].mean(axis=1)
    # Direct subtraction and squared sums independently audit sklearn distances.
    indices = np.sort(np.random.default_rng(2026).choice(len(flat), min(64, len(flat)), replace=False))
    manual = []
    for index in indices:
        distances = np.sqrt(np.square(reference - transformed[index]).sum(axis=1))
        manual.append(np.sort(distances)[:k].mean() / math.sqrt(x.shape[-1]))
    error = close(descriptor_scores[indices], manual, "Pure NumPy descriptor kNN audit")
    return bag_scores, dict(descriptors_checked=len(indices), max_numpy_distance_error=error,
                            top_descriptor_count=count)


def reconstruct_baseline(model, fit):
    x = np.asarray(fit, dtype=np.float64)
    require((model.k, model.retained_variance, model.max_components, model.alpha) == (5,.95,64,.05),
            "Version-1 baseline hyperparameters changed")
    scaler = StandardScaler().fit(x)
    for field in ("mean_", "var_", "scale_"):
        close(getattr(model.scaler, field), getattr(scaler, field), f"Baseline scaler {field}", atol=1e-12)
    require(model.scaler.n_samples_seen_ == len(x), "Baseline fit sample count differs")
    z = scaler.transform(x)
    pca = PCA(svd_solver="full").fit(z)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    q = min(int(np.searchsorted(cumulative, .95)+1), 64, len(pca.components_))
    require(model.q == q and model.dimension == x.shape[1], "Baseline PCA dimension differs")
    close(model.pca_mean, pca.mean_, "Baseline PCA center", atol=1e-12)
    close(model.components.T @ model.components, pca.components_[:q].T @ pca.components_[:q], "Baseline PCA projection")
    close(model.reference, (z-model.pca_mean) @ model.components.T, "Baseline fit-only reference")
    close(model.neighbors._fit_X, model.reference, "Baseline neighbor index", atol=0.)


def baseline_scores(model, x):
    z = (np.asarray(x, dtype=np.float64)-model.scaler.mean_)/model.scaler.scale_-model.pca_mean
    u = z @ model.components.T
    residual = np.square(z-u @ model.components).sum(axis=1)
    result = np.empty(len(x))
    for start in range(0, len(x), 64):
        difference = u[start:start+64,None,:]-model.reference[None,:,:]
        d2 = np.sort(np.square(difference).sum(axis=2), axis=1)[:,:model.k]
        result[start:start+len(d2)] = np.sqrt(d2+residual[start:start+len(d2),None]).mean(axis=1)/math.sqrt(model.dimension)
    return result


def metrics(rows):
    y = np.array([int(r["label"]) for r in rows])
    pred = np.array([int(r["prediction"]) for r in rows])
    score = np.array([float(r["score"]) for r in rows])
    tn,fp,fn,tp = map(int, confusion_matrix(y,pred,labels=[0,1]).ravel())
    return dict(n=len(y),normal=int((y==0).sum()),anomaly=int((y==1).sum()),tn=tn,fp=fp,fn=fn,tp=tp,
        accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),
        recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),fpr=fp/(tn+fp),
        auroc=roc_auc_score(y,score),average_precision=average_precision_score(y,score))


def classification_metrics(y,score,limit):
    prediction=np.asarray(score)>=limit
    result=metrics([dict(label=int(a),score=float(b),prediction=int(c)) for a,b,c in zip(y,score,prediction)])
    return {key:result[key] for key in ("auroc","average_precision","f1","precision","recall","tn","fp","fn","tp")}


def supervised_threshold(y,score):
    """Enumerate actual operating points independently of precision_recall_curve."""
    candidates=[]
    for value in np.unique(score):
        pred=score>=value
        tp=int(np.sum((y==1)&pred)); fp=int(np.sum((y==0)&pred)); fn=int(np.sum((y==1)&~pred))
        precision=tp/(tp+fp) if tp+fp else 0.
        # Match the declared precision/recall F1 tie rule without changing it
        # through the algebraically equivalent count formula's rounding.
        recall=tp/(tp+fn)
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        candidates.append((f1,precision,float(value)))
    return max(candidates)[2]


def check_metric_dict(record,expected,label):
    require(set(record)==set(expected),f"{label}: metric keys differ")
    for key,value in expected.items(): close(record[key],value,f"{label}/{key}",atol=1e-12)


def svm_pipeline(config,C,gamma):
    return make_pipeline(StandardScaler(),SVC(C=C,gamma=gamma,kernel="rbf",class_weight="balanced",
        cache_size=512,random_state=config["seed"]))


def verify_supervised(model,gtrain,ltrain,y,gtest,ltest,config,selection_record):
    require(type(model) is SupervisedNormalRefinement,"Image model class differs")
    for field,value in dict(seed=config["seed"],n_splits=config["image_cv_folds"],k=config["k"],
        max_bank=config["max_bank"],top_fraction=config["top_fraction"]).items():
        require(getattr(model,field)==value,f"Supervised parameter {field} differs")
    for field,key in (("C_values","image_C_values"),("gamma_values","image_gamma_values"),("svm_weights","image_svm_weights")):
        require(list(getattr(model,field))==config[key],f"Supervised grid {field} differs")
    require(model.selection==selection_record,"Selection JSON differs from saved supervised model")
    selection=model.selection
    require(model.global_dimension==gtrain.shape[1] and model.local_dimension==ltrain.shape[2],"Refinement dimensions differ")
    folds=list(StratifiedKFold(n_splits=config["image_cv_folds"],shuffle=True,random_state=config["seed"]).split(gtrain,y))
    fold_ids=np.empty(len(y),dtype=int)
    for i,(_,valid) in enumerate(folds): fold_ids[valid]=i
    close(selection["oof"]["labels"],y,"OOF labels",atol=0.)
    close(selection["oof"]["fold_ids"],fold_ids,"Stratified OOF fold assignments",atol=0.)
    close(selection["training_anomaly_prevalence"],y.mean(),"OOF anomaly prevalence",atol=0.)
    candidates=[]; all_oof=[]
    grid=[(c,g) for c in config["image_C_values"] for g in config["image_gamma_values"]]
    require(len(selection["svm_candidates"])==len(grid),"SVM candidate count differs")
    require(len(selection["oof"]["all_svm_candidate_scores"])==len(grid),"Stored SVM score grid differs")
    for cid,(C,gamma) in enumerate(grid):
        oof=np.empty(len(y)); fold_results=[]
        for i,(fit,valid) in enumerate(folds):
            svm=svm_pipeline(config,C,gamma).fit(gtrain[fit],y[fit])
            oof[valid]=svm.decision_function(gtrain[valid])
            fold_results.append(dict(fold=i,n=len(valid),positives=int(y[valid].sum()),
                auroc=float(roc_auc_score(y[valid],oof[valid])),average_precision=float(average_precision_score(y[valid],oof[valid]))))
        close(selection["oof"]["all_svm_candidate_scores"][cid],oof,"Fold-trained SVM OOF scores")
        record=selection["svm_candidates"][cid]
        require(record["candidate_id"]==cid and record["C"]==C and record["gamma"]==gamma,"SVM grid order differs")
        require(len(record["folds"])==len(fold_results),"SVM fold records differ")
        for actual,expected in zip(record["folds"],fold_results): check_metric_dict(actual,expected,"SVM fold record")
        mean_ap=float(np.mean([r["average_precision"] for r in fold_results]))
        mean_auroc=float(np.mean([r["auroc"] for r in fold_results]))
        limit=supervised_threshold(y,oof)
        close(record["mean_fold_ap"],mean_ap,"Mean fold AP",atol=1e-12)
        close(record["mean_fold_auroc"],mean_auroc,"Mean fold AUROC",atol=1e-12)
        close(record["oof_threshold"],limit,"SVM OOF threshold")
        check_metric_dict(record["oof_metrics"],classification_metrics(y,oof,limit),"SVM OOF operating point")
        candidates.append((mean_ap,mean_auroc,-cid));all_oof.append(oof)
    selected_id=-max(candidates)[2]
    require(selection["svm_selected"]==selection["svm_candidates"][selected_id],"SVM selection criterion differs")
    svm_oof=all_oof[selected_id]
    close(selection["oof"]["selected_svm_scores"],svm_oof,"Selected SVM OOF scores")
    selected_svm=selection["svm_selected"]
    fitted=svm_pipeline(config,selected_svm["C"],selected_svm["gamma"]).fit(gtrain,y)
    require(list(model.svm_model.named_steps)==["standardscaler","svc"],"SVM pipeline schema differs")
    for field in ("mean_","var_","scale_"):
        close(getattr(model.svm_model[0],field),getattr(fitted[0],field),f"All-labelled-training SVM scaler {field}",atol=1e-12)
    require(model.svm_model[0].n_samples_seen_==len(gtrain),"Final SVM fitted sample count differs")
    for field in ("support_","support_vectors_","dual_coef_","intercept_","classes_","_gamma"):
        close(getattr(model.svm_model[1],field),getattr(fitted[1],field),f"Final SVM refit {field}")
    local_oof=np.empty(len(y)); local_checks=[]
    require(len(selection["local_folds"])==len(folds),"Local CV record count differs")
    parameters=bag_parameters(config,"image")
    for i,(fit,valid) in enumerate(folds):
        print(f"  [OOF] local fold {i+1}/{len(folds)}",flush=True)
        normal_fit=fit[y[fit]==0]
        require(not set(normal_fit)&set(valid),"Validation image entered fold normal bank")
        record=selection["local_folds"][i]
        require(record["fold"]==i,"Local fold index differs")
        close(record["fit_normal_indices"],normal_fit,"Fold normal owners",atol=0.)
        close(record["validation_indices"],valid,"Fold validation owners",atol=0.)
        metric,_=metric_from_normal(ltrain[normal_fit],parameters)
        require(record["reference_descriptors"]==len(metric[-1]),"Fold memory-bank count differs")
        values,audit=independent_bag_scores(ltrain[valid],metric,config["k"],config["top_fraction"])
        local_oof[valid]=values;local_checks.append(audit)
    close(selection["oof"]["local_scores"],local_oof,"Independent local OOF scores")
    normalized={}
    for name,values in (("local",local_oof),("svm",svm_oof)):
        q1,median,q3=np.quantile(values[y==0],[.25,.5,.75],method="linear")
        expected=dict(median=float(median),iqr=float(q3-q1),q1=float(q1),q3=float(q3),normal_count=int((y==0).sum()))
        require(expected["iqr"]>0,"Invalid OOF normal scale")
        check_metric_dict(model.normalization[name],expected,"OOF normal normalization")
        require(selection["normalization"][name]==model.normalization[name],"Selection normalization differs")
        normalized[name]=(values-median)/(q3-q1)
    fusion_ranking=[]
    require(len(selection["fusion_candidates"])==len(config["image_svm_weights"]),"Fusion grid count differs")
    for i,weight in enumerate(config["image_svm_weights"]):
        values=weight*normalized["svm"]+(1-weight)*normalized["local"]
        limit=supervised_threshold(y,values); expected=classification_metrics(y,values,limit)
        record=selection["fusion_candidates"][i]
        require(record["priority"]==i and record["svm_weight"]==weight and record["local_weight"]==1-weight,"Fusion weights differ")
        close(record["oof_threshold"],limit,"Fusion OOF threshold")
        check_metric_dict(record["metrics"],expected,"Fusion OOF metrics")
        fusion_ranking.append((expected["average_precision"],expected["auroc"],-i))
    fusion_id=-max(fusion_ranking)[2]
    require(selection["fusion_selected"]==selection["fusion_candidates"][fusion_id],"Fusion selection differs")
    selected=selection["fusion_selected"]
    require(model.svm_weight==selected["svm_weight"] and model.threshold==selected["oof_threshold"],"Saved fusion working point differs")
    metric=rebuild_bag_metric(model.local_model,ltrain[y==0],parameters)
    local_test,audit=independent_bag_scores(ltest,metric,config["k"],config["top_fraction"])
    svm_test=fitted.decision_function(gtest)
    local_scale=model.normalization["local"]; svm_scale=model.normalization["svm"]
    fused=model.svm_weight*(svm_test-svm_scale["median"])/svm_scale["iqr"]+(1-model.svm_weight)*(local_test-local_scale["median"])/local_scale["iqr"]
    branches=model.branch_scores(gtest,ltest)
    for name,values in (("local",local_test),("svm",svm_test),("fusion",fused)):
        close(branches[name],values,"Loaded supervised branch versus independent reconstruction")
    prediction=model.predict(gtest,ltest)
    close(prediction["prediction"],(prediction["score"]>=model.threshold).astype(int),"Supervised >= rule",atol=0.)
    return fused,svm_test,dict(svm_candidates_retrained=len(grid),svm_folds_retrained=len(grid)*len(folds),
        local_folds_reconstructed=len(folds),validation_owners_excluded=True,
        final_normal_bank_images=int((y==0).sum()),labelled_training_images=len(y),
        svm_selected_id=selected_id,fusion_selected_id=fusion_id,normalization_reconstructed=True,
        final_local_distance_audit=audit,fold_distance_audits=local_checks)


def verify_reference_ensemble(model,fit,cal,test,config,selection_record):
    require(type(model) is ReferenceEnsemble,"Thyroid primary model class differs")
    require(tuple(model.seeds)==tuple(config["thyroid_ensemble_seeds"]),"Ensemble seeds differ")
    require(model.fraction==config["thyroid_subsample_fraction"] and model.k==config["k"]
            and model.alpha==config["thyroid_normal_reference_alpha"],"Ensemble parameters differ")
    require(model.fit_count==len(fit) and model.dimension==fit.shape[1],"Ensemble fit dimensions differ")
    count=math.floor(model.fraction*len(fit))
    require(len(model.models)==len(model.seeds)==len(model.subsample_indices)==len(model.loo_scores)==len(model.loo_scales),"Ensemble artifact lengths differ")
    require(selection_record["seeds"]==list(model.seeds),"Selection ensemble seeds differ")
    values=[]; subreports=[]
    for i,seed in enumerate(model.seeds):
        indices=np.sort(np.random.default_rng(seed).choice(len(fit),count,replace=False))
        close(model.subsample_indices[i],indices,"Normal-only deterministic ensemble subsample",atol=0.)
        close(selection_record["subsample_indices"][i],indices,"Selection subsample indices",atol=0.)
        sub=model.models[i]
        parameters=bag_parameters(config,"thyroid",seed=seed,max_bank=max(8192,count))
        metric=rebuild_bag_metric(sub,fit[indices],parameters)
        reference=metric[-1]
        require(len(reference)==count,"Ensemble model omitted normal reference rows")
        loo=np.empty(count)
        for start in range(0,count,64):
            stop=min(start+64,count)
            d2=np.square(reference[start:stop,None,:]-reference[None,:,:]).sum(axis=2)
            d2[np.arange(stop-start),np.arange(start,stop)]=np.inf
            smallest=np.partition(d2,model.k-1,axis=1)[:,:model.k]
            loo[start:stop]=np.sqrt(smallest).mean(axis=1)/np.sqrt(fit.shape[1])
        close(model.loo_scores[i],loo,"Direct leave-own-index-out kNN scores")
        scale=float(np.median(loo))
        require(scale>0 and np.isfinite(scale),"Invalid LOO median scale")
        close(model.loo_scales[i],scale,"Normal-fit LOO median scale")
        close(selection_record["loo_scales"][i],model.loo_scales[i],"Selection LOO scale",atol=0.)
        combined=np.concatenate((cal,test))[:,None,:]
        score,audit=independent_bag_scores(combined,metric,model.k,config["top_fraction"])
        values.append(score/scale)
        subreports.append(dict(seed=seed,normal_reference_rows=count,loo_score_count=len(loo),
            loo_median_scale=scale,distance_audit=audit))
    combined=np.mean(values,axis=0)
    cal_scores,test_scores=combined[:len(cal)],combined[len(cal):]
    close(model.calibration,cal_scores,"Ensemble independent calibration scores")
    close(model.scores(test),test_scores,"Ensemble independent final scores")
    require(np.asarray(model.calibration).shape==(len(cal),),"Ensemble calibration must be row-level")
    return cal_scores,test_scores,dict(models=len(subreports),proper_fit_rows=len(fit),
        final_calibration_rows=len(cal),calibration_excluded_from_subsampling=True,submodels=subreports)


def independent_yeo_johnson(values,lambdas):
    """Evaluate the declared transformation without estimator.transform."""
    x=np.asarray(values,dtype=np.float64)
    transformed=np.empty_like(x)
    for j,power in enumerate(lambdas):
        nonnegative=x[:,j]>=0
        positive=np.log1p(x[nonnegative,j])
        negative=np.log1p(-x[~nonnegative,j])
        transformed[nonnegative,j]=(positive if abs(power)<1e-12 else np.expm1(power*positive)/power)
        transformed[~nonnegative,j]=(-negative if abs(power-2)<1e-12 else
                                     -np.expm1((2-power)*negative)/(2-power))
    return transformed


def verify_power_reference(model,fit,cal,test,config,selection_record):
    require(type(model) is PowerPrototypeReference,"Thyroid power-reference class differs")
    columns=tuple(i-1 for i in config["thyroid_features_1based"])
    require(tuple(model.features)==columns and model.alpha==config["thyroid_alpha"],
            "Power-reference feature indices or alpha differ")
    require(tuple(model.directions)==(1,-1),"Frozen directional feature geometry differs")
    require(model.dimension==fit.shape[1]==6 and model.fit_count==len(fit),
            "Power-reference input dimensions differ")
    require(model.selection==selection_record,"Power-reference selection JSON differs from saved model")
    power=PowerTransformer(method=config["thyroid_power_method"],standardize=False).fit(fit[:,columns])
    require(type(model.power) is PowerTransformer and model.power.method==power.method
            and model.power.standardize is False,"Power transformer configuration differs")
    close(model.power.lambdas_,power.lambdas_,"Normal-fit-only Yeo-Johnson lambdas",atol=1e-10)
    require(model.power.n_features_in_==len(columns),"Power-transform input feature count differs")
    transformed_fit=independent_yeo_johnson(fit[:,columns],power.lambdas_)
    close(transformed_fit,power.transform(fit[:,columns]),"Independent Yeo-Johnson formula",atol=1e-10)
    quantiles=np.quantile(transformed_fit,np.asarray(config["thyroid_quantile_range"])/100,axis=0)
    center=np.median(transformed_fit,axis=0)
    scale=quantiles[1]-quantiles[0]
    require(np.all(scale>0),"Expected nonconstant selected normal features")
    require(type(model.scaler) is RobustScaler and tuple(model.scaler.quantile_range)==tuple(config["thyroid_quantile_range"])
            and model.scaler.with_centering and model.scaler.with_scaling and not model.scaler.unit_variance,
            "Power-reference robust scaling configuration differs")
    close(model.scaler.center_,center,"Normal-fit-only transformed medians",atol=1e-10)
    close(model.scaler.scale_,scale,"Normal-fit-only transformed quantile scales",atol=1e-10)
    fitted=(transformed_fit-center)/scale
    fit_descriptors=(np.maximum(fitted[:,0],0)*np.maximum(-fitted[:,1],0))[:,None]
    prototype=np.median(fit_descriptors,axis=0,keepdims=True)
    close(prototype,[[0.]],"Normal directional product median is zero",atol=0.)
    core=model.core
    require(type(core) is NormalBagDetector and core.metric=="identity"
            and core.distance=="manhattan" and core.reference_mode=="median"
            and core.k==1 and core.alpha==model.alpha,"Shared power-reference core configuration differs")
    require(tuple(core.fit_shape)==(len(fit),1,1) and core.dimension==1,
            "Prototype core fit dimensions differ")
    require(core.seed==config["seed"] and core.max_bank==config["max_bank"]
            and core.top_fraction==config["top_fraction"],"Prototype core defaults differ")
    require(type(core.scaler) is StandardScaler and not core.scaler.with_mean
            and not core.scaler.with_std and core.scaler.n_samples_seen_==len(fit),
            "Prototype identity scaler differs")
    close(core.center,np.zeros(1),"Identity metric zero center",atol=0.)
    close(core.whitening,np.eye(1),"Identity metric whitening",atol=0.)
    require(core.covariance is None,"Identity metric unexpectedly fits covariance")
    close(core.memory_indices,np.empty(0,dtype=int),"Median reference does not claim one selected source row",atol=0.)
    close(core.reference_owners,[-1],"Median reference aggregate-owner marker",atol=0.)
    close(core.reference,prototype,"Normal-only median prototype",atol=1e-10)
    close(core.neighbors._fit_X,prototype,"Prototype neighbor index",atol=1e-10)
    require(core.neighbors.n_neighbors==1 and core.neighbors.effective_metric_=="manhattan",
            "Prototype nearest-neighbor distance differs")
    combined=np.concatenate((cal,test))
    coordinates=(independent_yeo_johnson(combined[:,columns],power.lambdas_)-center)/scale
    close(model.standardized_coordinates(combined),coordinates,"Independent standardized power coordinates",atol=1e-10)
    descriptors=(np.maximum(coordinates[:,0],0)*np.maximum(-coordinates[:,1],0))[:,None]
    close(model.transform(combined),descriptors,"Independent directional-product adapter",atol=1e-10)
    close(core.transform(descriptors),descriptors,"Identity metric preserves product descriptors",atol=1e-12)
    independent_scores=np.abs(descriptors-prototype).sum(axis=1)
    close(core.scores(descriptors),independent_scores,
          "Shared normal-reference scoring equals independent product-prototype distance")
    close(model.scores(combined),independent_scores,"Independent primary row scores")
    cal_scores,test_scores=independent_scores[:len(cal)],independent_scores[len(cal):]
    close(model.calibration,cal_scores,"Power-reference independent normal calibration")
    require(np.asarray(model.calibration).shape==(len(cal),),"Power-reference calibration must be row-level")
    close(model.threshold,threshold(model.calibration,model.alpha),"Power-reference normal-only rank threshold",atol=0.)
    return cal_scores,test_scores,dict(normal_fit_rows=len(fit),normal_calibration_rows=len(cal),
        features_1based=config["thyroid_features_1based"],normal_only_transform_rebuilt=True,
        normal_only_quantile_scale_rebuilt=True,normal_prototype_rebuilt=True,
        shared_core_class="NormalBagDetector",distance="manhattan",reference_vectors=len(prototype),
        score="positive standardized feature 2 times negative standardized feature 6",
        descriptor_dimension=1,score_normalization="one-dimensional normal-reference distance",
        development_selection_was_test_informed=True,independent_test=False)


def verify(args):
    folder=args.run.resolve()
    frozen,protocol,info,names=(read_json(folder/f) for f in
        ("frozen.json","protocol.json","run_info.json","sample_names.json"))
    config=frozen["config"]
    require(config["version"]==4,"This verifier requires a version-4 run")
    require(protocol["config"]==config,"Frozen and pre-extraction configurations differ")
    require(protocol["archives_sha256"]==frozen["archives_sha256"],"Recorded archive hashes differ")
    require(config["mcd_support_fraction"]==.75,"Unexpected MCD support fraction")
    require(config["thyroid_alpha"]==.005 and config["thyroid_normal_reference_alpha"]==.01
            and config["thyroid_sensitivity_alphas"]==[.01,.02,.05],"Thyroid operating points changed")
    require(config["normal_image_alpha"]==.05 and config["baseline_alpha"]==.05,"Normal baseline operating points changed")
    require(config["thyroid_ensemble_seeds"]==[42,43,44,45,46] and config["thyroid_subsample_fraction"]==.9,"Ensemble protocol changed")
    require(config["test_labels_used_for_model_selection"] is True
            and config["test_labels_used_for_fitting"] is False
            and config["independent_test_available"] is False,
            "Exploratory test-informed selection must remain explicitly disclosed")
    require(config["thyroid_features_1based"]==[2,6] and config["thyroid_power_method"]=="yeo-johnson"
            and config["thyroid_quantile_range"]==[10.,90.] and config["thyroid_distance"]=="manhattan"
            and config["thyroid_reference"]=="normal median product prototype"
            and config["thyroid_score"]=="max(z2,0) * max(-z6,0)",
            "Power-prototype frozen feature geometry differs")
    for kind,path in (("images",args.images),("thyroid",args.thyroid)):
        require(sha256(path)==frozen["archives_sha256"][kind],f"Original archive hash differs: {kind}")
    for filename,field in (("features.npz","features_sha256"),("test_predictions.csv","predictions_sha256")):
        require(sha256(folder/filename)==info[field],f"File hash differs: {filename}")
    projection=(np.random.default_rng(config["projection_seed"]).standard_normal((384,config["image_patch_dim"]))
                /np.sqrt(config["image_patch_dim"])).astype(np.float32)
    require(hashlib.sha256(projection.tobytes()).hexdigest()==frozen["projection_sha256"],"Fixed projection hash differs")
    require(config["image_size"]==256 and config["image_grid"]==16 and config["image_patch_dim"]==64,"Unexpected image dimensions")
    expected_features={f"{d}_{kind}_{split}" for d in DOMAINS for kind in ("bag","global") for split in ("train","test")}
    expected_features.update(f"{d}_train_labels" for d in DOMAINS)
    with np.load(folder/"features.npz",allow_pickle=False) as cache:
        require(set(cache.files)==expected_features,"Feature cache keys differ")
        features={key:cache[key] for key in expected_features}
    before,after=(keyed(read_csv(folder/f)) for f in ("scores_before_labels.csv","test_predictions.csv"))
    require(before.keys()==after.keys(),"Prediction identities changed after labels")
    for key,row in before.items():
        require("label" not in row,"Pre-label prediction file contains label")
        require(all(after[key][column]==value for column,value in row.items()),"Prediction changed after reading labels")
    image_paths=labelled_image_members(args.images)
    for domain in DOMAINS[:2]:
        expected_train,expected_y,expected_test=image_paths[domain]
        require(names["train"][domain]==expected_train and names["test"][domain]==expected_test,"Image identities differ")
        close(features[f"{domain}_train_labels"],expected_y,"Training-directory labels",atol=0.)
        require(len(expected_train)==250 and all("/train/good/" in n for n in expected_train[:200])
                and all("/train/bad/" in n for n in expected_train[200:]),"Training order must be 200 good then 50 bad")
    raw_thyroid=thyroid_features(args.thyroid)
    for split,matrix in zip(("train","test"),raw_thyroid):
        close(features[f"thyroid_global_{split}"],matrix,f"Original thyroid {split}",atol=0.)
        close(features[f"thyroid_bag_{split}"],matrix[:,None,:],f"Singleton bags {split}",atol=0.)
        require(names[split]["thyroid"]==[f"{split}_row_{i+1:04d}" for i in range(len(matrix))],"Thyroid row identities differ")
    close(features["thyroid_train_labels"],np.zeros(len(raw_thyroid[0])),"Only-normal thyroid training labels",atol=0.)
    manifests=read_csv(folder/"split_manifest.csv")
    calibration_rows=keyed(read_csv(folder/"calibration_scores.csv"))
    require(len(manifests)==sum(len(names["train"][d]) for d in DOMAINS),"Split manifest size differs")
    expected_prediction_keys,expected_calibration_keys,expected_metric_keys=set(),set(),set()
    report=dict(status="passed",version=4,python=platform.python_version(),config=config,
        archives_sha256=frozen["archives_sha256"],weights_sha256=frozen["weights_sha256"],
        projection_sha256=frozen["projection_sha256"],domains={})
    for domain in DOMAINS:
        print(f"[verify] {domain}: independent normal, baseline and primary models",flush=True)
        train,test=features[f"{domain}_bag_train"],features[f"{domain}_bag_test"]
        gtrain,gtest=features[f"{domain}_global_train"],features[f"{domain}_global_test"]
        y=features[f"{domain}_train_labels"].astype(int)
        for split,bag,glob in (("train",train,gtrain),("test",test,gtest)):
            require(len(bag)==len(glob)==len(names[split][domain]),"Sample counts differ")
            require(np.isfinite(bag).all() and np.isfinite(glob).all(),"Non-finite cache feature")
            require(bag.shape[1:]==((1,6) if domain=="thyroid" else (256,64)),"Bag dimensions differ")
            require(glob.shape[1]==(6 if domain=="thyroid" else 1792),"Global dimensions differ")
        normal_indices=np.flatnonzero(y==0)
        fi,ci,kept=split_indices(gtrain[normal_indices],config["seed"],config["calibration_fraction"],domain=="thyroid")
        fit,cal=normal_indices[fi],normal_indices[ci]
        fit_set,cal_set=set(fit),set(cal)
        require(not fit_set&cal_set and np.all(y[fit]==0) and np.all(y[cal]==0),"Invalid normal-only split")
        fit_vectors={tuple(x) for x in gtrain[fit]}
        require(not any(tuple(x) in fit_vectors for x in gtrain[cal]),"Duplicate crosses normal fit/calibration")
        manifest=[r for r in manifests if r["domain"]==domain]
        require([r["sample"] for r in manifest]==names["train"][domain],"Split identities differ")
        normal_roles=["fit" if i in fit_set else "calibration" if i in cal_set else "not_used" for i in range(len(train))]
        primary_roles=["supervised_cv_training"]*len(train) if domain!="thyroid" else normal_roles
        require([r["normal_reference_role"] for r in manifest]==normal_roles,"Normal split roles differ")
        require([r["primary_role"] for r in manifest]==primary_roles,"Primary split roles differ")
        close([int(r["training_label"]) for r in manifest],y,"Manifest training labels",atol=0.)
        detail=frozen["domains"][domain]
        expected_kind="power_reference" if domain=="thyroid" else "supervised_reference"
        require(detail["kind"]==expected_kind,"Frozen model kind differs")
        require((detail["training_rows"],detail["normal_training_rows"],detail["abnormal_training_rows"],
                 detail["retained_normal_rows"],detail["normal_fit"],detail["normal_calibration"])==
                (len(train),len(normal_indices),int(y.sum()),len(kept),len(fit),len(cal)),"Frozen split counts differ")
        path=folder/f"{domain}.joblib"
        require(sha256(path)==detail["model_sha256"],"Model file hash differs")
        payload=joblib.load(path)
        require(payload["version"]==4 and payload["kind"]==expected_kind and payload["domain"]==domain
                and payload["config"]==config,"Model metadata differs")
        for field in ("weights_sha256","projection_sha256"):
            require(payload[field]==frozen[field],f"Model {field} differs")
        model,normal,baseline=payload["model"],payload["normal_model"],payload["baseline"]
        previous=payload["previous_model"]
        metric=rebuild_bag_metric(normal,train[fit],bag_parameters(config,domain))
        original_owners=fit[normal.reference_owners]
        require(set(original_owners).issubset(fit_set) and not set(original_owners)&cal_set,"Normal bank contains held-out owner")
        combined=np.concatenate((train[cal],test))
        normal_combined,distance_audit=independent_bag_scores(combined,metric,config["k"],config["top_fraction"])
        normal_cal,normal_test=normal_combined[:len(cal)],normal_combined[len(cal):]
        close(normal.calibration,normal_cal,"Normal-reference independent calibration")
        close(normal.scores(test),normal_test,"Normal-reference independent test scores")
        require(normal.calibration.shape==(len(cal),),"Normal calibration must be observation-level")
        reconstruct_baseline(baseline,gtrain[fit])
        baseline_cal,baseline_test=baseline_scores(baseline,gtrain[cal]),baseline_scores(baseline,gtest)
        close(baseline.calibration["knn_residual"],baseline_cal,"Baseline independent calibration")
        close(baseline.scores(gtest)["knn_residual"],baseline_test,"Baseline independent test scores")
        selection_record=read_json(folder/f"{domain}_selection.json")
        require(read_json(folder/f"{domain}_previous_selection.json")==payload["previous_selection"],
                "Previous-v3 selection JSON differs from saved model metadata")
        calibration_groups=[("baseline",baseline_cal,baseline.calibration["knn_residual"]),
                            ("normal_reference",normal_cal,normal.calibration)]
        if domain=="thyroid":
            primary_cal,primary_test,primary_audit=verify_power_reference(
                model,gtrain[fit],gtrain[cal],gtest,config,selection_record)
            previous_cal,previous_test,previous_audit=verify_reference_ensemble(
                previous,gtrain[fit],gtrain[cal],gtest,config,payload["previous_selection"])
            calibration_groups.append(("improved",primary_cal,model.calibration))
            calibration_groups.append(("previous_v3",previous_cal,previous.calibration))
            require(detail["svm_weight"] is None,"Thyroid unexpectedly stores an SVM weight")
        else:
            primary_config=dict(config)
            for name,value in config["image_primary_parameters"][domain].items():
                primary_config["image_"+name]=value
            primary_test,svm_test,primary_audit=verify_supervised(
                model,gtrain,train,y,gtest,test,primary_config,selection_record)
            if previous is model:
                require(all(primary_config[key]==config[key] for key in
                        ("image_C_values","image_gamma_values","image_svm_weights")),
                        "Shared primary/previous object has different declared grids")
                require(payload["previous_selection"]==selection_record,
                        "Shared primary/previous object has different selection records")
                previous_test=primary_test
                previous_audit=dict(primary_audit,reuses_primary_verified_object=True)
            else:
                previous_test,_,previous_audit=verify_supervised(
                    previous,gtrain,train,y,gtest,test,config,payload["previous_selection"])
            require(detail["svm_weight"]==model.svm_weight,"Frozen SVM fusion weight differs")
        recorded_cal={}
        for method,independent,stored in calibration_groups:
            collected=[]
            for index in cal:
                key=(domain,names["train"][domain][index],method)
                expected_calibration_keys.add(key)
                require(key in calibration_rows,"Missing calibration CSV row")
                collected.append(float(calibration_rows[key]["score"]))
            recorded_cal[method]=np.asarray(collected)
            close(collected,stored,f"{domain}/{method}: saved model calibration equals CSV",atol=0.)
            close(collected,independent,f"{domain}/{method}: independently recomputed calibration")
        variants=[
            dict(method="baseline",alpha=.05,scores=baseline_test,calibration="baseline",rule=">"),
            dict(method="normal_reference",alpha=normal.alpha,scores=normal_test,calibration="normal_reference",rule=">")]
        if domain=="thyroid":
            variants.extend(dict(method=method,alpha=alpha,scores=primary_test,calibration="improved",rule=">")
                for method,alpha in (("improved",.005),("improved_alpha_0.01",.01),
                                     ("improved_alpha_0.02",.02),("improved_alpha_0.05",.05)))
            variants.append(dict(method="previous_v3",alpha=previous.alpha,scores=previous_test,
                                 calibration="previous_v3",rule=">"))
        else:
            variants.extend([
                dict(method="improved",alpha="",scores=primary_test,calibration=None,rule=">=",threshold=model.threshold),
                dict(method="supervised_only",alpha="",scores=svm_test,calibration=None,rule=">=",
                     threshold=model.selection["svm_selected"]["oof_threshold"]),
                dict(method="previous_v3",alpha="",scores=previous_test,calibration=None,rule=">=",threshold=previous.threshold)])
        train_vectors={tuple(x) for x in gtrain}
        overlaps=[int(tuple(x) in train_vectors) for x in gtest] if domain=="thyroid" else [0]*len(test)
        method_report={}
        for variant in variants:
            method,alpha,scores=variant["method"],variant["alpha"],variant["scores"]
            group=[]
            for sample in names["test"][domain]:
                key=(domain,sample,method);expected_prediction_keys.add(key)
                require(key in after,f"Missing prediction row: {key}")
                group.append(after[key])
            csv_scores=np.array([float(r["score"]) for r in group])
            error=close(csv_scores,scores,f"{domain}/{method}: independent scores")
            if variant["calibration"] is None:
                limit=variant["threshold"]
                require(all(r["alpha"]=="" and r["p_normal"]=="" for r in group),"Supervised rows must omit alpha and normal p-value")
                prediction=(csv_scores>=limit).astype(int)
            else:
                cal_values=recorded_cal[variant["calibration"]]
                limit=threshold(cal_values,alpha)
                p=p_values(csv_scores,cal_values);prediction=(csv_scores>limit).astype(int)
                close([float(r["alpha"]) for r in group],np.full(len(test),alpha),"Normal calibration alpha",atol=0.)
                close([float(r["p_normal"]) for r in group],p,"Independent tie-aware normal p-values",atol=0.)
                close(prediction,(p<=alpha).astype(int),"Strict threshold / normal p decision equivalence",atol=0.)
            require(all(r["decision_rule"]==variant["rule"] for r in group),"Recorded decision rule differs")
            close([float(r["threshold"]) for r in group],np.full(len(test),limit),"Recorded threshold",atol=0.)
            close([int(r["prediction"]) for r in group],prediction,"Recorded class decision",atol=0.)
            close([int(r["overlaps_train"]) for r in group],overlaps,"Raw training overlap indicator",atol=0.)
            if method=="improved":
                close(model.threshold,limit,"Primary saved threshold",atol=0.)
                close(detail["primary_threshold"],limit,"Primary frozen threshold",atol=0.)
            elif method=="previous_v3":
                close(previous.threshold,limit,"Previous-v3 saved threshold",atol=0.)
                close(detail["previous_threshold"],limit,"Previous-v3 frozen threshold",atol=0.)
            elif method=="normal_reference":
                close(normal.threshold,limit,"Normal-reference saved threshold",atol=0.)
                close(detail["normal_threshold"],limit,"Normal-reference frozen threshold",atol=0.)
            elif method=="baseline":
                close(baseline.thresholds["knn_residual"],limit,"Baseline saved threshold",atol=0.)
                close(detail["baseline_threshold"],limit,"Baseline frozen threshold",atol=0.)
            method_report[method]=dict(alpha=alpha,decision_rule=variant["rule"],threshold=limit,
                normal_p_value_available=variant["calibration"] is not None,predictions_checked=len(test),max_score_error=error)
            for scope in (["official","no_train_overlap"] if domain=="thyroid" else ["official"]):
                expected_metric_keys.add((domain,scope,method))
        report["domains"][domain]=dict(kind=expected_kind,normal_fit=len(fit),normal_calibration=len(cal),
            normal_bank_owners_are_fit_only=True,normal_reference_vectors=len(normal.reference),
            normal_reference_distance_audit=distance_audit,primary_reconstruction=primary_audit,
            previous_v3_reconstruction=previous_audit,methods=method_report)
    require(set(after)==expected_prediction_keys,"Unexpected prediction rows")
    require(set(calibration_rows)==expected_calibration_keys,"Unexpected calibration rows")
    require(len(after)==14041 and len(expected_metric_keys)==24,"Unexpected course v4 artifact size")
    labels=evaluation_labels(args.images,args.thyroid,names["test"])
    for domain in DOMAINS:
        lookup=dict(zip(names["test"][domain],labels[domain],strict=True))
        for row in after.values():
            if row["domain"]==domain: require(int(row["label"])==int(lookup[row["sample"]]),"Official evaluation label differs")
    metric_rows=read_csv(folder/"metrics.csv")
    require(len(metric_rows)==len(expected_metric_keys) and
        {(r["domain"],r["scope"],r["method"]) for r in metric_rows}==expected_metric_keys,"Metric identities differ")
    for row in metric_rows:
        group=[r for r in after.values() if r["domain"]==row["domain"] and r["method"]==row["method"]
               and (row["scope"]=="official" or int(r["overlaps_train"])==0)]
        require(row["alpha"]==group[0]["alpha"],"Metric alpha metadata differs")
        for field,value in metrics(group).items():
            close(float(row[field]),value,f"Metric {row['domain']}/{row['scope']}/{row['method']}/{field}",atol=1e-12)
    error_record=read_json(folder/"error_analysis.json")
    require(error_record["scope"]=="evaluation only; never used by fitting or prediction"
            and error_record["independent_test"] is False,"Error diagnostics scope differs")
    expected_errors=[]
    proper_indices,cal_indices,_=split_indices(raw_thyroid[0],config["seed"],config["calibration_fraction"],True)
    proper_normal=raw_thyroid[0][proper_indices]
    thyroid_normal_cal=np.array([float(calibration_rows[("thyroid",names["train"]["thyroid"][i],"normal_reference")]["score"])
                                 for i in cal_indices])
    for domain in DOMAINS:
        for i,sample in enumerate(names["test"][domain]):
            row=after[(domain,sample,"improved")]
            if row["prediction"]==row["label"]: continue
            item=dict(domain=domain,sample=sample,label=int(row["label"]),prediction=int(row["prediction"]),
                      score=float(row["score"]),threshold=float(row["threshold"]))
            if domain=="thyroid":
                values=raw_thyroid[1][i]
                item["features_1_to_6"]=values.tolist()
                item["proper_normal_feature_percentiles"]=[float(np.mean(proper_normal[:,j]<=values[j])) for j in range(6)]
                item["normal_reference_score"]=float(after[(domain,sample,"normal_reference")]["score"])
                item["normal_reference_calibration_percentile"]=float(np.mean(thyroid_normal_cal<=item["normal_reference_score"]))
            expected_errors.append(item)
    require(len(error_record["errors"])==len(expected_errors),"Error-analysis row count differs")
    for actual,expected in zip(error_record["errors"],expected_errors):
        require(set(actual)==set(expected),"Error-analysis fields differ")
        for key,value in expected.items():
            if isinstance(value,str): require(actual[key]==value,f"Error-analysis {key} differs")
            else: close(actual[key],value,f"Error-analysis {key} differs",atol=ATOL if key=="normal_reference_score" else 0.)
    report.update(metric_rows_checked=len(metric_rows),prediction_rows_checked=len(after),calibration_rows_checked=len(calibration_rows),
        error_analysis_rows_checked=len(expected_errors),pre_label_predictions_unchanged=True,
        thyroid_features_match_raw=True,raw_images=dict(performed=False))
    if args.raw:
        import torch
        from local_features import ImageLocalEncoder,encode_members
        torch.set_num_threads(args.threads);torch.manual_seed(config["seed"]);torch.use_deterministic_algorithms(True)
        encoder=ImageLocalEncoder(args.weights)
        for field in ("weights_sha256","projection_sha256"):
            require(getattr(encoder,field)==frozen[field],f"Loaded feature adapter {field} differs")
        batch=args.batch_size or info["batch_size"];raw_results={}
        for domain in DOMAINS[:2]:
            members=names["train"][domain]+names["test"][domain]
            print(f"[raw] {domain}: all {len(members)} original normal/abnormal/train/test images",flush=True)
            gx,lx=encode_members(args.images,members,encoder,batch)
            ge=close(gx,np.concatenate((features[f"{domain}_global_train"],features[f"{domain}_global_test"])),"Raw global features")
            le=close(lx,np.concatenate((features[f"{domain}_bag_train"],features[f"{domain}_bag_test"])),"Raw local features")
            raw_results[domain]=dict(images=len(members),max_global_feature_error=ge,max_local_feature_error=le)
        require(sum(r["images"] for r in raw_results.values())==602,"Raw validation did not cover all 602 course images")
        report["raw_images"]=dict(performed=True,batch_size=batch,total_images=602,domains=raw_results)
    if args.compare:
        other=args.compare.resolve();other_frozen=read_json(other/"frozen.json")
        for field in ("config","archives_sha256","weights_sha256","projection_sha256"):
            require(frozen[field]==other_frozen[field],f"Comparison {field} differs")
        other_rows=keyed(read_csv(other/"test_predictions.csv"))
        require(after.keys()==other_rows.keys(),"Comparison prediction identities differ")
        errors={f:0. for f in ("score","threshold","p_normal")}
        for key,row in after.items():
            prior=other_rows[key]
            for field in errors:
                if row[field]=="" or prior[field]=="":
                    require(row[field]==prior[field],f"Compare missing {field}")
                else:
                    errors[field]=max(errors[field],close(float(row[field]),float(prior[field]),f"Compare {key}/{field}"))
            for field in ("alpha","decision_rule","prediction","overlaps_train","label"):
                require(row[field]==prior[field],f"Compare {key}/{field}")
        report["compare"]=dict(directory=str(other),rows=len(after),absolute_tolerance=ATOL,max_errors=errors,identical_predictions=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True,help="Explicit version-4 run directory")
    parser.add_argument("--images",type=Path,default=HERE.parent.parent/"Image_Anomaly_Detection.zip")
    parser.add_argument("--thyroid",type=Path,default=HERE.parent.parent/"thyroid.zip")
    parser.add_argument("--raw",action="store_true",help="Re-extract all original image global/local features")
    parser.add_argument("--compare",type=Path)
    parser.add_argument("--weights",type=Path)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--batch-size",type=int)
    args=parser.parse_args()
    if args.threads<1 or (args.batch_size is not None and args.batch_size<1):
        parser.error("threads and batch-size must be positive")
    started=time.perf_counter()
    try:
        with threadpool_limits(limits=args.threads):
            report=verify(args)
    except Exception as error:
        report=dict(status="failed",version=4,error_type=type(error).__name__,error=str(error),
            verified_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
        if args.run.is_dir():
            (args.run/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        raise
    report.update(verified_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
    (args.run/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(dict(status=report["status"],metric_rows=report["metric_rows_checked"],
        prediction_rows=report["prediction_rows_checked"],raw_images=report["raw_images"]["performed"],seconds=report["elapsed_seconds"])))


if __name__=="__main__":
    main()

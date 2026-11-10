"""Joblib-compatible normal-memory model with optional labelled SVM refinement.

This module contains no data loading, test-label access or file writing.
Training OOF scores determine all SVM
and fusion choices; after full refitting the threshold is an empirical CV
working point, without a split-conformal coverage claim.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from normal_bags import NormalBagDetector


def _threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = np.divide(2*precision[:-1]*recall[:-1],precision[:-1]+recall[:-1],
        out=np.zeros(len(thresholds)),where=(precision[:-1]+recall[:-1])>0)
    best=max(range(len(thresholds)),key=lambda i:(f1[i],precision[i],thresholds[i]))
    return float(thresholds[best])


def _metrics(y, score, threshold):
    pred=score>=threshold
    tp=int(np.sum((y==1)&pred)); fp=int(np.sum((y==0)&pred))
    fn=int(np.sum((y==1)&~pred)); tn=int(np.sum((y==0)&~pred))
    return dict(auroc=float(roc_auc_score(y,score)),average_precision=float(average_precision_score(y,score)),
        f1=2*tp/(2*tp+fp+fn),precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),
        tn=tn,fp=fp,fn=fn,tp=tp)


def _normal_scale(score,y):
    q1,median,q3=np.quantile(score[y==0],[.25,.5,.75],method="linear")
    if not np.isfinite(q3-q1) or q3<=q1:
        raise ValueError("OOF normal IQR must be finite and positive")
    return dict(median=float(median),iqr=float(q3-q1),q1=float(q1),q3=float(q3),normal_count=int((y==0).sum()))


class SupervisedNormalRefinement:
    """Shared normal patch memory plus training-labelled global SVM correction.

    fit(global_train, local_train, y) returns self. Full JSON-safe CV records
    are available as .selection. fit_supervised_refinement(...) is a wrapper
    returning (model, selection) when the caller prefers tuple unpacking.
    score(global_test, local_test) returns one fused anomaly score per image.
    """

    def __init__(self, seed=42, n_splits=5, k=5, max_bank=8192, top_fraction=.02,
                 C_values=(.1,1.,10.,100.), gamma_values=("scale",.0001,.001,.01),
                 svm_weights=(.5,.25,.75)):
        self.seed,self.n_splits=seed,n_splits
        self.k,self.max_bank,self.top_fraction=k,max_bank,top_fraction
        self.C_values,self.gamma_values=tuple(C_values),tuple(gamma_values)
        self.svm_weights=tuple(svm_weights)
        if not self.svm_weights or not all(0<w<1 for w in self.svm_weights):
            raise ValueError("Every fusion candidate must retain both branches")

    def _svm(self,C,gamma):
        return make_pipeline(StandardScaler(),SVC(C=C,gamma=gamma,kernel="rbf",
            class_weight="balanced",cache_size=512,random_state=self.seed))

    def _local(self):
        return NormalBagDetector(metric="standard",k=self.k,max_bank=self.max_bank,
            top_fraction=self.top_fraction,seed=self.seed)

    @staticmethod
    def _inputs(global_x,local_x):
        global_x=np.asarray(global_x)
        local_x=np.asarray(local_x)
        if global_x.ndim!=2 or local_x.ndim!=3 or len(global_x)!=len(local_x) or not len(global_x):
            raise ValueError("Need aligned global [N,D] and local [N,P,d] arrays")
        if not np.isfinite(global_x).all() or not np.isfinite(local_x).all():
            raise ValueError("Feature arrays must be finite")
        return global_x,local_x

    def fit(self,global_train,local_train,y):
        gx,lx=self._inputs(global_train,local_train)
        y=np.asarray(y)
        if y.ndim!=1 or len(y)!=len(gx) or set(np.unique(y))!={0,1}:
            raise ValueError("y must align with samples and contain classes 0 and 1")
        y=y.astype(int)
        if min(int((y==0).sum()),int((y==1).sum()))<self.n_splits:
            raise ValueError("Each class must contain at least n_splits observations")
        self.global_dimension=gx.shape[1]
        self.local_dimension=lx.shape[2]
        folds=list(StratifiedKFold(n_splits=self.n_splits,shuffle=True,random_state=self.seed).split(gx,y))
        fold_ids=np.empty(len(y),dtype=int)
        for i,(_,valid) in enumerate(folds): fold_ids[valid]=i
        svm_candidates=[]
        all_svm_oof=[]
        for candidate_id,(C,gamma) in enumerate((c,g) for c in self.C_values for g in self.gamma_values):
            oof=np.empty(len(y))
            fold_metrics=[]
            for i,(fit,valid) in enumerate(folds):
                model=self._svm(C,gamma).fit(gx[fit],y[fit])
                oof[valid]=model.decision_function(gx[valid])
                fold_metrics.append(dict(fold=i,n=len(valid),positives=int(y[valid].sum()),
                    auroc=float(roc_auc_score(y[valid],oof[valid])),
                    average_precision=float(average_precision_score(y[valid],oof[valid]))))
            threshold=_threshold(y,oof)
            svm_candidates.append(dict(candidate_id=candidate_id,C=C,gamma=gamma,
                mean_fold_ap=float(np.mean([m["average_precision"] for m in fold_metrics])),
                mean_fold_auroc=float(np.mean([m["auroc"] for m in fold_metrics])),
                oof_threshold=threshold,oof_metrics=_metrics(y,oof,threshold),folds=fold_metrics))
            all_svm_oof.append(oof)
        svm_selected=max(svm_candidates,key=lambda c:(c["mean_fold_ap"],c["mean_fold_auroc"],-c["candidate_id"]))
        svm_oof=all_svm_oof[svm_selected["candidate_id"]]
        local_oof=np.empty(len(y))
        local_manifest=[]
        for i,(fit,valid) in enumerate(folds):
            normal_fit=fit[y[fit]==0]
            if np.intersect1d(normal_fit,valid).size:
                raise AssertionError("Validation image entered the normal bank")
            local=self._local().fit(lx[normal_fit])
            local_oof[valid]=local.scores(lx[valid])
            local_manifest.append(dict(fold=i,fit_normal_indices=normal_fit.tolist(),
                validation_indices=valid.tolist(),reference_descriptors=len(local.reference)))
        self.normalization={"local":_normal_scale(local_oof,y),"svm":_normal_scale(svm_oof,y)}
        local_z=self._scale(local_oof,"local")
        svm_z=self._scale(svm_oof,"svm")
        fusion_candidates=[]
        for priority,weight in enumerate(self.svm_weights):
            fused=weight*svm_z+(1-weight)*local_z
            threshold=_threshold(y,fused)
            fusion_candidates.append(dict(priority=priority,svm_weight=weight,local_weight=1-weight,
                oof_threshold=threshold,metrics=_metrics(y,fused,threshold)))
        selected=max(fusion_candidates,key=lambda c:(c["metrics"]["average_precision"],c["metrics"]["auroc"],-c["priority"]))
        self.svm_weight=selected["svm_weight"]
        self.threshold=selected["oof_threshold"]
        self.svm_model=self._svm(svm_selected["C"],svm_selected["gamma"]).fit(gx,y)
        self.local_model=self._local().fit(lx[y==0])
        self.selection=dict(svm_selected=svm_selected,svm_candidates=svm_candidates,
            fusion_selected=selected,fusion_candidates=fusion_candidates,normalization=self.normalization,
            local_folds=local_manifest,training_anomaly_prevalence=float(y.mean()),
            oof=dict(labels=y.tolist(),fold_ids=fold_ids.tolist(),local_scores=local_oof.tolist(),
                     selected_svm_scores=svm_oof.tolist(),all_svm_candidate_scores=[a.tolist() for a in all_svm_oof]),
            parameter_selection="SVM mean fold AP, then mean AUROC, then fixed grid order",
            fusion_selection="pooled OOF AP, then OOF AUROC, then configured weight order",
            threshold_selection="OOF max F1; ties precision then larger threshold; prediction >=",
            statistical_scope="CV empirical working point after full refit; no split-conformal guarantee")
        return self

    def _scale(self,values,branch):
        scale=self.normalization[branch]
        return (np.asarray(values)-scale["median"])/scale["iqr"]

    def branch_scores(self,global_test,local_test):
        gx,lx=self._inputs(global_test,local_test)
        if gx.shape[1]!=self.global_dimension or lx.shape[2]!=self.local_dimension:
            raise ValueError("Feature dimensions differ from training")
        local=self.local_model.scores(lx)
        svm=self.svm_model.decision_function(gx)
        fused=self.svm_weight*self._scale(svm,"svm")+(1-self.svm_weight)*self._scale(local,"local")
        return dict(local=local,svm=svm,fusion=fused)

    def score(self,global_test,local_test):
        return self.branch_scores(global_test,local_test)["fusion"]

    def predict(self,global_test,local_test):
        branches=self.branch_scores(global_test,local_test)
        return dict(score=branches["fusion"],prediction=(branches["fusion"]>=self.threshold).astype(int),
                    threshold=self.threshold,branches=branches)


def fit_supervised_refinement(global_train,local_train,y,**kwargs):
    model=SupervisedNormalRefinement(**kwargs).fit(global_train,local_train,y)
    return model,model.selection

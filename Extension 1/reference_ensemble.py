"""Fit-only subsampling ensemble of the shared normal-memory kNN detector."""
import numpy as np

from normal_bags import NormalBagDetector
from unified import conformal_threshold, normal_p_values


class ReferenceEnsemble:
    def __init__(self, alpha=.01, seeds=(42,43,44,45,46), fraction=.9, k=5):
        if not 0 < alpha < 1 or not 0 < fraction <= 1 or not seeds or k < 1:
            raise ValueError("Invalid reference ensemble configuration")
        self.alpha,self.seeds,self.fraction,self.k=alpha,tuple(seeds),fraction,k

    @staticmethod
    def _rows(values):
        x=np.asarray(values,dtype=np.float64)
        if x.ndim==3 and x.shape[1]==1: x=x[:,0,:]
        if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError("Expected finite rows or singleton descriptor bags")
        return x

    def fit(self, normal_rows):
        x=self._rows(normal_rows)
        count=int(np.floor(self.fraction*len(x)))
        if count<=self.k: raise ValueError("Too few normal rows for leave-self-out scaling")
        self.fit_count,self.dimension=x.shape
        self.models,self.subsample_indices,self.loo_scales,self.loo_scores=[],[],[],[]
        for seed in self.seeds:
            indices=np.sort(np.random.default_rng(seed).choice(len(x),count,replace=False))
            sample=x[indices]
            model=NormalBagDetector(metric="mcd_whiten",k=self.k,max_bank=max(8192,count),
                                    alpha=self.alpha,seed=seed).fit(sample)
            query=model.transform(sample)
            distances,neighbors=model.neighbors.kneighbors(query,n_neighbors=self.k+1,return_distance=True)
            loo=np.empty(count)
            for i in range(count):
                selected=distances[i][neighbors[i]!=i][:self.k]
                if len(selected)!=self.k: raise ValueError("Leave-self-out neighbor count mismatch")
                loo[i]=selected.mean()/np.sqrt(self.dimension)
            scale=float(np.median(loo))
            if not np.isfinite(scale) or scale<=0: raise ValueError("Invalid leave-self-out scale")
            self.models.append(model); self.subsample_indices.append(indices)
            self.loo_scores.append(loo); self.loo_scales.append(scale)
        self.loo_scales=np.asarray(self.loo_scales)
        return self

    def scores(self, rows):
        x=self._rows(rows)
        if x.shape[1]!=self.dimension: raise ValueError("Feature dimension mismatch")
        return np.mean([model.scores(x)/scale for model,scale in zip(self.models,self.loo_scales)],axis=0)

    def calibrate(self, normal_rows):
        self.calibration=self.scores(normal_rows)
        self.threshold=conformal_threshold(self.calibration,self.alpha)
        return self

    def predict(self, rows, alpha=None):
        values=self.scores(rows)
        threshold=self.threshold if alpha is None else conformal_threshold(self.calibration,alpha)
        return dict(score=values,p_normal=normal_p_values(values,self.calibration),
                    prediction=(values>threshold).astype(int),threshold=threshold)

"""Render eight report figures from synchronized experiment data.

Run sync_results.py --run RUN_DIRECTORY first. No fitting or inference occurs.
Saved response arrays provide the actual network and local distance values.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import PercentFormatter
from PIL import Image

BASE=Path(__file__).resolve().parent
OUT=BASE.parent/"figures"
SRC=BASE/"originals"
DATA=json.loads((BASE/"verified-results.json").read_text(encoding="utf-8"))
if DATA.get("schema_version")!=2 or not DATA["provenance"]["raw_prediction_metrics_recomputed"]:
    raise ValueError("Run sync_results.py on a completed experiment before rendering")
OUT.mkdir(exist_ok=True)
for name in ("times.ttf","timesbd.ttf","timesi.ttf","timesbi.ttf","SimHei.ttf","Simsun.ttc"):
    path=BASE.parent/"fonts"/name
    if path.exists():font_manager.fontManager.addfont(str(path))

plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
                    "font.size":9,"axes.labelsize":9,"axes.titlesize":10,
                    "xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8.5,
                    "axes.linewidth":.65,"pdf.fonttype":42,"ps.fonttype":42,
                    "savefig.facecolor":"white","mathtext.fontset":"stix"})
BLUE,GREEN,ORANGE,PURPLE="#3B6B99","#368578","#CC7044","#77539D"
INK="#243444"
COLORS={"global":BLUE,"local":ORANGE,"fusion":GREEN}
NAMES={"global":"Global SVM","local":"Local NN","fusion":"Fusion"}
STYLES={"global":(0,(5,2)),"local":(0,(1.2,1.5)),"fusion":"-"}

def save(fig,name):
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight",pad_inches=.03)
    fig.savefig(OUT/f"{name}.png",dpi=320,bbox_inches="tight",pad_inches=.03)
    plt.close(fig)

def clean(ax,grid="both"):
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(length=3,width=.65)
    ax.set_axisbelow(True)
    if grid:ax.grid(axis=grid,color="#E2E5EA",linewidth=.6)

def rows_for(category):
    return [r for r in DATA["predictions"] if r["category"]==category]

def metric_for(category,branch):
    return next(m for m in DATA["metrics"] if m["category"]==category and m["method"]==branch)

def rank_curves(y,s):
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float)
    order=np.argsort(-s,kind="stable");ys=y[order];ss=s[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
    tp=np.cumsum(ys)[ends];fp=ends+1-tp
    rec=tp/y.sum();prec=tp/(ends+1)
    return np.r_[0,fp/(len(y)-y.sum())],np.r_[0,rec],np.r_[0,rec],np.r_[1,prec]

def method():
    import runpy
    runpy.run_path(str(BASE/"build_method_figure.py"))["main"]()

def training_samples():
    entries=[("hazelnut_train_good002.jpg","Hazelnut · normal","train/good/002"),
             ("hazelnut_train_bad003.jpg","Hazelnut · anomalous","train/bad/003"),
             ("zipper_train_good002.jpg","Zipper · normal","train/good/002"),
             ("zipper_train_bad006.jpg","Zipper · anomalous","train/bad/006")]
    fig,axes=plt.subplots(1,4,figsize=(7.1,2.0))
    for ax,(file,title,path) in zip(axes,entries):
        ax.imshow(Image.open(SRC/file));ax.axis("off")
        ax.set_title(title,pad=5,fontsize=9.3)
        ax.text(.5,-.055,path,transform=ax.transAxes,ha="center",va="top",fontsize=8.5)
    fig.subplots_adjust(left=.005,right=.995,bottom=.12,top=.88,wspace=.055)
    save(fig,"training-samples")

def feature_responses():
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.4))
    for ax,layer in zip(axes,(2,3,4)):
        spec=DATA["visuals"]["feature_maps"][f"layer{layer}"]
        values=np.load(BASE/spec["file"],allow_pickle=False)
        values=(values-values.min())/max(float(values.max()-values.min()),1e-8)
        shape=DATA["feature_maps"][f"layer{layer}"]
        ax.imshow(values,cmap="viridis",vmin=0,vmax=1,interpolation="nearest")
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_title(f"layer{layer}: {shape[0]} × {shape[1]} × {shape[2]:,}",pad=5,fontsize=9.5)
    fig.subplots_adjust(left=.04,right=.94,bottom=.06,top=.9,wspace=.09)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap="viridis"),
                    ax=axes,location="right",fraction=.02,pad=.025,ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=7.5,length=2);cb.outline.set_linewidth(.5)
    save(fig,"feature-responses")

def roc_pr():
    cats=DATA["categories"]
    fig,axes=plt.subplots(len(cats),2,figsize=(7.1,2.4*len(cats)),squeeze=False)
    for row,cat in enumerate(cats):
        records=rows_for(cat);y=np.array([r["label"] for r in records])
        # Distinct dashes remain visible when curves coincide exactly.
        # No display jitter changes any score or curve coordinate.
        for branch in ("fusion","global","local"):
            scores=np.array([r[f"{branch}_score"] for r in records])
            fpr,tpr,rec,prec=rank_curves(y,scores)
            m=metric_for(cat,branch)
            style=dict(color=COLORS[branch],linestyle=STYLES[branch],
                       linewidth=2.7 if branch=="fusion" else 1.25,
                       alpha=.48 if branch=="fusion" else 1)
            axes[row,0].plot(fpr,tpr,label=f"{NAMES[branch]}  {m['auroc']:.4f}",**style)
            axes[row,1].step(rec,prec,where="pre",label=f"{NAMES[branch]}  {m['ap']:.4f}",**style)
        for col,kind in enumerate(("ROC","PR")):
            ax=axes[row,col]
            ax.set_title(f"{cat.title()} · {kind} curves",pad=6)
            ax.set(xlim=(-.025,1.025),ylim=(-.025,1.035))
            ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6))
            ax.set_xlabel("False positive rate" if col==0 else "Recall")
            ax.set_ylabel("True positive rate" if col==0 else "Precision")
            clean(ax)
            handles,labels=ax.get_legend_handles_labels();order=[1,2,0]
            ax.legend([handles[i] for i in order],[labels[i] for i in order],
                      title="AUROC" if col==0 else "AP",loc="lower right" if col==0 else "lower left",
                      frameon=False,fontsize=8.2,title_fontsize=9,handlelength=2.5)
    fig.subplots_adjust(left=.09,right=.99,bottom=.09,top=.94,wspace=.27,hspace=.47)
    save(fig,"roc-pr-curves")

def confusion():
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.45))
    for ax,branch in zip(axes,DATA["branches"]):
        a=np.asarray(DATA["confusion"][branch],dtype=int)
        row_fraction=a/np.maximum(a.sum(axis=1,keepdims=True),1)
        ax.imshow(row_fraction,cmap="Blues",vmin=0,vmax=1)
        ax.set_xticks([0,1],["Normal","Anomalous"],fontsize=8)
        ax.set_yticks([0,1],["Normal","Anomalous"],fontsize=8)
        ax.set_xlabel("Predicted label",labelpad=4)
        ax.set_title(NAMES[branch],pad=7)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f"{a[i,j]}\n({row_fraction[i,j]:.2%})",ha="center",va="center",
                        fontsize=9.7,color="white" if row_fraction[i,j]>.55 else INK,linespacing=1.42)
        ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    axes[0].set_ylabel("True label")
    fig.subplots_adjust(left=.09,right=.995,bottom=.22,top=.86,wspace=.48)
    save(fig,"confusion-matrices")

def errors():
    branches=DATA["branches"]
    records=[r for r in DATA["predictions"] if any(r[f"{b}_prediction"]!=r["label"] for b in branches)]
    records.sort(key=lambda r:(r["category"],r["path"]))
    fig,ax=plt.subplots(figsize=(3.45,max(1.65,.38*len(records)+.65)))
    if not records:
        ax.axis("off");ax.text(.5,.5,"All branches classify every test image correctly.",ha="center",va="center",wrap=True)
    else:
        values=np.array([[0 if r[f"{b}_prediction"]==r["label"] else (1 if r["label"]==0 else 2)
                          for b in branches] for r in records])
        ax.imshow(values,cmap=ListedColormap(["#EDF4F1","#DCAE7A","#C77C75"]),vmin=0,vmax=2,aspect="auto")
        ax.set_xticks(range(len(branches)),[NAMES[b] for b in branches],fontsize=8.5)
        ax.set_yticks(range(len(records)),[f"{r['category'].title()} {Path(r['path']).stem}" for r in records],fontsize=8.5)
        for i in range(len(records)):
            for j in range(len(branches)):
                ax.text(j,i,["Correct","FP","FN"][values[i,j]],ha="center",va="center",fontsize=8.7,color=INK)
        ax.tick_params(length=0,pad=5)
        ax.set_xticks(np.arange(-.5,len(branches),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(records),1),minor=True)
        ax.grid(which="minor",color="white",linewidth=2)
        ax.tick_params(which="minor",bottom=False,left=False)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.subplots_adjust(left=.31,right=.99,bottom=.17,top=.97)
    save(fig,"error-complementarity")

def score_distribution():
    cats=DATA["categories"]
    fig,axes=plt.subplots(1,len(cats),figsize=(7.1,2.65),squeeze=False)
    for ax,cat in zip(axes[0],cats):
        records=rows_for(cat)
        values=np.array([r["fusion_score"] for r in records]);labels=np.array([r["label"] for r in records])
        threshold=DATA["configs"][cat]["thresholds"]["fusion"]
        lo,hi=min(values.min(),threshold),max(values.max(),threshold)
        width=max(float(hi-lo),1e-6)
        bin_width=1.05*width/14
        # Equal-width bins have the frozen threshold as an exact edge, so a
        # histogram bar cannot visually straddle the decision boundary solely
        # because of an arbitrary bin origin.
        first=int(np.floor((lo-threshold)/bin_width))
        last=int(np.ceil((hi-threshold)/bin_width))
        bins=threshold+bin_width*np.arange(first,last+1)
        for label,name,color in ((0,"Normal",BLUE),(1,"Anomalous",ORANGE)):
            group=values[labels==label]
            ax.hist(group,bins=bins,weights=np.full(len(group),1/len(group)),color=color,alpha=.62,
                    edgecolor="white",linewidth=.4,label=f"{name} ($n$={len(group)})")
            # Rug marks preserve observed scores without adding jitter.
            y=-.035 if label==0 else -.065
            ax.plot(group,np.full(len(group),y),"|",color=color,markersize=4,
                    transform=ax.get_xaxis_transform(),clip_on=False,markeredgewidth=.7)
        ax.axvline(threshold,color=PURPLE,lw=1.4,ls=(0,(4,2)),label=rf"$\tau$ = {threshold:.4f}")
        ax.set_title(cat.title(),pad=6)
        ax.set_xlabel("Fused anomaly score",labelpad=11);ax.set_ylabel("Within-class proportion")
        ax.tick_params(axis="x",pad=12)
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.set_xlim(bins[0],bins[-1]);ax.set_ylim(bottom=0);clean(ax,"y")
        ax.legend(frameon=False,loc="best",fontsize=8,handlelength=1.8)
    fig.subplots_adjust(left=.085,right=.99,bottom=.23,top=.9,wspace=.30)
    save(fig,"score-distribution")

def heatmaps():
    boundary=DATA["boundary_samples"]
    if len(boundary)!=4:raise ValueError("The heatmap layout expects two classes for two product categories")
    fig=plt.figure(figsize=(7.1,4.65));row_y=[.57,.075]
    for n,d in enumerate(boundary):
        row=n//2;pair=n%2;x=.008+pair*.465
        spec=DATA["visuals"]["boundary"][d["path"]]
        if set(spec)!={"original","distance"}:raise ValueError(f"Missing saved original or distance array for {d['path']}")
        original=Image.open(BASE/spec["original"]).convert("RGB")
        distances=np.load(BASE/spec["distance"],allow_pickle=False)
        if distances.ndim!=2 or not np.isfinite(distances).all():raise ValueError(f"Invalid distance map: {d['path']}")
        normalized=np.clip(distances/max(float(np.percentile(distances,99)),1e-8),0,1)
        fig.text(x+.218,row_y[row]+.365,f"{d['category'].title()} {d['id']:03d} · {d['label']}",
                 ha="center",va="bottom",fontsize=9.2)
        for j,(title,pixels) in enumerate((("Original",original),("Local distance",normalized))):
            ax=fig.add_axes([x+j*.225,row_y[row],.213,.345])
            if j:ax.imshow(pixels,cmap="viridis",vmin=0,vmax=1,interpolation="nearest")
            else:ax.imshow(pixels)
            ax.axis("off")
            ax.text(.5,-.045,title,transform=ax.transAxes,ha="center",va="top",fontsize=8.4)
        fig.text(x+.218,row_y[row]-.057,rf"$S$ = {d['score']:.6f}   $\Delta$ = {d['margin']:+.6f}",
                 ha="center",va="top",fontsize=8.2)
    cax=fig.add_axes([.951,.20,.015,.57])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap="viridis"),cax=cax,ticks=[0,.25,.5,.75,1])
    cb.ax.tick_params(labelsize=7.5,length=2);cb.outline.set_linewidth(.5)
    cb.set_label("Normalized local distance",fontsize=8,labelpad=4)
    save(fig,"qualitative-heatmaps")

if __name__=="__main__":
    for function in (method,training_samples,feature_responses,roc_pr,confusion,errors,score_distribution,heatmaps):
        function()
    print("Rendered 8 figures from verified run data (PDF + PNG).")

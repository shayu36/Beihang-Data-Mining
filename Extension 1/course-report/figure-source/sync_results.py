"""Recompute report charts and the LaTeX table from a verified run snapshot."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (confusion_matrix,precision_score,recall_score,f1_score,
                             roc_auc_score,average_precision_score,roc_curve,precision_recall_curve)

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,default=ROOT/'results'/'run2')
args=parser.parse_args()
run=args.run.resolve()
with (run/'metrics.csv').open(encoding='utf-8-sig',newline='') as f: metrics=list(csv.DictReader(f))
with (run/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f: preds=list(csv.DictReader(f))
verification=json.loads((run/'verification.json').read_text(encoding='utf-8'))
assert verification['status']=='passed' and verification['raw_images']['performed']
assert verification['raw_images']['total_images']==602
lookup={(r['domain'],r['scope'],r['method']):r for r in metrics}
for row in metrics:
    sub=[r for r in preds if r['domain']==row['domain'] and r['method']==row['method'] and (row['scope']=='official' or r['overlaps_train']=='0')]
    y=np.array([int(r['label']) for r in sub]); p=np.array([int(r['prediction']) for r in sub]); score=np.array([float(r['score']) for r in sub])
    assert len(y)==int(row['n'])
    assert list(confusion_matrix(y,p,labels=[0,1]).ravel())==[int(row[k]) for k in ('tn','fp','fn','tp')]
    for k,v in [('precision',precision_score(y,p,zero_division=0)),('recall',recall_score(y,p,zero_division=0)),('f1',f1_score(y,p,zero_division=0)),('auroc',roc_auc_score(y,score)),('average_precision',average_precision_score(y,score))]:
        assert abs(float(row[k])-v)<1e-12,(row['domain'],row['method'],k)

groups=['hazelnut','zipper','thyroid']
methods=['baseline','normal_reference','improved']
names={'hazelnut':'榛子','zipper':'拉链','thyroid':'甲状腺'}
labels={'baseline':'基线','normal_reference':'正常参考','improved':'适配方法'}
lines=[r'\begin{tabular}{llrrrrrr}',r'\toprule',r'数据&方法&TN/FP/FN/TP&Precision&Recall&F1&AUROC&AP\\\midrule']
for domain in groups:
    for method in methods:
        r=lookup[(domain,'official',method)]
        counts='/'.join(r[k] for k in ('tn','fp','fn','tp'))
        fields=' & '.join(f'{float(r[k]):.4f}' for k in ('precision','recall','f1','auroc','average_precision'))
        lines.append(f'{names[domain]} & {labels[method]} & {counts} & {fields} '+r'\\')
lines.extend([r'\bottomrule',r'\end{tabular}'])
(ROOT/'sec').mkdir(exist_ok=True)
(ROOT/'sec'/'results-table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# The analysis paragraphs discuss this exact run. Refuse to overwrite plots if
# later results change until those paragraphs are reviewed and revised.
expected={'hazelnut':(40,0,0,15),'zipper':(32,0,0,15),'thyroid':(1833,6,1,93)}
for d,counts in expected.items():
    assert tuple(int(lookup[(d,'official','improved')][k]) for k in ('tn','fp','fn','tp'))==counts, 'Review prose for the new results.'
assert tuple(int(lookup[('thyroid','no_train_overlap','improved')][k]) for k in ('tn','fp','fn','tp'))==(1777,6,1,93)

plt.rcParams.update({'font.family':'serif','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
figdir=ROOT/'figures'; figdir.mkdir(exist_ok=True)
def save(fig,name):
    fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(figdir/(name+'.png'),dpi=250,bbox_inches='tight')
    plt.close(fig)

chinese=FontProperties(fname=str(ROOT/'fonts'/'SimHei.ttf'))
fig,ax=plt.subplots(figsize=(7.2,3.0));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,title,detail,colour='#f2f6fb'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',
                               facecolor=colour,edgecolor='#708ba6',linewidth=.8))
    ax.text(x+w/2,y+h*.67,title,fontproperties=chinese,fontsize=9,ha='center',va='center',color='#243444')
    ax.text(x+w/2,y+h*.30,detail,fontsize=7.5,ha='center',va='center',color='#476786')
def arrow(a,b,dashed=False):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#3b617c',lw=1.0,linestyle='--' if dashed else '-'))
ax.text(.03,.96,'正常训练与参数确定',fontproperties=chinese,fontsize=10,color='#263d53',va='top')
box(.03,.72,.25,.17,'正常样本与标注图像','normal / labelled images')
box(.38,.72,.25,.17,'拟合参考与校准统计','fit / calibration')
box(.73,.72,.24,.17,'冻结阈值与模型','frozen parameters')
arrow((.28,.805),(.38,.805));arrow((.63,.805),(.73,.805))
ax.text(.03,.60,'待测样本与统一评分',fontproperties=chinese,fontsize=10,color='#263d53',va='top')
box(.03,.35,.17,.17,'图像输入','RGB image')
box(.03,.08,.17,.17,'数值输入','6 features')
box(.26,.35,.20,.17,'卷积特征','ResNet18 + patches')
box(.26,.08,.20,.17,'数值适配','normal transform')
box(.53,.22,.18,.17,'正常参考','distance / pooling','#edf7f4')
box(.78,.22,.19,.17,'融合与判断','score / threshold','#f5f1fa')
arrow((.20,.435),(.26,.435));arrow((.20,.165),(.26,.165))
arrow((.46,.435),(.53,.32));arrow((.46,.165),(.53,.29));arrow((.71,.305),(.78,.305))
arrow((.51,.72),(.51,.54),True);arrow((.85,.72),(.85,.40),True)
fig.subplots_adjust(left=.015,right=.985,top=.98,bottom=.025);save(fig,'framework')

fig,axes=plt.subplots(1,3,figsize=(7.1,2.6))
for ax,domain,title in zip(axes,groups,['Hazelnut','Zipper','Thyroid']):
    r=lookup[(domain,'official','improved')]
    a=np.array([[int(r['tn']),int(r['fp'])],[int(r['fn']),int(r['tp'])]])
    b=a/a.sum(axis=1,keepdims=True)
    ax.imshow(b,vmin=0,vmax=1,cmap='Blues');ax.set_title(title)
    ax.set_xticks([0,1],['Normal','Anomaly'],fontsize=8);ax.set_yticks([0,1],['Normal','Anomaly'],fontsize=8)
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{a[i,j]}\n({b[i,j]:.1%})',ha='center',va='center',color='white' if b[i,j]>.55 else '#16324f',fontsize=9)
    ax.set_xlabel('Predicted');ax.set_ylabel('True')
fig.tight_layout();save(fig,'confusion')

fig,ax=plt.subplots(figsize=(6.8,2.8)); x=np.arange(3);width=.24
for j,(method,label,color) in enumerate([('baseline','Baseline','#9bb4cc'),('normal_reference','Normal reference','#4a8394'),('improved','Adapted method','#8366a2')]):
    vals=[float(lookup[(d,'official',method)]['f1']) for d in groups]
    ax.bar(x+(j-1)*width,vals,width,label=label,color=color)
    for xi,v in zip(x+(j-1)*width,vals):ax.text(xi,v+.015,f'{v:.3f}',ha='center',fontsize=8)
ax.set_xticks(x,['Hazelnut','Zipper','Thyroid']);ax.set_ylabel('F1');ax.set_ylim(0,1.18)
ax.legend(ncol=3,loc='upper center',frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();save(fig,'comparison')

fig,axes=plt.subplots(1,2,figsize=(7,3.0))
for method,label,color,style in [('baseline','Baseline','#3776ad','--'),('normal_reference','Normal reference','#d17e36',':'),('improved','Adapted method','#218976','-')]:
    subset=[r for r in preds if r['domain']=='thyroid' and r['method']==method]
    y=np.array([int(r['label']) for r in subset]);s=np.array([float(r['score']) for r in subset])
    fpr,tpr,_=roc_curve(y,s);precision,recall,_=precision_recall_curve(y,s)
    axes[0].plot(fpr,tpr,label=f'{label}  {roc_auc_score(y,s):.4f}',color=color,linestyle=style,lw=1.7)
    axes[1].step(recall,precision,where='post',label=f'{label}  {average_precision_score(y,s):.4f}',color=color,linestyle=style,lw=1.7)
axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Thyroid · ROC curves')
axes[1].set(xlabel='Recall',ylabel='Precision',title='Thyroid · PR curves')
for ax in axes:
    ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.02);ax.grid(alpha=.18);ax.legend(loc='lower right',fontsize=7,frameon=False)
fig.tight_layout();save(fig,'thyroid-roc-pr')

subset=[r for r in preds if r['domain']=='thyroid' and r['method']=='improved']
y=np.array([int(r['label']) for r in subset]);s=np.array([float(r['score']) for r in subset]);tau=float(subset[0]['threshold'])
fig,ax=plt.subplots(figsize=(6.4,2.8))
bins=np.unique(np.r_[np.linspace(0,max(s)*1.001,27),tau]);bins.sort()
for value,label,color in [(0,'Normal','#5d89b5'),(1,'Disease','#d77d4d')]:
    vals=s[y==value]
    ax.hist(vals,bins=bins,weights=np.ones(len(vals))/len(vals),alpha=.63,label=f'{label} (n={len(vals)})',color=color)
ax.axvline(tau,color='#6a4c93',linestyle='--',label=f'Threshold = {tau:.4f}')
ax.set(xlabel='Anomaly score',ylabel='Within-class proportion',title='Thyroid · score distribution')
ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.tight_layout();save(fig,'thyroid-score-distribution')
print(f'Checked {len(metrics)} metric rows and {len(preds)} predictions; rebuilt 5 figures and the results table.')

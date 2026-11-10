"""Chinese scientific method diagram, styled after the supplied course report.

The small matrices and classifier geometry illustrate operations, not measured
feature values. Output is a vector PDF and a high-resolution PNG.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
BLUE, TEAL, PURPLE = "#3B6B99", "#368578", "#77539D"
INK, LINE = "#263442", "#98A5B3"

def font_file(name, fallback):
    p=ROOT/"fonts"/name
    return p if p.exists() else Path(fallback)

SONG=FontProperties(fname=font_file("Simsun.ttc",r"C:\Windows\Fonts\simsun.ttc"))
HEI=FontProperties(fname=font_file("SimHei.ttf",r"C:\Windows\Fonts\simhei.ttf"))

def main():
    data=json.loads((ROOT/"figure-source"/"verified-results.json").read_text(encoding="utf-8"))
    configs=data["configs"]
    weights={float(configs[c]["local_weight"]) for c in data["categories"]}
    fractions={float(configs[c]["fraction"]) for c in data["categories"]}
    height,width=data["feature_maps"]["layer2"][:2]
    if len(weights)==1:
        local_weight=next(iter(weights));global_weight=1-local_weight
        global_label=f"{global_weight:g}";local_label=f"{local_weight:g}"
        fusion_formula=rf"$S={global_weight:g}\widetilde{{G}}+{local_weight:g}\widetilde{{P}}$"
    else:
        global_label="1−α";local_label="α"
        fusion_formula=r"$S=(1-\alpha)\widetilde{G}+\alpha\widetilde{P}$"
    k_values={max(1,int(np.ceil(height*width*q))) for q in fractions}
    top_label=f"最高{next(iter(k_values))}点均值" if len(k_values)==1 else "最高 k 点均值"
    bank_size=int(data["frozen"]["bank_size"])
    train_counts={sum(data["samples"]["train"][c].values()) for c in data["categories"]}
    train_label=f"{next(iter(train_counts))} 张 / 类" if len(train_counts)==1 else "按产品分别训练"
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf", "SimHei.ttf", "Simsun.ttc"):
        path=ROOT/"fonts"/name
        if path.exists():font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
                         "pdf.fonttype":42,"ps.fonttype":42,"mathtext.fontset":"stix"})
    fig,ax=plt.subplots(figsize=(7.1,3.1))
    ax.set(xlim=(0,14.2),ylim=(0,6.2));ax.axis("off")
    def text(x,y,s,size=8,color=INK,heading=False,ha="center",va="center",**kwargs):
        ax.text(x,y,s,fontproperties=HEI if heading else SONG,fontsize=size,color=color,
                ha=ha,va=va,linespacing=1.16,**kwargs)
    def math(x,y,s,size=10,color=INK):
        ax.text(x,y,s,fontsize=size,color=color,ha="center",va="center")
    def panel(x,y,w,h,edge=LINE,fill="white",dashed=False,r=.1):
        p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=.015,rounding_size={r}",
                        facecolor=fill,edgecolor=edge,linewidth=.65,
                        linestyle=(0,(3,2)) if dashed else "-")
        ax.add_patch(p)
    def box(x,y,w,h,s,size=8,edge=LINE,fill="white"):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,linewidth=.65))
        text(x+w/2,y+h/2,s,size)
    def arrow(points,color=INK,dashed=False):
        for i in range(len(points)-1):
            ax.add_patch(FancyArrowPatch(points[i],points[i+1],
                         arrowstyle="-|>" if i==len(points)-2 else "-",
                         mutation_scale=5.4,linewidth=.65,color=color,
                         shrinkA=0,shrinkB=0,linestyle=(0,(3,2)) if dashed else "-"))
    def grid(x,y,w,h,nx=4,ny=5,color=BLUE,fill="#DCEAF6"):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=color,linewidth=.55))
        for j in range(1,nx):ax.plot([x+j*w/nx]*2,[y,y+h],color=color,lw=.4)
        for j in range(1,ny):ax.plot([x,x+w],[y+j*h/ny]*2,color=color,lw=.4)
    # The top lane follows the reference: fitting on the left and transformation
    # definitions on the right, using the same light fills and restrained lines.
    panel(.1,4.84,6.7,1.24,edge="#93ADC8",fill="#F7FAFD")
    panel(7.0,4.84,7.1,1.24,edge="#CABBDD",fill="#FCFAFE")
    text(.25,5.91,"训练内拟合",10.7,heading=True,ha="left")
    grid(.32,5.03,.53,.5,color=TEAL,fill="#D8E9E2")
    math(.58,5.65,r"$\mathcal{D}_{\rm tr}$",9.4)
    text(1.8,4.92,train_label,6.4)
    box(1.17,5.04,1.25,.5,"分层五折",8.2)
    grid(2.84,5.03,.46,.5,nx=1,color=TEAL,fill="#D8E9E2")
    math(3.07,5.67,r"$s_{\rm OOF}$",9)
    box(3.75,5.00,2.68,.61,"选择参数与阈值\n全量重拟合并冻结",8)
    arrow([(.85,5.28),(1.17,5.28)])
    arrow([(2.42,5.28),(2.84,5.28)])
    arrow([(3.3,5.28),(3.75,5.28)])
    text(8.7,5.91,"折外正常分数校准",10.1,heading=True)
    math(8.72,5.40,r"$\widetilde{B}=\frac{B-m_B}{r_B}$",12)
    text(8.72,5.03,"中位数与分位差均由训练折外分数估计",6.4,color=TEAL)
    text(12.25,5.91,"加权融合",10.7,heading=True)
    math(12.2,5.36,fusion_formula,10.2)
    # Main stages: same hierarchy and quiet grey groupings as the reference.
    text(1.97,4.40,"共享特征",10.9,heading=True)
    text(5.88,4.40,"全局与局部评分",10.7,heading=True)
    text(9.12,4.40,"折外校准",10.7,heading=True)
    text(12.24,4.40,"融合与判断",10.7,heading=True)
    panel(4.38,.53,2.98,3.51,dashed=True)
    panel(8.28,.53,1.78,3.51,edge="#D6DAE0",fill="#F5F6F8")
    # Shared representation: input matrix, frozen network and feature tensors.
    grid(.26,2.13,.6,.74,nx=4,ny=5)
    math(.56,3.1,r"$x$",11)
    text(.56,1.92,"384 × 384",7)
    box(1.16,2.13,1.75,.74,"冻结 WRN50-2\nImageNet-1K V1",7.8)
    for dx,dy in [(.14,.14),(.07,.07),(0,0)]:
        grid(3.25+dx,2.13+dy,.48,.67,nx=3,ny=4)
    math(3.56,3.16,r"$F_2,F_3,F_4$",9.3)
    text(3.56,1.92,"多层特征",7.4)
    arrow([(.86,2.5),(1.16,2.5)])
    arrow([(2.91,2.5),(3.25,2.5)])
    arrow([(3.87,2.5),(4.10,2.5),(4.10,3.28),(4.38,3.28)])
    arrow([(4.10,2.5),(4.10,1.32),(4.38,1.32)])
    # A global pooled vector and a tiny SVM decision illustration.
    text(5.87,3.80,"全局 SVM",9.4,heading=True)
    text(5.87,3.48,"均值 / 最大池化 · 7 168 维",7.1)
    grid(4.65,2.91,.51,.36,nx=5,ny=1)
    arrow([(5.16,3.09),(5.46,3.09)])
    gx,gy=5.56,2.85
    points_a=[(.05,.07),(.19,.15),(.16,.36),(.36,.39)]
    points_b=[(.65,.1),(.75,.31),(.93,.29),(.96,.44)]
    for dx,dy in points_a:ax.plot(gx+dx,gy+dy,"o",ms=2.3,mec=BLUE,mfc="#B5D1E8",mew=.5)
    for dx,dy in points_b:ax.plot(gx+dx,gy+dy,"o",ms=2.3,mec=BLUE,mfc="white",mew=.5)
    tt=np.linspace(0,.54,40)
    ax.plot(gx+.5+.10*np.sin(tt*7),gy+tt,color=BLUE,lw=.7)
    text(5.87,2.60,"标准化 · RBF 核判别",7.5,color=BLUE)
    # Local feature grid, normal memory, nearest distance and top-k aggregation.
    text(5.87,2.12,"正常局部参照",9.4,heading=True)
    text(5.87,1.80,f"{height} × {width} 个 256 维描述子",7.2)
    grid(4.64,.96,.65,.56,nx=5,ny=5,color=TEAL,fill="#D8E9E2")
    math(5.00,.79,rf"$\mathcal{{M}}:{bank_size}\times256$",6.7,color=TEAL)
    ax.add_patch(Circle((5.66,1.25),.055,facecolor=TEAL,edgecolor=TEAL,lw=.5))
    arrow([(5.60,1.25),(5.29,1.25)],TEAL)
    math(5.65,1.52,r"$\mathbf{z}_p$",8.1,color=TEAL)
    arrow([(5.73,1.25),(6.01,1.25)],TEAL)
    heights=[.1,.16,.23,.33,.43,.5]
    for j,h in enumerate(heights):
        ax.add_patch(Rectangle((6.08+j*.13,.99),.11,h,facecolor="#BBDCCF",edgecolor=TEAL,lw=.45))
    text(6.43,.78,top_label,6.7,color=TEAL)
    # Raw scores and calibrated vectors.
    for yy,sym,col,fill,zsym in [(2.94,r"$G$",BLUE,"#DCEAF6",r"$\widetilde{G}$"),
                               (1.00,r"$P$",TEAL,"#D8E9E2",r"$\widetilde{P}$")]:
        grid(7.62,yy,.3,.69,nx=1,ny=4,color=col,fill=fill)
        math(7.77,yy+.89,sym,10.6)
        arrow([(7.36,yy+.345),(7.62,yy+.345)])
        box(8.45,yy,1.43,.69,"减中位数\n除分位差",8.1)
        arrow([(7.92,yy+.345),(8.45,yy+.345)])
        grid(10.36,yy,.29,.69,nx=1,ny=4,color=col,fill=fill)
        math(10.505,yy+.89,zsym,10.2)
        arrow([(9.88,yy+.345),(10.36,yy+.345)])
    # Dashed teal links carry training-fitted models and fixed calibration stats.
    arrow([(4.72,5.00),(4.72,4.71),(4.18,4.71),(4.18,3.92),(4.38,3.92)],TEAL,True)
    arrow([(6.02,5.00),(6.02,4.69),(8.14,4.69),(8.14,3.72),(8.44,3.72)],TEAL,True)
    text(9.17,.73,"复用训练统计量",6.6,color=TEAL)
    # Final weighted score and image-level judgment.
    arrow([(10.65,3.285),(11.14,3.285),(11.14,2.69)],BLUE)
    arrow([(10.65,1.345),(11.14,1.345),(11.14,1.99)],TEAL)
    text(11.11,3.53,global_label,7.5,color=BLUE)
    text(11.11,1.10,local_label,7.5,color=TEAL)
    ax.add_patch(Circle((11.22,2.34),.36,facecolor="#F0EAF6",edgecolor=PURPLE,lw=.8))
    text(11.22,2.34,"加权",8.5)
    grid(12.02,1.995,.34,.69,nx=1,ny=4,color=PURPLE,fill="#DCD0EA")
    math(12.19,2.9,r"$S$",11)
    arrow([(11.58,2.34),(12.02,2.34)])
    box(12.80,2.02,1.18,.64,"阈值 τ",8.8)
    arrow([(12.36,2.34),(12.80,2.34)])
    text(12.17,1.76,"异常分数",7.4,color=PURPLE)
    panel(12.73,.87,1.32,.53,edge="#B59CCF",fill="#F1ECF6",r=.03)
    text(13.39,1.135,"正常 / 异常",8.5,heading=True)
    arrow([(13.39,2.02),(13.39,1.4)])
    # Compact legend is positioned exactly like the reference's lower-left key.
    arrow([(.28,.88),(.76,.88)])
    text(.91,.88,"样本与评分",8.4,ha="left")
    arrow([(.28,.42),(.76,.42)],TEAL,True)
    text(.91,.42,"拟合与统计量",8.4,ha="left",color=TEAL)
    fig.subplots_adjust(0,0,1,1)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/"method-overview.pdf",bbox_inches="tight",pad_inches=.025)
    fig.savefig(OUT/"method-overview.png",dpi=400,bbox_inches="tight",pad_inches=.025)
    plt.close(fig)

if __name__=="__main__":main()

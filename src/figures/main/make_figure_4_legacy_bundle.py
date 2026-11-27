from __future__ import annotations

import csv
import json
import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

# Portable release overrides. The historical definitions above are retained as provenance.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("RAG_OSM_DATA_ROOT", REPOSITORY_ROOT.parent / "public_data_release_v1.0_candidate"))
OUT = Path(os.environ.get("RAG_OSM_FIGURE_OUTPUT", REPOSITORY_ROOT / "outputs" / "Figure_4"))
FIG_OUT = OUT / "figures"
OBJ_CSV = DATA_ROOT / "data" / "figure_source_data" / "main_figures" / "Figure_2" / "object_level_results.csv"
GRAPH_CSV = DATA_ROOT / "data" / "figure_source_data" / "main_figures" / "Figure_3" / "graph_assembly_results.csv"
MAP_GEOJSON = DATA_ROOT / "data" / "figure_source_data" / "main_figures" / "Figure_4" / "zhengzhou_rag_llm_topology_map.geojson"
BOUNDARY_GEOJSON = Path(os.environ.get("ZHENGZHOU_BOUNDARY_GEOJSON", DATA_ROOT / "third_party" / "zhengzhou_administrative_boundary.geojson"))
S20 = DATA_ROOT / "data" / "figure_source_data" / "main_figures" / "Figure_5"
S21 = S20

COL = {
    "h0": "#B8B8B8",
    "h1": "#4C78A8",
    "h2": "#E76F51",
    "ink": "#263238",
    "mid": "#66727A",
    "grid": "#E5E9EC",
    "pale": "#F5F7F8",
    "teal": "#2A9D8F",
    "green": "#59A14F",
    "gold": "#E9C46A",
    "purple": "#8E6CBA",
    "example": "#5B8FF9",
    "topology": "#61A5A0",
    "rule": "#B279A2",
}


TEXT = {
    "en": {
        "methods": ["H1  LLM", "H2  RAG-LLM"],
        "metrics": ["Accuracy", "Macro F1", "Prediction coverage"],
        "classes": ["Power lines", "Network connection points", "Power generation facilities"],
        "states": ["Prediction returned", "Insufficient evidence", "Invalid or missing"],
        "score": "Score",
        "f1": "F1 score",
        "share": "Share of prediction positions (%)",
        "higher": "Higher values are favourable",
        "lower": "Lower values are favourable",
        "rate": "Rate",
        "evidence": ["Retrieved labelled examples", "Shared topology context", "Retrieved power system rules"],
        "rule_state": ["Rule retrieved", "No rule retrieved"],
        "correct": "Correct predictions (%)",
        "paths": ["Retained correct", "Rescued to correct", "Degraded", "Unresolved / other"],
    },
    "zh": {
        "methods": ["H1  LLM", "H2  RAG-LLM"],
        "metrics": ["准确率", "宏平均 F1", "预测覆盖率"],
        "classes": ["电力线路", "网络连接点", "发电设施"],
        "states": ["返回预测", "证据不足", "无效或缺失"],
        "score": "指标值",
        "f1": "F1 值",
        "share": "占全部预测位置的比例（%）",
        "higher": "数值越高越好",
        "lower": "数值越低越好",
        "rate": "比率",
        "evidence": ["检索到的已标注样例", "共享拓扑上下文", "检索到的电力领域规则"],
        "rule_state": ["检索到规则", "未检索到规则"],
        "correct": "正确预测率（%）",
        "paths": ["保持正确", "由错误转为正确", "由正确转为错误", "未解决或其他"],
    },
}


def set_style(lang: str) -> None:
    mpl.rcParams.update({
        "font.family": "Arial" if lang == "en" else "Microsoft YaHei",
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def save_all(fig: plt.Figure, stem: str, lang: str) -> None:
    target = FIG_OUT / ("English" if lang == "en" else "Chinese")
    target.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("png", 450), ("tif", 600), ("pdf", 450), ("svg", 450)]:
        fig.savefig(target / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def panel(ax, label: str) -> None:
    ax.text(-0.055, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", color=COL["ink"])


def clean(ax, left=True, bottom=True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.grid(axis="x", color=COL["grid"], linewidth=.65, zorder=0)
    ax.set_axisbelow(True)


def rounded(ax, x, y, w, h, text, fc="white", ec=COL["mid"], lw=1.0, fs=7.2, bold=False):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.014,rounding_size=.018",
                           facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            color=COL["ink"], fontweight="bold" if bold else "normal", linespacing=1.16)
    return patch


def arrow(ax, a, b, color=COL["mid"], lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                connectionstyle=f"arc3,rad={rad}", color=color,
                                linewidth=lw, shrinkA=2, shrinkB=2))


def draw_osm(ax, x, y, scale=.1):
    pts = np.array([[-.55,-.08],[-.22,.38],[.22,.28],[.55,-.15],[.12,-.40],[-.28,-.28]]) * scale
    pts[:, 0] += x; pts[:, 1] += y
    ax.plot(pts[:,0], pts[:,1], color=COL["h1"], lw=1.4)
    for px, py in pts:
        ax.add_patch(Circle((px,py), .07*scale, facecolor="white", edgecolor=COL["h1"], lw=1.1))


def draw_graph(ax, x, y, scale=.1, color=COL["h2"]):
    nodes = np.array([[-.55,0],[0,.42],[.55,0],[0,-.42]])*scale + np.array([x,y])
    for i,j in [(0,1),(1,2),(2,3),(3,0),(0,2)]:
        ax.plot(nodes[[i,j],0], nodes[[i,j],1], color=color, lw=1.3)
    for px,py in nodes:
        ax.add_patch(Circle((px,py), .08*scale, facecolor="white", edgecolor=color, lw=1.1))


def figure1(lang: str) -> None:
    set_style(lang); zh = lang == "zh"
    fig, ax = plt.subplots(figsize=(7.20, 3.35))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Single continuous narrative lane
    draw_osm(ax, .075, .66, .12)
    ax.text(.075, .50, "类别标签缺失的\nOSM 电力对象" if zh else "OSM power objects\nwith missing categories",
            ha="center", va="top", fontsize=7.1, color=COL["ink"])
    rounded(ax, .145, .56, .145, .20,
            "遮蔽字段\n共享拓扑上下文" if zh else "Masked fields\nShared topology context",
            fc="#EDF6F4", ec=COL["topology"])

    ax.text(.405, .84, "综合电力知识库" if zh else "Integrated power knowledge base",
            ha="center", fontsize=7.6, fontweight="bold", color=COL["ink"])
    rounded(ax, .325, .66, .16, .15,
            "已标注样例\nBM25 检索" if zh else "Labelled examples\nBM25 retrieval",
            fc="#EEF3FA", ec=COL["example"])
    rounded(ax, .325, .46, .16, .15,
            "电力领域规则\n精确匹配" if zh else "Power system rules\nExact matching",
            fc="#F6EFF8", ec=COL["rule"])

    rounded(ax, .545, .68, .12, .14, "H1\nLLM", fc="white", ec=COL["h1"], lw=1.4, bold=True)
    rounded(ax, .545, .45, .12, .14, "H2\nRAG-LLM", fc="white", ec=COL["h2"], lw=1.4, bold=True)
    rounded(ax, .72, .56, .12, .20, "设施类别\n预测" if zh else "Facility\ncategory", fc="#F7F8F8", ec=COL["ink"], bold=True)
    draw_graph(ax, .92, .66, .13)
    ax.text(.92, .50, "结构角色映射\n与图装配" if zh else "Structural roles\nand graph assembly",
            ha="center", va="top", fontsize=7.1, fontweight="bold", color=COL["ink"])

    arrow(ax, (.115,.66), (.145,.66), COL["mid"])
    arrow(ax, (.29,.68), (.545,.75), COL["topology"], rad=-.08)
    arrow(ax, (.29,.62), (.545,.52), COL["topology"], rad=.06)
    arrow(ax, (.485,.735), (.545,.54), COL["example"], rad=.12)
    arrow(ax, (.485,.535), (.545,.52), COL["rule"])
    arrow(ax, (.665,.75), (.72,.68), COL["h1"])
    arrow(ax, (.665,.52), (.72,.63), COL["h2"])
    arrow(ax, (.84,.66), (.865,.66), COL["mid"])

    # Three compact validation cards keep the lower reading lane ordered and uncluttered.
    ax.plot([.04,.96],[.315,.315], color=COL["grid"], lw=.9)
    validation_cards = [
        (.05, .08, .26, .16,
         "空间隔离\n0.125° 空间块 · 五折" if zh else "Spatial isolation\n0.125° blocks · five folds"),
        (.37, .08, .26, .16,
         "配对评价\n同一对象 · 同一窗口 · 三次重复" if zh else "Paired evaluation\nsame objects · same window · three repeats"),
        (.69, .08, .26, .16,
         "冻结后评价\n类别恢复 → 图装配" if zh else "Frozen evaluation\ncategory recovery → graph assembly"),
    ]
    for x,y,w,h,label in validation_cards:
        rounded(ax,x,y,w,h,label,fc="#F7F8F8",ec="#D9DEE1",lw=.8,bold=False,fs=6.4)
    arrow(ax,(.31,.16),(.37,.16),COL["mid"],.8)
    arrow(ax,(.63,.16),(.69,.16),COL["mid"],.8)
    save_all(fig, "Figure1_Study_Workflow_Round4", lang)


def object_data():
    return pd.read_csv(OBJ_CSV)


def condition_key(x: str) -> str:
    return "h1" if str(x).startswith("H1") else "h2"


def figure2(lang: str) -> None:
    set_style(lang); t=TEXT[lang]; df=object_data()
    fig = plt.figure(figsize=(7.20, 4.10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0,.85], width_ratios=[1.0,1.02], hspace=.42, wspace=.42)
    ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,:])

    # a overall repeated results
    panel(ax1,"a")
    sub=df[df.panel=="a"].copy(); sub["cond"]=sub.condition.map(condition_key)
    metric_cols=["accuracy","macro_f1","prediction_coverage"]
    y=np.arange(3)[::-1]
    for yi,col in zip(y,metric_cols):
        a=sub[sub.cond=="h1"][col].astype(float).to_numpy(); b=sub[sub.cond=="h2"][col].astype(float).to_numpy()
        for va,vb in zip(a,b): ax1.plot([va,vb],[yi,yi],color="#DDE2E5",lw=1.0,zorder=1)
        ax1.scatter(a, np.full_like(a,yi)+.055, s=16, facecolor="white", edgecolor=COL["h1"], lw=.8,zorder=2)
        ax1.scatter(b, np.full_like(b,yi)-.055, s=16, facecolor="white", edgecolor=COL["h2"], lw=.8,zorder=2)
        ma,mb=a.mean(),b.mean()
        ax1.scatter(ma,yi+.055,s=42,color=COL["h1"],edgecolor="white",lw=.6,zorder=3)
        ax1.scatter(mb,yi-.055,s=42,color=COL["h2"],edgecolor="white",lw=.6,zorder=3)
        ax1.text(ma-.012,yi+.18,f"{ma:.3f}",ha="right",color=COL["h1"],fontsize=6.1)
        ax1.text(mb+.012,yi-.20,f"{mb:.3f}",ha="left",color=COL["h2"],fontsize=6.1)
    ax1.set_yticks(y,t["metrics"]); ax1.set_xlim(.15,1.025); ax1.set_xticks([.2,.4,.6,.8,1.0]); ax1.set_xlabel(t["score"])
    clean(ax1,left=False); ax1.tick_params(axis="y",length=0)
    ax1.legend(handles=[Line2D([0],[0],marker="o",color=COL["h1"],label=t["methods"][0],lw=1.2),
                        Line2D([0],[0],marker="o",color=COL["h2"],label=t["methods"][1],lw=1.2)],
               frameon=False,loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,columnspacing=.9,handletextpad=.4)

    # b class-specific F1
    panel(ax2,"b")
    sub=df[df.panel=="b"].copy(); sub["cond"]=sub.condition.map(condition_key)
    cls_codes=["ACLineSegment","ConnectivityNode","EnergySource"]
    y=np.arange(3)[::-1]
    for yi,c in zip(y,cls_codes):
        a=sub[(sub["class"]==c)&(sub.cond=="h1")].f1.astype(float).to_numpy()
        b=sub[(sub["class"]==c)&(sub.cond=="h2")].f1.astype(float).to_numpy()
        ma,mb=a.mean(),b.mean()
        ax2.plot([ma,mb],[yi,yi],color="#DDE2E5",lw=4,solid_capstyle="round",zorder=1)
        ax2.plot([a.min(),a.max()],[yi+.07,yi+.07],color=COL["h1"],lw=1)
        ax2.plot([b.min(),b.max()],[yi-.07,yi-.07],color=COL["h2"],lw=1)
        ax2.scatter(ma,yi+.07,s=42,color=COL["h1"],edgecolor="white",lw=.6,zorder=3)
        ax2.scatter(mb,yi-.07,s=42,color=COL["h2"],edgecolor="white",lw=.6,zorder=3)
        ax2.text(ma,yi+.23,f"{ma:.3f}",ha="center",color=COL["h1"],fontsize=6.1)
        ax2.text(mb,yi-.25,f"{mb:.3f}",ha="center",color=COL["h2"],fontsize=6.1)
    ax2.set_yticks(y,t["classes"]); ax2.set_xlim(.65,1.01); ax2.set_xticks([.7,.8,.9,1.0]); ax2.set_xlabel(t["f1"])
    clean(ax2,left=False); ax2.tick_params(axis="y",length=0)

    # c response composition, simplified but fully denominator-preserving
    panel(ax3,"c")
    sub=df[df.panel=="c"].copy(); sub["cond"]=sub.condition.map(condition_key)
    state_group={"valid_prediction":0,"valid_insufficient_evidence":1,"INVALID":2,"missing":2}
    colors=[COL["teal"],COL["gold"],"#9AA1A6"]
    for row_y,cond,label in [(1,"h1",t["methods"][0]),(0,"h2",t["methods"][1])]:
        vals=np.zeros(3)
        for _,r in sub[sub.cond==cond].iterrows(): vals[state_group[str(r.state)]] += float(r.proportion)*100
        left=0
        for j,v in enumerate(vals):
            ax3.barh(row_y,v,left=left,height=.38,color=colors[j],edgecolor="white",lw=.7)
            if v>=4: ax3.text(left+v/2,row_y,f"{v:.1f}%",ha="center",va="center",fontsize=6.5,
                              color="white" if j==0 else COL["ink"],fontweight="bold" if j==0 else "normal")
            # Very small segments remain visible but are not labelled in place; this avoids
            # overlapping annotations at the right edge. Exact values remain in the data table.
            left+=v
    ax3.set_yticks([1,0],t["methods"]); ax3.set_xlim(0,100); ax3.set_xlabel(t["share"])
    clean(ax3,left=False); ax3.tick_params(axis="y",length=0)
    ax3.legend(handles=[mpl.patches.Patch(color=c,label=l) for c,l in zip(colors,t["states"])],
               frameon=False,loc="lower center",bbox_to_anchor=(.5,1.02),ncol=3,columnspacing=1.2)
    save_all(fig,"Figure2_Object_Level_Recovery_Round4",lang)


def figure3(lang: str) -> None:
    set_style(lang); t=TEXT[lang]; df=pd.read_csv(GRAPH_CSV)
    df=df[df.condition.isin(["H1_no_retrieval_LLM","H2_KB_RAG_LLM"])].copy(); df["cond"]=df.condition.map(condition_key)
    high=[("effective_assembly_rate","Effective assembly","有效装配"),
          ("endpoint_valid_connection_rate","Valid endpoints","端点有效连接"),
          ("largest_connected_component_retention_rate","Largest component retained","最大连通分量保留")]
    low=[("missing_gold_line_rate","Missing lines","缺失线路"),
         ("isolated_node_rate","Isolated nodes","孤立节点"),
         ("dangling_line_rate","Dangling lines","悬空线路")]
    fig,axes=plt.subplots(1,2,figsize=(7.20,3.35),gridspec_kw={"wspace":.55})
    for ax,group,letter,title in zip(axes,[high,low],["a","b"],[t["higher"],t["lower"]]):
        panel(ax,letter); ax.set_title(title,loc="left",fontweight="bold",pad=12)
        y=np.arange(3)[::-1]
        labs=[]
        for yi,(metric,en,zh) in zip(y,group):
            labs.append(zh if lang=="zh" else en)
            a=df[(df.metric==metric)&(df.cond=="h1")].value.astype(float).to_numpy()
            b=df[(df.metric==metric)&(df.cond=="h2")].value.astype(float).to_numpy()
            ma,mb=a.mean(),b.mean()
            ax.plot([ma,mb],[yi,yi],color="#DDE2E5",lw=4,solid_capstyle="round",zorder=1)
            ax.scatter(a,np.full_like(a,yi+.07),s=9,color=COL["h1"],alpha=.22,edgecolor="none",zorder=2)
            ax.scatter(b,np.full_like(b,yi-.07),s=9,color=COL["h2"],alpha=.22,edgecolor="none",zorder=2)
            ax.scatter(ma,yi+.07,s=45,color=COL["h1"],edgecolor="white",lw=.6,zorder=3)
            ax.scatter(mb,yi-.07,s=45,color=COL["h2"],edgecolor="white",lw=.6,zorder=3)
            ax.text(ma,yi+.25,f"{ma:.3f}",ha="center",fontsize=6.2,color=COL["h1"])
            ax.text(mb,yi-.27,f"{mb:.3f}",ha="center",fontsize=6.2,color=COL["h2"])
        ax.set_yticks(y,labs); ax.set_xlim(-.02,1.035); ax.set_ylim(-.36,2.36); ax.set_xticks([0,.25,.5,.75,1]); ax.set_xlabel(t["rate"])
        clean(ax,left=False); ax.tick_params(axis="y",length=0)
    fig.legend(handles=[Line2D([0],[0],marker="o",color=COL["h1"],label=t["methods"][0],lw=1.2),
                        Line2D([0],[0],marker="o",color=COL["h2"],label=t["methods"][1],lw=1.2)],
               loc="upper center",bbox_to_anchor=(.5,1.03),frameon=False,ncol=2,columnspacing=1.4)
    save_all(fig,"Figure3_Graph_Assembly_Round4",lang)


def iter_rings(geom):
    typ=geom.get("type"); coords=geom.get("coordinates",[])
    if typ=="Polygon":
        for ring in coords: yield ring
    elif typ=="MultiPolygon":
        for poly in coords:
            for ring in poly: yield ring


def centroid_of_feature(feat):
    rings=list(iter_rings(feat["geometry"]))
    ring=max(rings,key=len)
    a=np.asarray(ring,float)
    return float(np.mean(a[:,0])),float(np.mean(a[:,1]))


def plot_boundary(ax, feat, fill=True):
    for ring in iter_rings(feat["geometry"]):
        a=np.asarray(ring,float)
        ax.fill(a[:,0],a[:,1],facecolor="#F5F4F0" if fill else "none",edgecolor="#BFC3C5",lw=.65,zorder=0)


def figure4(lang: str) -> None:
    set_style(lang)
    boundary=json.loads(BOUNDARY_GEOJSON.read_text(encoding="utf-8"))
    data=json.loads(MAP_GEOJSON.read_text(encoding="utf-8"))
    fig,ax=plt.subplots(figsize=(7.20,4.55))
    for feat in boundary["features"]: plot_boundary(ax,feat,fill=True)
    line_segments=[]
    node_xy=[]
    source_xy=[]
    for feat in data["features"]:
        g=feat["geometry"]; c=feat["properties"].get("consensus_class")
        if g["type"]=="LineString" and c=="ACLineSegment":
            line_segments.append(np.asarray(g["coordinates"],float))
        elif g["type"]=="Point" and c=="ConnectivityNode":
            node_xy.append(g["coordinates"][:2])
        elif c=="EnergySource":
            if g["type"]=="Point": x,y=g["coordinates"][:2]
            else:
                a=np.asarray(g["coordinates"],float); x,y=float(a[:,0].mean()),float(a[:,1].mean())
            source_xy.append((x,y))
    line_count=len(line_segments); node_count=len(node_xy); source_count=len(source_xy)
    if line_segments:
        ax.add_collection(LineCollection(line_segments,colors=COL["h1"],linewidths=.43,alpha=.75,zorder=2))
    if node_xy:
        node_arr=np.asarray(node_xy,float)
        ax.scatter(node_arr[:,0],node_arr[:,1],s=.42,color="#2F3B43",alpha=.22,linewidths=0,zorder=1.5)
    if source_xy:
        source_arr=np.asarray(source_xy,float)
        ax.scatter(source_arr[:,0],source_arr[:,1],s=18,marker="*",facecolor=COL["h2"],edgecolor="white",lw=.35,zorder=4)

    pinyin={"中原区":"Zhongyuan","二七区":"Erqi","登封市":"Dengfeng","新密市":"Xinmi","中牟县":"Zhongmu",
            "上街区":"Shangjie","巩义市":"Gongyi","荥阳市":"Xingyang","金水区":"Jinshui","管城回族区":"Guancheng Hui",
            "惠济区":"Huiji","新郑市":"Xinzheng"}
    manual={"Zhongyuan":(113.61,34.76),"Erqi":(113.62,34.66),"Jinshui":(113.72,34.80),
            "Guancheng Hui":(113.73,34.71),"Huiji":(113.59,34.89)}
    for feat in boundary["features"]:
        name=pinyin.get(feat["properties"].get("name"),feat["properties"].get("name","")); x,y=centroid_of_feature(feat)
        x,y=manual.get(name,(x,y))
        ax.text(x,y,name,ha="center",va="center",fontsize=5.4,color="#4E555A",zorder=5,
                path_effects=[pe.withStroke(linewidth=2.2,foreground="white",alpha=.92)])

    # external legend in reserved upper margin
    handles=[Line2D([0],[0],color=COL["h1"],lw=2,label=(f"Power lines  (n={line_count:,})" if lang=="en" else f"电力线路（n={line_count:,}）")),
             Line2D([0],[0],marker="o",color="none",markerfacecolor="#2F3B43",markersize=4,label=(f"Network connection points  (n={node_count:,})" if lang=="en" else f"网络连接点（n={node_count:,}）")),
             Line2D([0],[0],marker="*",color="none",markerfacecolor=COL["h2"],markersize=7,label=(f"Power generation facilities  (n={source_count:,})" if lang=="en" else f"发电设施（n={source_count:,}）"))]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.995),ncol=3,frameon=False,columnspacing=1.2,handlelength=2.4)

    # Scale bar and north arrow
    xmin,xmax=112.62,114.36; ymin,ymax=34.16,35.08
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    lat0=(ymin+ymax)/2; km_per_deg_lon=111.32*math.cos(math.radians(lat0)); dx=25/km_per_deg_lon
    sx,sy=xmin+.12,ymin+.08
    ax.plot([sx,sx+dx],[sy,sy],color=COL["ink"],lw=2.2,zorder=6); ax.plot([sx,sx],[sy-.012,sy+.012],color=COL["ink"],lw=1.2); ax.plot([sx+dx,sx+dx],[sy-.012,sy+.012],color=COL["ink"],lw=1.2)
    ax.text(sx+dx/2,sy+.025,"25 km",ha="center",fontsize=6.2)
    ax.annotate("",xy=(xmax-.07,ymax-.04),xytext=(xmax-.07,ymax-.13),arrowprops=dict(arrowstyle="-|>",lw=1.4,color=COL["ink"]))
    ax.text(xmax-.07,ymax-.15,"N",ha="center",va="top",fontweight="bold",fontsize=7)
    ax.set_xlabel("Longitude (°E)" if lang=="en" else "经度（°E）")
    ax.set_ylabel("Latitude (°N)" if lang=="en" else "纬度（°N）")
    ax.grid(color="#E7E9EA",lw=.45); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False); ax.spines[["left","bottom"]].set_color("#7A7F82")
    fig.subplots_adjust(top=.89)
    save_all(fig,"Figure4_Zhengzhou_Topology_Map_Round4",lang)


def figure5(lang: str) -> None:
    set_style(lang); t=TEXT[lang]
    ev=pd.read_csv(S20/"evidence_channel_coverage.csv")
    pred=pd.read_csv(S20/"prediction_outcomes_by_rule_applicability.csv")
    paths=pd.read_csv(S21/"paired_disposition_by_gold_class.csv")
    fig=plt.figure(figsize=(7.20,3.55)); gs=fig.add_gridspec(1,3,width_ratios=[.88,1.0,1.38],wspace=.55)
    ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[0,2])

    panel(ax1,"a")
    channel_order=["EX retrieved examples","Shared TOPO context","RULE evidence"]
    cmap={"EX retrieved examples":COL["example"],"Shared TOPO context":COL["topology"],"RULE evidence":COL["rule"]}
    y=np.arange(3)[::-1]
    vals=[]
    for yi,ch,lab in zip(y,channel_order,t["evidence"]):
        v=float(ev.loc[ev.evidence_channel==ch,"coverage_percent"].iloc[0]); vals.append(v)
        ax1.plot([0,v],[yi,yi],color=cmap[ch],lw=5,alpha=.24,solid_capstyle="round")
        ax1.scatter(v,yi,s=50,color=cmap[ch],edgecolor="white",lw=.7,zorder=3)
        ax1.text(min(v+3,104),yi,f"{v:.1f}%",va="center",fontsize=6.2,color=cmap[ch],fontweight="bold")
    ax1.set_yticks(y,t["evidence"]); ax1.set_xlim(0,112); ax1.set_xticks([0,50,100]); ax1.set_xlabel("Evidence availability (%)" if lang=="en" else "证据可用率（%）")
    clean(ax1,left=False); ax1.tick_params(axis="y",length=0)

    panel(ax2,"b")
    states=["MATCHED","RULE_NONE"]; xs=[0,1]
    for x,state,label in zip(xs,states,t["rule_state"]):
        v=[]
        for method in ["H1_no_retrieval_LLM","H2_KB_RAG_LLM"]:
            q=pred[(pred.rule_match_state==state)&(pred.method==method)&(pred.outcome=="Correct")]
            v.append(float(q.percent.iloc[0]))
        ax2.plot([x-.08,x+.08],v,color="#C9CED2",lw=2.5,zorder=1)
        ax2.scatter(x-.08,v[0],s=48,color=COL["h1"],edgecolor="white",lw=.7,zorder=2)
        ax2.scatter(x+.08,v[1],s=48,color=COL["h2"],edgecolor="white",lw=.7,zorder=2)
        ax2.text(x-.08,v[0]+1.3,f"{v[0]:.1f}",ha="center",fontsize=6,color=COL["h1"])
        ax2.text(x+.08,v[1]+1.3,f"{v[1]:.1f}",ha="center",fontsize=6,color=COL["h2"])
    ax2.set_xticks(xs,t["rule_state"]); ax2.set_ylim(55,102); ax2.set_ylabel(t["correct"]); ax2.grid(axis="y",color=COL["grid"],lw=.65)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.legend(handles=[Line2D([0],[0],marker="o",color=COL["h1"],label=t["methods"][0],lw=0),Line2D([0],[0],marker="o",color=COL["h2"],label=t["methods"][1],lw=0)],
               frameon=False,loc="lower center",bbox_to_anchor=(.5,.01),ncol=1,handletextpad=.4)

    panel(ax3,"c")
    class_codes=["ACLineSegment","ConnectivityNode","EnergySource"]
    disp=["Retained correct","Rescued to correct","Degraded from correct","Unresolved / other"]
    mat=np.zeros((3,4))
    for i,c in enumerate(class_codes):
        for j,d in enumerate(disp):
            mat[i,j]=float(paths[(paths.gold_class==c)&(paths.paired_disposition==d)].percent.iloc[0])
    # Draw the matrix as vector rectangles so the SVG remains fully editable.
    path_cmap=mpl.colors.LinearSegmentedColormap.from_list("paths",["#F4F6F7","#A8D5CF","#2A9D8F"])
    path_norm=mpl.colors.Normalize(vmin=0,vmax=100)
    for i in range(3):
        for j in range(4):
            ax3.add_patch(mpl.patches.Rectangle((j-.5,i-.5),1,1,
                          facecolor=path_cmap(path_norm(mat[i,j])),edgecolor="none"))
    for i in range(3):
        for j in range(4):
            v=mat[i,j]; ax3.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=6.1,color="white" if v>55 else COL["ink"],fontweight="bold" if v>20 else "normal")
    path_labels = (["保持\n正确","纠正为\n正确","由正确\n变差","未解决\n或其他"] if lang=="zh" else
                   ["Retained\ncorrect","Rescued\nto correct","Degraded","Unresolved\nor other"])
    ax3.set_xlim(-.5,3.5); ax3.set_ylim(2.5,-.5)
    ax3.set_yticks(range(3),t["classes"]); ax3.set_xticks(range(4),path_labels,rotation=0,ha="center")
    ax3.tick_params(length=0); ax3.spines[:].set_visible(False)
    ax3.set_xlabel("Share within each facility class (%)" if lang=="en" else "各设施类别内部占比（%）")
    save_all(fig,"Figure5_Retrieval_Mechanism_Round4",lang)


def main():
    parser = argparse.ArgumentParser(description="Export selected Round 4 main figures.")
    parser.add_argument("--language", choices=("en", "zh", "all"), default="all")
    parser.add_argument("--figure", choices=("1", "2", "3", "4", "5", "all"), default="all")
    args = parser.parse_args()
    languages = ("en", "zh") if args.language == "all" else (args.language,)
    figures = ("1", "2", "3", "4", "5") if args.figure == "all" else (args.figure,)
    makers = {"1": figure1, "2": figure2, "3": figure3, "4": figure4, "5": figure5}
    for lang in languages:
        for number in figures:
            makers[number](lang)


if __name__ == "__main__":
    main()

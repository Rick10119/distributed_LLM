#!/usr/bin/env python3
"""Build the progress briefing from the active manuscript figures only."""
from __future__ import annotations
import argparse
from html import escape
from pathlib import Path
def svg(path:Path)->str:
    s=path.read_text(encoding="utf-8");
    if "<svg" not in s:raise ValueError(f"Not an SVG: {path}")
    return s[s.index("<svg"):]
def main():
    p=argparse.ArgumentParser();p.add_argument("--model-version",required=True);p.add_argument("--method-svg",type=Path,required=True)
    for n in range(1,6):p.add_argument(f"--figure{n}-svg",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();figs={n:svg(getattr(a,f"figure{n}_svg")) for n in range(1,6)};method=svg(a.method_svg)
    titles={1:"需求、异构计算与三种企业内部架构",2:"31行业部署成本与原负荷匹配价值",3:"集团单节点与跨节点灵活性",4:"企业内部架构与独立大型云反事实",5:"取水空间分布与水稀缺压力"}
    captions={1:"核心架构为IF、IG-1host与IG-multisite；II-1host已退出。",2:"成本仅在相同有效服务下比较；IF为整数安装，两个IG为连续等效定容。",3:"跨节点价值由实际负荷下IG-multisite减IG-1host识别；零负荷配对只用于IG-1host。",4:"大型云读取独立全国云情景，不以II-1host替代；当前图不报告尚未重建的材料足迹。",5:"本地侧使用新IF行业等效电量，云端空间份额仍为容量代理。"}
    sections="".join(f'<section><h2>Figure {n}｜{titles[n]}</h2><div class="figure">{figs[n]}</div><p>{captions[n]}</p></section>' for n in range(1,6))
    html=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>制造业AI部署进度简报</title><style>body{{margin:0;background:#f5f1e8;color:#1d2924;font-family:-apple-system,"PingFang SC",sans-serif}}main{{max-width:1160px;margin:auto;padding:42px 20px}}header,section{{background:#fffdf8;border:1px solid #d8d1c2;padding:24px;margin-bottom:24px}}h1{{font-size:42px;margin:.2em 0}}h2{{font-size:24px}}.figure svg{{width:100%;height:auto}}p{{color:#5f6b65}}.boundary{{border-left:4px solid #d9793c;padding:12px;background:#eee7d8}}</style></head><body><main><header><small>CORE RESULTS · {escape(a.model_version)}</small><h1>制造业AI的集团部署架构</h1><p>活动核心比较逐厂独立、集团单节点与集团多节点协调，并将大型云作为独立反事实。</p></header><section><h2>方法框架</h2><div class="figure">{method}</div></section>{sections}<div class="boundary">当前图件代码和依赖已切换至新核心口径；全国优化尚未完成，因此本页不预先陈述成本排序或跨节点收益方向。</div></main></body></html>'''
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(html,encoding="utf-8")
if __name__=="__main__":main()

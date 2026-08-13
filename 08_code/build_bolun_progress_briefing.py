#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a concise, self-contained HTML page containing only core results."""

from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path


def read_svg(path: Path, number: int) -> str:
    content = path.read_text(encoding="utf-8")
    if "<svg" not in content or "</svg>" not in content:
        raise ValueError(f"Figure {number} input must be a complete SVG")
    return content[content.index("<svg") :]


def read_grid_comparison(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {row["scenario"] for row in rows} != {"IF", "IG", "II_1host"}:
        raise ValueError(f"Grid comparison must contain IF, IG, and II_1host: {path}")
    return {
        row["scenario"]: {
            "architecture_mw": float(row["architecture_national_incremental_grid_expansion_mw"]),
            "cloud_mw": float(row["all_industry_cloud_incremental_grid_expansion_mw"]),
            "avoided_fraction": float(row["avoided_grid_connection_capacity_vs_cloud_fraction"]),
        }
        for row in rows
    }


def read_factor_classes(path: Path) -> dict[str, list[str]]:
    classes: dict[str, list[str]] = {"high": [], "medium": [], "low": []}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            impact = row["impact_class"]
            if impact in classes:
                classes[impact].append(row["factor_label_cn"])
    if not all(classes.values()):
        raise ValueError(f"Factor results must contain high, medium, and low classes: {path}")
    return classes


def analysis_cards(result: str, implication: str, boundary: str) -> str:
    return (
        '<div class="analysis">'
        f'<article class="result"><b>{result.split("｜", 1)[0]}</b><p>{result.split("｜", 1)[1]}</p></article>'
        f'<article class="result"><b>{implication.split("｜", 1)[0]}</b><p>{implication.split("｜", 1)[1]}</p></article>'
        f'<article class="caution"><b>{boundary.split("｜", 1)[0]}</b><p>{boundary.split("｜", 1)[1]}</p></article>'
        '</div>'
    )


def enhance_html(
    html: str,
    factor_classes: dict[str, list[str]],
    core_grid: dict[str, dict[str, float]],
    no_shift_grid: dict[str, dict[str, float]],
) -> str:
    css = r'''
main{display:flex;flex-direction:column}header{order:0}section{order:20}.introduction{order:5;margin-top:8px}.sensitivity{order:80}.method-section{order:90}.boundary{order:100}footer{order:110}
.prose{background:var(--card);border:1px solid var(--line);padding:clamp(20px,4vw,34px)}.prose p{margin:0 0 14px;color:#34413c}.prose p:last-child{margin-bottom:0}.research-question{margin-top:18px;padding:15px 18px;background:#e7efe9;border-left:4px solid var(--green);font-weight:650}
.analysis{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}.analysis article{background:#f2eee4;border:1px solid var(--line);padding:14px}.analysis b{display:block;margin-bottom:5px;font-size:13px}.analysis p{margin:0;color:var(--muted);font-size:12px}.analysis .caution{border-top:3px solid var(--orange)}.analysis .result{border-top:3px solid var(--green)}
.method-grid,.sensitivity-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.method-card,.sensitivity-card{background:var(--card);border:1px solid var(--line);padding:18px}.method-card h3,.sensitivity-card h3{font-size:16px;margin:5px 0 8px}.method-card p,.sensitivity-card p{font-size:13px;color:var(--muted);margin:0}.status{font-size:11px;font-weight:800;color:var(--green)}.future{margin-top:14px}.future li{margin:7px 0;color:var(--muted);font-size:13px}
@media(max-width:760px){.analysis,.method-grid,.sensitivity-grid{grid-template-columns:1fr}}
'''
    html = html.replace("</style>", css + "</style>")

    introduction = r'''
<section class="introduction"><div class="section-head"><div><span class="kicker">INTRODUCTION</span><h2>为什么要重新讨论制造业AI的部署位置</h2></div><p>根据投稿信的核心问题意识，用中文说明研究动机、争议和贡献。</p></div><div class="prose"><p>制造业AI的增长正在同时触及企业成本和基础设施约束。大型数据中心需要集中接入大量电力，并伴随冷却用水、土地和设施建设；制造企业长期采购云端算力或AI服务，则需要持续支付容量和订阅费用。与此同时，生产记录、技术文件、现场图像、设备状态和工艺数据中的部分任务受到隐私、时延、可靠性和生产安全约束，并不天然适合全部发送到远程公有云。</p><p>因此，制造业AI不一定只有“集中到大型数据中心”这一条路径。计算可以部署在工厂，也可以由集团在多个工厂之间协调，或集中到更大的共享节点。传统判断认为云平台可以池化设备、提高利用率并运营高效机房，企业自建必然更贵；但这种判断往往没有同时核算任务期限、生产负荷、服务器装机、企业电费和新增电网接入容量。</p><p>本研究不预设本地或云端必然占优，而是在相同质量、时延、可靠性和数据边界下，要求各架构完成完全相同的有效AI服务。服务器由企业拥有时，任务期限、设备投资和用电成本由同一主体决策，可能形成不同于公有云的调度激励；但分散部署也可能付出重复装机、较低利用率和运维成本。</p><p>由此，“AI在哪里运行”不仅是IT采购问题，也是企业投资、负荷灵活性和能源基础设施问题。本文希望识别不同架构的成本与物理边界，并判断哪些方向在参数变化后仍然成立，而不是把任一架构描述为普遍最优。</p><div class="research-question">核心问题：在完成相同制造业AI服务的条件下，工厂侧分布式、集团集中和大型集中式部署如何改变企业成本、逐时负荷、新增接入容量以及水、土地和建设材料足迹？</div></div></section>
'''
    html = html.replace("</header>", "</header>" + introduction, 1)

    method_open = '<section><div class="section-head"><div><span class="kicker">METHOD</span><h2>从制造业活动到逐时AI设施负荷</h2></div><p>将需求核算方法从结果图中独立出来，并显式加入CPU/GPU任务路由。</p></div><div class="figure">'
    method_replacement = r'''<section class="method-section"><div class="section-head"><div><span class="kicker">METHOD</span><h2>方法概述：从制造活动到企业和电网结果</h2></div><p>简要说明研究如何连接需求、计算、企业决策和基础设施；不展开公式。</p></div><div class="method-grid"><article class="method-card"><span class="status">01｜需求</span><h3>自下而上估计有效AI服务</h3><p>将31个制造业大类按生产机制和企业规模区分，分别估计办公知识、业务Agent、异常复核、预测维护、生产排程和研发仿真。官方“使用任何AI”约束采用外延，任务覆盖、事件频率和调用深度另行估计。</p></article><article class="method-card"><span class="status">02｜计算</span><h3>保持服务等价并路由CPU/GPU</h3><p>不同架构在共同质量、时延、可靠性和数据边界下完成相同服务。实时任务按时完成，可延迟任务在截止期内移动；不同任务依据适用性进入CPU或GPU容量池。</p></article><article class="method-card"><span class="status">03｜企业与电网</span><h3>分别核算企业成本和容量影响</h3><p>逐时模型计算服务器状态、设施功率、生产负荷、企业购电和新增接入容量。AI情景与匹配的无AI反事实比较；企业目标不含电网升级成本，电网侧只报告新增容量、相对参考架构避免的容量及避免比例。</p></article></div><div class="figure" style="margin-top:14px">'''
    html = html.replace(method_open, method_replacement, 1)

    analyses = {
        "Figure 1｜": analysis_cards(
            "读图：需求并不由单一任务主导｜六类任务共同构成全国服务需求，且行业任务组合差异明显。这意味着统一的“每家企业一台GPU”或统一token强度不能代表制造业。",
            "含义：硬件路由影响装机和电力｜可路由至CPU的份额在行业间变化，说明服务器数量、功率和成本需要按任务结构计算，不能把全部AI服务等同为GPU推理。",
            "边界：路由比例仍是结构情景｜图中CPU/GPU份额用于检验异构计算机制，尚不是全国实测比例；正式结论需要相同质量和SLA下的任务吞吐基准。",
        ),
        "Figure 2｜": analysis_cards(
            "读图：成本差主要来自容量与付款边界｜本地成本由GPU/CPU服务器、设施、电费和运维共同构成；云端柱表示企业采购付款。因此图比较的是企业决策边界，不是本地与云商的社会资源成本。",
            "含义：架构成本接近时需要看非成本约束｜中国三种自建架构的总成本量级接近，部署选择可能进一步受数据驻留、时延、单点规模和组织运维能力影响。",
            "边界：成本排序尚待v0.8.0全国重估｜中国联合CPU/GPU逐时模型目前只完成单例验证；美国仍是下游估计。图中成本适合展示比较结构，不能作为当前无条件的本地—云排序结论。",
        ),
        "Figure 3｜": analysis_cards(
            "读图：工厂级增量峰值明显较小｜在当前行业样本中，IF标注的新增容量均低于IG。集中任务池形成更大的单点增量，即使总服务量没有变化。",
            "含义：生产负荷提供局部非同时性｜工厂侧AI负荷叠加到各自生产曲线，能够利用工厂之间和时段之间的差异；集团集中则把分散需求汇聚到同一接入点。",
            "边界：这不是全年配网工程结果｜图使用三个行业的连续代表周，且无光伏和储能。它支持负荷匹配机制，不足以识别季节峰值、合同容量或具体配变升级。",
        ),
        "Figure 4｜": analysis_cards(
            "读图：空间复用改变新增建设边界｜IF把服务器放入既有工厂，因此专用绿地壳体和土地在当前定义下为零；大型集中式云需要单列园区建筑、土地和结构材料。",
            "含义：电力、水和材料不能只看一个指标｜部署位置同时改变接入规模、现场用水和新增建设。某一架构在企业成本上较低，并不自动意味着所有物理足迹都更低。",
            "边界：零新增壳体不等于零改造｜工厂机房占用、供配电改造、冷却和改造材料尚未完整计量；大型云面积与结构原型也不是国家平均，图示为筛查而非完整生命周期清单。",
        ),
        "Figure 5｜": analysis_cards(
            "读图：总量与稀缺压力并不等价｜省级取水量较大不一定具有最高水稀缺权重；贵州等地的绝对取水量并非最大，但较高AWARE系数会放大稀缺加权影响。",
            "含义：集中部署形成空间暴露｜本地部署随制造活动分布，云端情景则随当前智算容量代理集中到少数省份。部署位置因此会改变地方水压力，而不仅是全国总取水量。",
            "边界：云端空间分配仍是代理｜当前份额不是制造业AI任务的真实云厂商路由，也未细分水源和月份。地图来源与审图号、设施分配和季节稀缺仍需核验。",
        ),
    }
    for marker, cards in analyses.items():
        caption_start = html.index(f'<p class="caption"><strong>{marker}')
        section_end = html.index("</div></section>", caption_start)
        html = html[:section_end] + cards + html[section_end:]

    labels = {key: "、".join(escape(value) for value in values) for key, values in factor_classes.items()}
    core_cloud = core_grid["IF"]["cloud_mw"]
    no_shift_cloud = no_shift_grid["IF"]["cloud_mw"]
    sensitivity = f'''
<section class="sensitivity"><div class="section-head"><div><span class="kicker">SENSITIVITY</span><h2>敏感性分析：目前发现与下一步</h2></div><p>优先判断架构差值和结论是否翻转，而不是只看总量变化百分比。</p></div><div class="sensitivity-grid"><article class="sensitivity-card"><span class="status">已完成｜C38单因素筛查</span><h3>需求规模与单位服务算力影响最大</h3><p>高影响：{labels['high']}；中影响：{labels['medium']}；低影响：{labels['low']}。该分级用于筛选全国扩展因素，不替代全国结论检验。</p></article><article class="sensitivity-card"><span class="status">已完成｜全国核心比较</span><h3>企业侧架构降低单一云中心接入容量</h3><p>单一绿地云中心新增接入容量为{core_cloud:,.2f} MW；IF、IG和II_1host分别为{core_grid['IF']['architecture_mw']:,.2f}、{core_grid['IG']['architecture_mw']:,.2f}和{core_grid['II_1host']['architecture_mw']:,.2f} MW，相对减少{core_grid['IF']['avoided_fraction']:.2%}、{core_grid['IG']['avoided_fraction']:.2%}和{core_grid['II_1host']['avoided_fraction']:.2%}。</p></article><article class="sensitivity-card"><span class="status">已完成｜不可调度压力测试</span><h3>方向未翻转，但行业单节点接近临界</h3><p>全部任务按到达时刻执行时，云中心升至{no_shift_cloud:,.2f} MW；IF、IG和II_1host仍分别减少{no_shift_grid['IF']['avoided_fraction']:.2%}、{no_shift_grid['IG']['avoided_fraction']:.2%}和{no_shift_grid['II_1host']['avoided_fraction']:.2%}。II_1host的最小余量明显较小。</p></article></div><div class="prose future"><strong>后续敏感性工作</strong><ul><li>完成31行业×三架构的v0.8.0联合CPU/GPU逐时重跑，再估计完整云成本、打平云价、服务器寿命、运维FTE和吞吐边界；当前Figure 2成本排序仅作历史边界展示。</li><li>完成低/高需求与高效率情景的全国输出，并运行关键参数角点，检验需求、算效、柔性、设施倍率和电价共同变化时的架构切换。</li><li>分别重估成本、水和土地材料结论；物理容量敏感性不能替代这些下游账户的稳健性分析。</li><li>加入配变与馈线档位、项目实现系数、地区单位造价和建设时序，判断连续容量差是否真正跨越投资阈值。</li></ul></div></section>
'''
    html = html.replace('<div class="boundary">', sensitivity + '<div class="boundary">', 1)
    old_boundary = "核心IF解释为工厂侧分布式、集团专网协同，但不单列专网、数据治理和协同平台成本。Figure 3展示无光伏、无储能的全GPU基准；光储采用率与协同价值将在敏感性分析中单独报告。异构CPU/GPU路由对接入容量的影响仍需下一步逐时重算。"
    new_boundary = "核心IF解释为工厂侧分布式、集团专网协同，但不单列专网、数据治理和协同平台成本。Figure 3展示无光伏、无储能的代表周；光储协同价值需用匹配的无AI反事实单独报告。v0.8.0已完成全国接入容量敏感性，但全国联合CPU/GPU成本、水和土地材料仍需重新估计。"
    html = html.replace(old_boundary, new_boundary)
    return html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--method-svg", type=Path, required=True)
    parser.add_argument("--sensitivity-factor-results", type=Path, required=True)
    parser.add_argument("--core-grid-comparison", type=Path, required=True)
    parser.add_argument("--no-shift-grid-comparison", type=Path, required=True)
    for number in range(1, 6):
        parser.add_argument(f"--figure{number}-svg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figures = {n: read_svg(getattr(args, f"figure{n}_svg"), n) for n in range(1, 6)}
    method = read_svg(args.method_svg, 0)
    factor_classes = read_factor_classes(args.sensitivity_factor_results)
    core_grid = read_grid_comparison(args.core_grid_comparison)
    no_shift_grid = read_grid_comparison(args.no_shift_grid_comparison)

    html = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>制造业AI部署：核心结果</title>
<style>
:root{--ink:#17241f;--muted:#617069;--paper:#f6f2e8;--card:#fffdf7;--line:#d8d2c3;--green:#176b52;--orange:#d9793c}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.65}
main{width:min(1160px,calc(100% - 28px));margin:auto;padding:45px 0 70px}header{max-width:900px;margin-bottom:48px}h1{font-family:"Songti SC",serif;font-size:clamp(34px,5vw,58px);line-height:1.12;margin:8px 0 18px}h1 em{color:var(--orange);font-style:normal}.eyebrow{color:var(--green);font-size:13px;font-weight:750;letter-spacing:.12em}.lead{font-size:18px;color:var(--muted)}
.findings{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:28px 0 10px}.finding{background:var(--card);border-top:4px solid var(--green);padding:18px;border-left:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.finding:nth-child(2){border-top-color:var(--orange)}.finding b{display:block;margin-bottom:5px}.finding span{font-size:13px;color:var(--muted)}
section{margin:48px 0}.section-head{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:14px}.section-head h2{font-size:25px;margin:0}.section-head p{max-width:550px;margin:0;color:var(--muted);font-size:13px}.kicker{display:block;color:var(--green);font-size:12px;font-weight:800;letter-spacing:.1em;margin-bottom:5px}
.figure{background:var(--card);border:1px solid var(--line);padding:18px}.figure svg{display:block;width:100%;height:auto;background:#fff}.caption{font-size:12px;color:var(--muted);margin:12px 5px 0}.caption strong{color:var(--ink)}
.boundary{margin-top:30px;padding:16px 18px;border-left:4px solid var(--orange);background:#eee7d8;color:var(--muted);font-size:13px}footer{border-top:1px solid var(--line);padding-top:22px;color:var(--muted);font-size:12px}
@media(max-width:760px){.findings{grid-template-columns:1fr}.section-head{display:block}.section-head p{margin-top:8px}.figure{padding:8px}}
</style></head><body><main>
<header><div class="eyebrow">CORE RESULTS · MANUFACTURING AI DEPLOYMENT · @@MODEL_VERSION@@</div><h1>制造业AI部署的<em>成本、电网与物理足迹</em></h1><p class="lead">以工厂侧分布式、集团专网协同为核心，比较集团集中算力池和大型云数据中心。中国与美国分别使用本国制造业AI需求；不同部署方式在服务边界内保持可比。</p>
<div class="findings"><div class="finding"><b>企业选择</b><span>比较本地自建与正式云端付款，识别企业采用本地部署的经济边界。</span></div><div class="finding"><b>电网接入</b><span>分布式部署降低最大单点规模；全国容量差异取决于调度、储能与大型云节点拆分。</span></div><div class="finding"><b>物理足迹</b><span>工厂空间复用减少新增壳体和土地，但现场水、改造材料和噪声仍需明确边界。</span></div></div></header>

<section><div class="section-head"><div><span class="kicker">METHOD</span><h2>从制造业活动到逐时AI设施负荷</h2></div><p>将需求核算方法从结果图中独立出来，并显式加入CPU/GPU任务路由。</p></div><div class="figure">@@METHOD@@<p class="caption"><strong>方法框架｜需求—计算—电力转换。</strong>由制造业活动和采用情景形成六类任务的有效服务需求，再按任务路由至CPU或GPU，配置含工程裕量的服务器容量，最终生成168小时设施负荷。</p></div></section>

<section><div class="section-head"><div><span class="kicker">FIGURE 1</span><h2>制造业AI需求、异构计算与部署架构</h2></div><p>展示六类任务的CPU/GPU路由、三种空间部署反事实、设施电量构成和31行业需求差异。</p></div><div class="figure">@@FIGURE1@@<p class="caption"><strong>Figure 1｜需求、异构硬件与架构。</strong>默认采用实践路由情景：预测维护50%、生产排程80%、数字孪生仿真50%分配至CPU，其余任务服务由GPU承担；该路由是结构性情景，并非观测硬件份额。电量面板将GPU与CPU设施电量分别堆叠，外部低、中、高情景按异构模型的电量比例缩放。核心部署为工厂侧分布式、集团专网协同，当前不新增专网、治理或协同平台成本。</p></div></section>

<section><div class="section-head"><div><span class="kicker">FIGURE 2</span><h2>满足制造业AI需求的企业成本</h2></div><p>中国方案使用中国需求，美国方案使用美国本土需求；本地与云端在各自国家内比较。</p></div><div class="figure">@@FIGURE2@@<p class="caption"><strong>Figure 2｜企业成本与部署选择。</strong>c和d分别按工厂侧分布式、集团集中算力池和大型集中节点分解中国与美国自建成本，均换算为十亿美元/年并使用相同纵轴。美国采用本土基准需求、CPU/GPU异构路由和基准服务器价格；当前以中国模型中的架构能耗差异作为代理，美国本土逐时三架构重算仍待完成。云端付款不是云商底层社会资源成本。</p></div></section>

<section><div class="section-head"><div><span class="kicker">FIGURE 3</span><h2>AI服务器负荷与制造业原始负荷的周内匹配</h2></div><p>选取三个代表行业，比较工厂侧分布式与集团集中算力池在连续168小时内如何叠加于企业原始负荷。</p></div><div class="figure">@@FIGURE3@@<p class="caption"><strong>Figure 3｜实测连续周的原始负荷与AI增量负荷。</strong>蓝灰色为企业原始负荷，橙色为新增AI服务器负荷，外轮廓为两者之和。C30、C38和C39分别从EWELD华南同行业企业数据中筛选完整连续周，并保持当前行业平均负荷规模；三行业代表周不要求来自同一日历周。核心基准不配置企业侧光伏和储能。</p></div></section>

<section><div class="section-head"><div><span class="kicker">FIGURE 4</span><h2>以IF为基准的运行资源与新增建设足迹</h2></div><p>以IF为统一基准，比较中国和美国低、中、高制造业AI需求下大型集中式云带来的资源和建设变化。</p></div><div class="figure">@@FIGURE4@@<p class="caption"><strong>Figure 4｜以IF为基准的资源与建设足迹。</strong>a给出IF和大型集中式云的现场用水及电网接入容量绝对量；中国两种架构的容量均来自活动物理优化，美国按本国需求容量缩放。b为绿地大型云相对IF增加的专用建筑壳体、土地转换和建设材料；当前IF假设复用既有工厂，因此专用新建壳体和土地为零。材料阴影表示钢框架至RC框架；材料结果尚未计入IF工厂改造材料，因为该项仍为NR。项目面积和结构原型不是国家平均，水泥尚未估算。</p></div></section>

<section><div class="section-head"><div><span class="kicker">FIGURE 5</span><h2>取水量的空间分布与水稀缺压力</h2></div><p>比较本地部署和云服务情景的省级取水分布；地图气泡与右侧柱图均使用统一的绝对取水量尺度。</p></div><div class="figure">@@FIGURE5@@<p class="caption"><strong>Figure 5｜空间用水与稀缺性。</strong>专题数据覆盖大陆31省；台湾、港澳和南海界线作为完整中国底图展示，但不补造研究数据。气泡面积和柱长表示估算取水量，颜色越红表示AWARE2.0年度非农业水稀缺系数越高。底图采用项目内 <code>province_shapes/CHN_full_adm</code>；正式公开发表前需进一步核验其来源和审图号。云端省级份额采用中国信通院《综合算力指数蓝皮书（2025年）》图7的在用智算规模（FP16）树图数字化结果，并以报告明确的河北14.8%校准；这是当前智算容量空间代理。</p></div></section>

<div class="boundary"><strong>当前最重要的结果边界：</strong>核心IF解释为工厂侧分布式、集团专网协同，但不单列专网、数据治理和协同平台成本。Figure 3展示无光伏、无储能的全GPU基准；光储采用率与协同价值将在敏感性分析中单独报告。异构CPU/GPU路由对接入容量的影响仍需下一步逐时重算。</div>
<footer>工作版本 @@MODEL_VERSION@@ · 核心结果页。发电侧水耗、实际配网工程、云厂商真实设施分配和本地改造材料不在当前量化边界内。</footer>
</main></body></html>'''
    for number, svg in figures.items():
        html = html.replace(f"@@FIGURE{number}@@", svg)
    html = html.replace("@@METHOD@@", method)
    html = html.replace("@@MODEL_VERSION@@", args.model_version)
    html = enhance_html(html, factor_classes, core_grid, no_shift_grid)
    if "@@FIGURE" in html:
        raise ValueError("Unresolved figure token")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()

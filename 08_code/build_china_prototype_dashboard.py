"""Build a standalone HTML summary of the China prototype results."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "05_results" / "figures"
OUTPUT = ROOT / "05_results" / "china_prototype_dashboard.html"


def svg(name: str) -> str:
    content = (FIGURES / name).read_text(encoding="utf-8")
    return content.replace('<rect width="100%" height="100%" fill="#ffffff"/>', "")


def main() -> None:
    direct = svg("enterprise-direct-cost.svg")
    social = svg("social-cost.svg")
    grid = svg("grid-expansion.svg")
    switch = svg("cloud-price-switch.svg")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>企业本地与云端 AI 推理：最小原型结果</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f4f7fa;
      --surface: #ffffff;
      --surface-soft: #edf3f7;
      --text: #17212b;
      --muted: #5d6b78;
      --line: #d6e0e7;
      --blue: #2f6f9f;
      --orange: #d9782d;
      --green: #5b9a68;
      --shadow: 0 14px 40px rgba(28, 50, 68, .08);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #10161c;
        --surface: #182129;
        --surface-soft: #202c35;
        --text: #edf3f7;
        --muted: #aab8c3;
        --line: #34434f;
        --shadow: none;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      line-height: 1.65;
    }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 52px 0 72px; }}
    header {{ padding: 8px 0 30px; border-bottom: 1px solid var(--line); }}
    .eyebrow {{ color: var(--blue); letter-spacing: .08em; font-size: 13px; font-weight: 600; }}
    h1 {{ max-width: 900px; margin: 10px 0 12px; font-size: clamp(30px, 5vw, 54px); line-height: 1.14; letter-spacing: -.025em; }}
    .lead {{ max-width: 850px; margin: 0; color: var(--muted); font-size: 18px; }}
    .thesis {{ margin: 28px 0 0; padding: 20px 24px; border-left: 5px solid var(--orange); background: var(--surface); box-shadow: var(--shadow); border-radius: 0 14px 14px 0; font-size: 18px; }}
    section {{ margin-top: 42px; }}
    h2 {{ margin: 0 0 16px; font-size: 25px; letter-spacing: -.01em; }}
    h3 {{ margin: 0 0 10px; font-size: 18px; }}
    p {{ margin: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 22px; box-shadow: var(--shadow); }}
    .case-title {{ display: flex; justify-content: space-between; gap: 16px; align-items: baseline; margin-bottom: 18px; }}
    .case-title span {{ color: var(--muted); font-size: 13px; }}
    .comparison {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .metric {{ padding: 16px; border-radius: 12px; background: var(--surface-soft); }}
    .metric .label {{ color: var(--muted); font-size: 13px; }}
    .metric .value {{ margin-top: 5px; font-size: 25px; line-height: 1.2; font-weight: 650; }}
    .metric .mode {{ margin-top: 5px; font-size: 14px; }}
    .private .value, .private .mode {{ color: var(--blue); }}
    .social .value, .social .mode {{ color: var(--orange); }}
    .wedge {{ margin-top: 18px; color: var(--muted); font-size: 14px; }}
    .charts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .chart {{ overflow: hidden; }}
    .chart svg {{ display: block; width: 100%; height: auto; color: var(--text); }}
    .chart svg text {{ fill: var(--text); }}
    .chart svg line {{ stroke: var(--muted); }}
    .thresholds {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .threshold {{ padding: 20px; border-top: 4px solid var(--blue); }}
    .threshold:nth-child(2) {{ border-color: var(--orange); }}
    .threshold:nth-child(3) {{ border-color: var(--green); }}
    .threshold strong {{ display: block; margin: 3px 0 8px; font-size: 25px; }}
    .threshold p {{ color: var(--muted); font-size: 14px; }}
    .flow {{ display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items: stretch; gap: 12px; }}
    .flow-step {{ background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 20px; }}
    .arrow {{ align-self: center; color: var(--muted); font-size: 24px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 18px; margin: 0 0 14px; color: var(--muted); font-size: 13px; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }}
    details {{ background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 16px 20px; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    details ul {{ color: var(--muted); margin-bottom: 0; }}
    footer {{ margin-top: 44px; padding-top: 22px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }}
    footer a {{ color: var(--blue); }}
    @media (max-width: 820px) {{
      main {{ width: min(100% - 22px, 720px); padding-top: 30px; }}
      .cards, .charts, .thresholds {{ grid-template-columns: 1fr; }}
      .flow {{ grid-template-columns: 1fr; }}
      .arrow {{ transform: rotate(90deg); justify-self: center; }}
    }}
    @media print {{
      :root {{ --bg: #fff; --surface: #fff; --surface-soft: #f3f5f7; --text: #111; --muted: #555; --line: #ccc; }}
      main {{ width: 100%; padding: 20px; }}
      .card {{ box-shadow: none; break-inside: avoid; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">中国情景 · 24小时合成原型 · 24个基础场景</div>
    <h1>企业会选择本地 AI，但社会成本可能更偏好云端</h1>
    <p class="lead">比较企业本地购买双L20服务器、订阅同型号云端实例和50%混合部署，并同时计算企业直接成本、底层社会资源成本以及电网扩容。</p>
    <div class="thesis">当前云价下，本地方案对企业最便宜；但云端较高利用率和较低PUE节省的服务器与能源资源，暂时超过了本地光伏储能带来的电网容量收益。</div>
  </header>

  <section>
    <h2>最核心的成本分歧</h2>
    <div class="cards">
      <article class="card">
        <div class="case-title"><h3>制造企业</h3><span>既有DER · 容量紧张</span></div>
        <div class="comparison">
          <div class="metric private"><div class="label">企业成本最低</div><div class="value">168.5万元</div><div class="mode">本地部署 / 年</div></div>
          <div class="metric social"><div class="label">社会成本最低</div><div class="value">110.1万元</div><div class="mode">云端部署 / 年</div></div>
        </div>
        <p class="wedge">企业选择与社会最优之间约有57.5万元/年的资源成本差距。</p>
      </article>
      <article class="card">
        <div class="case-title"><h3>办公／公共服务园区</h3><span>既有DER · 容量紧张</span></div>
        <div class="comparison">
          <div class="metric private"><div class="label">企业成本最低</div><div class="value">1321.6万元</div><div class="mode">本地部署 / 年</div></div>
          <div class="metric social"><div class="label">社会成本最低</div><div class="value">869.7万元</div><div class="mode">云端部署 / 年</div></div>
        </div>
        <p class="wedge">企业选择与社会最优之间约有443.9万元/年的资源成本差距。</p>
      </article>
    </div>
  </section>

  <section>
    <h2>为什么会出现分歧</h2>
    <div class="flow">
      <div class="flow-step"><h3>企业账</h3><p>本地服务器以资产投资年化，当前阿里云双L20租金较高，因此本地直接成本更低。</p></div>
      <div class="arrow">→</div>
      <div class="flow-step"><h3>物理系统</h3><p>云端利用率65%、PUE 1.2；本地利用率50%、PUE 1.6，导致本地需要更多服务器并消耗更多电力。</p></div>
      <div class="arrow">→</div>
      <div class="flow-step"><h3>电网位置</h3><p>本地可复用企业接入余量和屋顶DER；云端形成集中负荷，但当前扩容成本不足以反转资源成本排序。</p></div>
    </div>
  </section>

  <section>
    <h2>主要结果图</h2>
    <div class="legend"><span><i class="dot" style="background:var(--blue)"></i>本地</span><span><i class="dot" style="background:var(--orange)"></i>云端</span><span><i class="dot" style="background:var(--green)"></i>50%混合</span></div>
    <div class="charts">
      <article class="card chart">{direct}</article>
      <article class="card chart">{social}</article>
      <article class="card chart">{grid}</article>
      <article class="card chart">{switch}</article>
    </div>
  </section>

  <section>
    <h2>决定结论的三个阈值</h2>
    <div class="thresholds">
      <article class="card threshold"><span>企业成本切换</span><strong>33%–38%</strong><p>本地服务器利用率超过这一范围后，企业直接成本开始低于云端。</p></article>
      <article class="card threshold"><span>社会成本切换</span><strong>约77%</strong><p>本地利用率需要接近云端水平，社会资源成本才可能反转为本地更低。</p></article>
      <article class="card threshold"><span>云价格切换</span><strong>现价的66%–75%</strong><p>云价降至这一范围后，企业账面的最优选择将从本地转向云端。</p></article>
    </div>
  </section>

  <section>
    <h2>光伏储能带来的容量价值</h2>
    <div class="cards">
      <article class="card"><h3>制造企业</h3><p>屋顶面积758 m²，经90%可用率、22%组件效率和80%实现比例修正，光伏上限约120 kWp。本地模式新增接入容量由21.2 kW降至0。</p></article>
      <article class="card"><h3>办公／公共服务园区</h3><p>屋顶面积3030 m²，同口径修正后光伏上限约480 kWp。本地模式新增接入容量由185.2 kW降至0。</p></article>
    </div>
  </section>

  <section>
    <details>
      <summary>模型边界与解读限制</summary>
      <ul>
        <li>结果来自24小时合成代表日，不是现实企业或全国平均估计。</li>
        <li>本地与云端均采用2×L20 48GB；AI曲线是满载等效IT任务，不是实测吞吐量。</li>
        <li>社会成本不计完整云服务费，使用底层服务器、电力、设施和电网扩容代理重建。</li>
        <li>既有DER不重复计入AI投资；“为AI新建DER”的年化成本另列。</li>
        <li>零新增容量表示当前小时级筛查未超过接入上限，不代表已经取消具体电网工程。</li>
      </ul>
    </details>
  </section>

  <footer>
    数据文件：<a href="china_prototype_24_scenarios.csv">24场景结果</a> ·
    <a href="china_prototype_sensitivity.csv">敏感性结果</a> ·
    <a href="china_prototype_preliminary_findings.md">初步研究笔记</a>
  </footer>
</main>
</body>
</html>
"""
    OUTPUT.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--china',type=Path,required=True); p.add_argument('--us',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--findings-output',type=Path,required=True); p.add_argument('--done-output',type=Path,required=True); a=p.parse_args()
cn_all=pd.read_csv(a.china,encoding='utf-8-sig'); cn=cn_all.query("owned_architecture=='IF'").copy(); us=pd.read_csv(a.us,encoding='utf-8-sig').query("parameter_case=='base' and cpu_server_price_case=='base'")
out=pd.concat([
 cn.assign(country='China',local_currency='CNY').rename(columns={'local_joint_physical_annual_cost_rmb':'local_annual_cost','cloud_total_annual_cost_rmb':'cloud_annual_cost'}),
 us.assign(country='US',local_currency='USD').rename(columns={'local_total_annual_cost_usd':'local_annual_cost','cloud_total_annual_cost_usd':'cloud_annual_cost'})
],ignore_index=True)[['country','provider','local_currency','local_annual_cost','cloud_annual_cost','cloud_to_local_cost_ratio','local_savings_vs_cloud_fraction','aggregation_boundary']]
out.insert(1,'owned_architecture',['IF']*len(cn)+['IF']*len(us))
a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False,encoding='utf-8-sig')
cn_energy=float(cn['local_total_facility_energy_twh'].iloc[0]); us_base=pd.read_csv(a.us,encoding='utf-8-sig').query("parameter_case=='base' and cpu_server_price_case=='base'"); us_energy=float(us_base['annual_facility_energy_twh'].iloc[0])
cn_alibaba=float(cn.loc[cn.provider.eq('Alibaba Cloud'),'local_savings_vs_cloud_fraction'].iloc[0])
cn_deepseek=float(cn.loc[cn.provider.eq('DeepSeek'),'local_savings_vs_cloud_fraction'].iloc[0])
lines=[
 '# 中美制造业 AI 异构硬件成本主要结果', '',
 '## 基准情景', '',
 '| 国家 | 本地年度成本 | 云服务商 | 云端年度付款 | 云/本地 | 本地相对云节省 |',
 '|---|---:|---|---:|---:|---:|',
]
for row in out.itertuples(index=False):
    scale=1e9; unit='十亿元人民币/年' if row.local_currency=='CNY' else '十亿美元/年'
    lines.append(f'| {row.country} | {row.local_annual_cost/scale:.3f} {unit} | {row.provider} | {row.cloud_annual_cost/scale:.3f} {unit} | {row.cloud_to_local_cost_ratio:.3f} | {row.local_savings_vs_cloud_fraction:.1%} |')
lines += [
 '', '## 结论', '',
        f'- 中国基准需求下，本地异构部署耗电 {cn_energy:.3f} TWh/年。本地联合物理优化成本为 {cn.local_joint_physical_annual_cost_rmb.iloc[0]/1e9:.3f} 十亿元人民币/年；相对阿里云节省 {cn_alibaba:.1%}，相对 DeepSeek 节省 {cn_deepseek:.1%}。',
 f'- 美国基准需求下，本地异构部署耗电 {us_energy:.3f} TWh/年。本地成本为 {us_base.local_total_annual_cost_usd.iloc[0]/1e9:.3f} 十亿美元/年；三家云服务商的云/本地比为 {us_base.cloud_to_local_cost_ratio.min():.3f}–{us_base.cloud_to_local_cost_ratio.max():.3f}，本地节省 {us_base.local_savings_vs_cloud_fraction.min():.1%}–{us_base.local_savings_vs_cloud_fraction.max():.1%}。',
 '- 以上是企业付款口径，不是云商底层资源成本；中美分别满足本国制造业 AI 需求，且服务器寿命统一为 5 年、每个硬件池只保留一次 10% 裕量。',
 '- 结果对 CPU/GPU 路由、CPU 相对服务时间、本地 CPU 采购价和云端 CPU 容量价格敏感。美国高 CPU 采购价情景下，本地仍节省约 35%，但该结果仍依赖物理核比例代理，不能替代匹配任务质量与 SLA 的吞吐基准。',
]
a.findings_output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
a.done_output.write_text(json.dumps({'status':'complete_validated_country_heterogeneous_cost_summary','rows':len(out),'cross_currency_cost_comparison':False,'findings':a.findings_output.as_posix()},ensure_ascii=False,indent=2)+'\n')

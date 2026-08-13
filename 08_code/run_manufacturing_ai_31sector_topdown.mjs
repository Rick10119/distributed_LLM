import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

// External-alignment input only. This script allocates 8/14/28 TWh reference
// scenarios across industries for calibration diagnostics; v0.2.0 does not use
// these allocations as architecture electricity or an equal-energy constraint.

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const dataDir = path.join(root, "02_data");
const resultDir = path.join(root, "05_results");

const nationalAdoption2023 = 0.209347;
const nationalAdoption2030 = 0.480046;
const efficiency2030 = 3.0;
const hardware = {
  gpusPerServer: 2,
  targetUtilisation: 0.65,
  reserveFactor: 1.10,
  serverIdleKw: 0.42,
  serverFullKw: 1.30,
  pooledPue: 1.40,
  edgeActiveKw: 0.040,
  edgeIdleKw: 0.010,
  edgeActiveHours: 16,
  edgeAuxFactor: 1.15,
};

const tasks = [
  { id: "office", c36h2030: 10 },
  { id: "agent", c36h2030: 12 },
  { id: "vision", c36h2030: 4 },
  { id: "maintenance", c36h2030: 8 },
  { id: "scheduling", c36h2030: 7 },
  { id: "simulation", c36h2030: 16 },
];

const templates = {
  C13: { diffusionSpeed: .85, edge2030: 4, mult: [.75, 1.1, 1.2, .8, 1.1, .4] },
  C14: { diffusionSpeed: .90, edge2030: 3, mult: [.8, 1.1, 1.0, .8, 1.1, .5] },
  C17: { diffusionSpeed: .80, edge2030: 4, mult: [.9, 1.0, 1.2, .7, .9, .4] },
  C22: { diffusionSpeed: .95, edge2030: 3, mult: [.65, .8, .7, 1.4, 1.4, .9] },
  C26: { diffusionSpeed: 1.00, edge2030: 2, mult: [.7, .8, .5, 1.5, 1.3, 1.5] },
  C27: { diffusionSpeed: 1.05, edge2030: 5, mult: [1.0, 1.1, 1.2, 1.0, 1.2, 1.6] },
  C31: { diffusionSpeed: .90, edge2030: 3, mult: [.55, .7, .6, 1.6, 1.4, 1.1] },
  C34: { diffusionSpeed: 1.05, edge2030: 6, mult: [.9, 1.0, 1.4, 1.3, 1.1, 1.4] },
  C36: { diffusionSpeed: 1.10, edge2030: 6, mult: [1, 1, 1, 1, 1, 1] },
  C37: { diffusionSpeed: 1.00, edge2030: 6, mult: [.9, 1.0, 1.3, 1.2, 1.1, 1.8] },
  C38: { diffusionSpeed: 1.05, edge2030: 7, mult: [.9, 1.0, 1.6, 1.1, 1.1, 1.4] },
  C39: { diffusionSpeed: 1.15, edge2030: 8, mult: [.9, .9, 2.0, 1.0, 1.0, 1.2] },
  C40: { diffusionSpeed: 1.10, edge2030: 6, mult: [1.0, 1.0, 1.5, 1.0, 1.0, 1.6] },
};

function parseCsvText(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(field); field = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some(value => value !== "")) rows.push(row);
      row = [];
    } else field += char;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const headers = rows.shift().map(header => header.replace(/^﻿/, ""));
  return rows.map(values => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

const workloadRows = parseCsvText(await fs.readFile(path.join(dataDir, "raw", "curated", "china_manufacturing_ai_workload_parameters.csv"), "utf8"));
const taskExposureMap = {
  office: "office_rag_copilot",
  agent: "business_agent",
  vision: "vlm_anomaly_review",
  maintenance: "predictive_maintenance",
  scheduling: "production_scheduling",
  simulation: "digital_twin_simulation",
};
function taskExposure(industryCode, workloadTask) {
  const selected = workloadRows.filter(row =>
    row.industry_code === industryCode && row.task_type === workloadTask &&
    row.scenario_year === "2030" && row.scenario_level === "base"
  );
  if (!selected.length) throw new Error(`Missing 2030 base workload rows for ${industryCode}/${workloadTask}`);
  return selected.reduce((sum, row) => sum
    + Number(row.ai_adoption_rate)
    * Number(row.active_user_or_equipment_share)
    * Number(row.service_intensity_per_driver_day)
    * Number(row.calls_per_task || 1), 0) / selected.length;
}

// 对四个高影响代理行业做内部一致性自动校准。相同任务内与C36比较，避免跨单位相加。
// 普通机器视觉只校准边缘点位；中央vision仍使用VLM异常复核暴露量。
const autoCalibrationBase = {
  C29: { diffusionSpeed: templates.C34.diffusionSpeed },
  C30: { diffusionSpeed: templates.C31.diffusionSpeed },
  C33: { diffusionSpeed: templates.C34.diffusionSpeed },
  C35: { diffusionSpeed: templates.C34.diffusionSpeed },
};
for (const [industryCode, base] of Object.entries(autoCalibrationBase)) {
  const mult = Object.values(taskExposureMap).map(workloadTask =>
    taskExposure(industryCode, workloadTask) / taskExposure("C36", workloadTask)
  );
  const edgeRatio = taskExposure(industryCode, "machine_vision_qc") / taskExposure("C36", "machine_vision_qc");
  templates[industryCode] = { diffusionSpeed: base.diffusionSpeed, edge2030: 6 * edgeRatio, mult };
}

// 2023规上企业数和“使用至少一种AI”的企业数来自五经普/中国统计年鉴交叉核对。
// proxyTemplate只决定2030年采用企业内部的六任务组合、扩散速度和边缘点位。
const industries = [
  ["C13", "农副食品加工业", 25581, 4753, "C13", "direct"],
  ["C14", "食品制造业", 10075, 2052, "C14", "direct"],
  ["C15", "酒、饮料和精制茶制造业", 5860, 1277, "C14", "proxy_batch_food"],
  ["C16", "烟草制品业", 194, 71, "C14", "proxy_batch_consumer"],
  ["C17", "纺织业", 20858, 3269, "C17", "direct"],
  ["C18", "纺织服装、服饰业", 13346, 1738, "C17", "proxy_light_textile"],
  ["C19", "皮革、毛皮、羽毛及其制品和制鞋业", 8566, 992, "C17", "proxy_light_textile"],
  ["C20", "木材加工和木、竹、藤、棕、草制品业", 12887, 1471, "C17", "proxy_light_manufacturing"],
  ["C21", "家具制造业", 7349, 1294, "C34", "proxy_discrete_product"],
  ["C22", "造纸和纸制品业", 7859, 1408, "C22", "direct"],
  ["C23", "印刷和记录媒介复制业", 6696, 1420, "C39", "proxy_visual_digital"],
  ["C24", "文教、工美、体育和娱乐用品制造业", 10505, 1709, "C17", "proxy_light_manufacturing"],
  ["C25", "石油、煤炭及其他燃料加工业", 2324, 500, "C26", "proxy_continuous_process"],
  ["C26", "化学原料和化学制品制造业", 25489, 5853, "C26", "direct"],
  ["C27", "医药制造业", 9563, 2474, "C27", "direct"],
  ["C28", "化学纤维制造业", 2362, 469, "C26", "proxy_continuous_process"],
  ["C29", "橡胶和塑料制品业", 26495, 4877, "C29", "auto_calibrated_internal_task_exposure"],
  ["C30", "非金属矿物制品业", 49121, 8622, "C30", "auto_calibrated_internal_task_exposure"],
  ["C31", "黑色金属冶炼和压延加工业", 6203, 1017, "C31", "direct"],
  ["C32", "有色金属冶炼和压延加工业", 10005, 1828, "C31", "proxy_metal_process"],
  ["C33", "金属制品业", 36000, 6429, "C33", "auto_calibrated_internal_task_exposure"],
  ["C34", "通用设备制造业", 34915, 8323, "C34", "direct"],
  ["C35", "专用设备制造业", 27065, 6868, "C35", "auto_calibrated_internal_task_exposure"],
  ["C36", "汽车制造业", 18899, 5497, "C36", "direct"],
  ["C37", "铁路、船舶、航空航天和其他运输设备制造业", 6164, 1323, "C37", "direct"],
  ["C38", "电气机械和器材制造业", 34242, 8188, "C38", "direct"],
  ["C39", "计算机、通信和其他电子设备制造业", 27776, 9104, "C39", "direct"],
  ["C40", "仪器仪表制造业", 7069, 2346, "C40", "direct"],
  ["C41", "其他制造业", 2173, 403, "C17", "proxy_light_manufacturing"],
  ["C42", "废弃资源综合利用业", 3578, 585, "C31", "proxy_material_process"],
  ["C43", "金属制品、机械和设备修理业", 835, 151, "C34", "proxy_equipment_service"],
].map(([code, name, firms, aiFirms, templateCode, parameterStatus]) => ({
  code, name, firms, aiFirms, adoption2023: aiFirms / firms, templateCode, parameterStatus,
}));

function odds(p) { return p / (1 - p); }
function invOdds(o) { return o / (1 + o); }
function adoption2030(p2023, speed) {
  const multiplier = odds(nationalAdoption2030) / odds(nationalAdoption2023);
  return invOdds(odds(p2023) * Math.pow(multiplier, speed));
}
function csvEscape(value) {
  const s = String(value ?? "");
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}
function toCsv(rows, headers) {
  return [headers.join(","), ...rows.map(row => headers.map(h => csvEscape(row[h])).join(","))].join("\n") + "\n";
}

const rows = industries.map(industry => {
  const template = templates[industry.templateCode];
  const adoption = adoption2030(industry.adoption2023, template.diffusionSpeed);
  const adoptingFirms = industry.firms * adoption;
  const perAdopterL20eq = tasks.reduce((sum, task, index) => sum + task.c36h2030 * template.mult[index], 0);
  const l20eqGpuHDay = adoptingFirms * perAdopterL20eq;
  const actualGpuHDay = l20eqGpuHDay / efficiency2030;
  const pooledServers = Math.ceil(actualGpuHDay / (24 * hardware.gpusPerServer * hardware.targetUtilisation) * hardware.reserveFactor);
  const dynamicKwPerGpu = (hardware.serverFullKw - hardware.serverIdleKw) / hardware.gpusPerServer;
  const centralAverageKw = (pooledServers * hardware.serverIdleKw + actualGpuHDay * dynamicKwPerGpu / 24) * hardware.pooledPue;
  const edgeDevices = adoptingFirms * template.edge2030;
  const edgeKwhDayPerDevice = (hardware.edgeActiveKw * hardware.edgeActiveHours + hardware.edgeIdleKw * (24 - hardware.edgeActiveHours)) * hardware.edgeAuxFactor;
  const edgeAverageKw = edgeDevices * edgeKwhDayPerDevice / 24;
  const totalAverageKw = centralAverageKw + edgeAverageKw;
  return {
    industry_code: industry.code,
    industry_name_cn: industry.name,
    above_size_firms_2023: industry.firms,
    ai_using_firms_2023: industry.aiFirms,
    any_ai_adoption_2023: industry.adoption2023,
    any_ai_adoption_2030: adoption,
    adopting_firms_2030: adoptingFirms,
    task_template: industry.templateCode,
    task_parameter_status: industry.parameterStatus,
    per_adopter_l20eq_gpu_h_day_2030: perAdopterL20eq,
    sector_l20eq_gpu_h_day_2030: l20eqGpuHDay,
    future_accelerator_gpu_h_day_2030: actualGpuHDay,
    central_average_facility_kw_2030: centralAverageKw,
    edge_average_kw_2030: edgeAverageKw,
    raw_bottomup_average_kw_2030: totalAverageKw,
    raw_bottomup_annual_gwh_2030: totalAverageKw * 8760 / 1e6,
  };
});

const rawTotalGwh = rows.reduce((sum, row) => sum + row.raw_bottomup_annual_gwh_2030, 0);
for (const row of rows) {
  row.manufacturing_weight = row.raw_bottomup_annual_gwh_2030 / rawTotalGwh;
  row.lower_8twh_allocation_twh = row.manufacturing_weight * 8;
  row.central_14twh_allocation_twh = row.manufacturing_weight * 14;
  row.upper_28twh_allocation_twh = row.manufacturing_weight * 28;
  row.central_average_load_mw = row.central_14twh_allocation_twh * 1e6 / 8760;
  row.evidence_status = row.task_parameter_status === "direct"
    ? "observed_stock_adoption_existing_task_parameters"
    : row.task_parameter_status === "auto_calibrated_internal_task_exposure"
      ? "observed_stock_adoption_internally_calibrated_task_parameters"
      : "observed_stock_adoption_proxy_task_parameters";
}

const parameterHeaders = [
  "industry_code", "industry_name_cn", "above_size_firms_2023", "ai_using_firms_2023",
  "any_ai_adoption_2023", "task_template", "task_parameter_status", "template_diffusion_speed",
  "template_edge_devices_per_adopter_2030", "office_multiplier", "agent_multiplier", "vision_multiplier",
  "maintenance_multiplier", "scheduling_multiplier", "simulation_multiplier", "observed_source", "parameter_note",
];
const parameterRows = industries.map(industry => {
  const template = templates[industry.templateCode];
  return {
    industry_code: industry.code,
    industry_name_cn: industry.name,
    above_size_firms_2023: industry.firms,
    ai_using_firms_2023: industry.aiFirms,
    any_ai_adoption_2023: industry.adoption2023,
    task_template: industry.templateCode,
    task_parameter_status: industry.parameterStatus,
    template_diffusion_speed: template.diffusionSpeed,
    template_edge_devices_per_adopter_2030: template.edge2030,
    office_multiplier: template.mult[0],
    agent_multiplier: template.mult[1],
    vision_multiplier: template.mult[2],
    maintenance_multiplier: template.mult[3],
    scheduling_multiplier: template.mult[4],
    simulation_multiplier: template.mult[5],
    observed_source: "02_data/china_manufacturing_ai_evidence_update.md",
    parameter_note: industry.parameterStatus === "direct"
      ? "沿用十三行业模型的独立六任务参数。"
      : industry.parameterStatus === "auto_calibrated_internal_task_exposure"
        ? "六任务乘数按本行业2030 base任务暴露量相对C36同任务暴露量自动校准；中央视觉使用VLM异常复核，边缘点位使用普通机器视觉。任务暴露量等于任务采用率×活跃驱动比例×每日强度×每任务调用数，并对large与sme等权平均。"
        : `六任务组合、扩散速度和边缘点位借用${industry.templateCode}模板；企业数和2023采用率仍为本行业观测。`,
  };
});

const resultHeaders = Object.keys(rows[0]);
const allocationPath = path.join(resultDir, "manufacturing_topdown_allocation_31sectors.csv");
const preCalibrationPath = path.join(resultDir, "manufacturing_topdown_allocation_31sectors_precalibration.csv");
try {
  await fs.access(preCalibrationPath);
} catch {
  try { await fs.copyFile(allocationPath, preCalibrationPath); } catch { /* first run without a baseline */ }
}
await fs.writeFile(path.join(dataDir, "china_manufacturing_ai_31sector_proxy_parameters.csv"), toCsv(parameterRows, parameterHeaders), "utf8");
await fs.writeFile(path.join(resultDir, "manufacturing_ai_31sector_bottomup_weights.csv"), toCsv(rows, resultHeaders), "utf8");
await fs.writeFile(
  allocationPath,
  toCsv([...rows].sort((a, b) => b.manufacturing_weight - a.manufacturing_weight), [
    "industry_code", "industry_name_cn", "manufacturing_weight", "lower_8twh_allocation_twh",
    "central_14twh_allocation_twh", "upper_28twh_allocation_twh", "central_average_load_mw",
    "task_template", "task_parameter_status", "evidence_status",
  ]),
  "utf8",
);

try {
  const baselineRows = parseCsvText(await fs.readFile(preCalibrationPath, "utf8"));
  const baselineMap = new Map(baselineRows.map(row => [row.industry_code, row]));
  const comparison = [...rows].sort((a, b) => b.manufacturing_weight - a.manufacturing_weight).map(row => {
    const old = baselineMap.get(row.industry_code);
    const oldWeight = Number(old?.manufacturing_weight ?? NaN);
    const oldCentral = Number(old?.central_14twh_allocation_twh ?? NaN);
    return {
      industry_code: row.industry_code,
      industry_name_cn: row.industry_name_cn,
      calibration_status: row.task_parameter_status,
      old_weight: oldWeight,
      calibrated_weight: row.manufacturing_weight,
      weight_change_percentage_points: (row.manufacturing_weight - oldWeight) * 100,
      old_central_14twh: oldCentral,
      calibrated_central_14twh: row.central_14twh_allocation_twh,
      central_change_twh: row.central_14twh_allocation_twh - oldCentral,
    };
  });
  await fs.writeFile(
    path.join(resultDir, "manufacturing_topdown_31sector_calibration_comparison.csv"),
    toCsv(comparison, Object.keys(comparison[0])),
    "utf8",
  );
} catch { /* comparison is optional when no pre-calibration output exists */ }

const ranked = [...rows].sort((a, b) => b.manufacturing_weight - a.manufacturing_weight);
const top10 = ranked.slice(0, 10);
const directShare = rows.filter(r => r.task_parameter_status === "direct").reduce((sum, r) => sum + r.manufacturing_weight, 0);
const calibratedShare = rows.filter(r => r.task_parameter_status === "auto_calibrated_internal_task_exposure").reduce((sum, r) => sum + r.manufacturing_weight, 0);
let calibratedFourOldTwh = NaN;
let calibratedFourNewTwh = rows.filter(r => r.task_parameter_status === "auto_calibrated_internal_task_exposure").reduce((sum, r) => sum + r.central_14twh_allocation_twh, 0);
try {
  const baselineRows = parseCsvText(await fs.readFile(preCalibrationPath, "utf8"));
  const targetCodes = new Set(Object.keys(autoCalibrationBase));
  calibratedFourOldTwh = baselineRows.filter(r => targetCodes.has(r.industry_code)).reduce((sum, r) => sum + Number(r.central_14twh_allocation_twh), 0);
} catch { /* no comparison baseline */ }
const note = `# 2030年制造业AI需求：31行业完整分配（筛查版）

## 结论

在“全国AI数据中心用电200 TWh、制造业份额为4%/7%/14%”的三情景下，制造业总量分别为8、14和28 TWh。本次已将该总量分配到C13—C43全部31个制造业大类。中心情景仍为14 TWh，对应制造业平均负荷约1.60 GW。

## 行业排序（中心情景）

| 排名 | 行业 | 制造业内部权重 | 14 TWh分配 | 平均负荷 |
|---:|---|---:|---:|---:|
${top10.map((r, i) => `| ${i + 1} | ${r.industry_code} ${r.industry_name_cn} | ${(r.manufacturing_weight * 100).toFixed(2)}% | ${r.central_14twh_allocation_twh.toFixed(2)} TWh | ${r.central_average_load_mw.toFixed(1)} MW |`).join("\n")}

前十行业合计占制造业需求的${(top10.reduce((s, r) => s + r.manufacturing_weight, 0) * 100).toFixed(1)}%。原十三行业独立任务参数覆盖完整权重的${(directShare * 100).toFixed(1)}%，C29、C30、C33、C35四个自动校准行业占${(calibratedShare * 100).toFixed(1)}%，其余行业继续使用透明的近邻生产类型模板。

四个高影响行业在中心情景中的合计分配由校准前的${Number.isFinite(calibratedFourOldTwh) ? calibratedFourOldTwh.toFixed(2) : "不可得"} TWh降至${calibratedFourNewTwh.toFixed(2)} TWh。下降来自行业自身任务暴露量普遍低于原先借用的C31/C34模板，而不是企业数或2023年采用率变化。

## 计算方法

1. 31行业均使用2023年规上企业数和应用至少一种AI的企业数观测。
2. 采用率按全国中间情景的odds增长外推到2030年；扩散速度沿用现有行业模板。
3. 原十三行业保留各自六任务参数。C29、C30、C33、C35按详细任务表中的“任务采用率×活跃驱动比例×每日强度×调用数”计算任务暴露量，并在相同任务内相对C36归一化；中央视觉使用VLM异常复核，普通机器视觉只校准边缘点位。其余十四行业继续借用近邻生产类型模板。
4. 由采用企业数、每采用企业六任务L20等效小时、未来效率和边缘视觉点位得到行业原始电力权重。
5. 只用该权重分配8/14/28 TWh的自上而下制造业总量；原始自下而上电量不作为全国总量预测。

## 解释边界

- 企业数和2023年任意AI采用率是行业观测；自动校准使用的是项目内情景参数，不是新的外部实测，只提高内部一致性。
- 权重体现规上企业，不含规模以下企业，也未显式加入就业、设备、研发人员、集团集中部署和多厂址结构。
- 年度电量分配不能直接说明各行业峰值。峰值还需要行业负荷曲线、可延迟任务和本地/云端部署比例。
- 8/14/28 TWh三种总量共享同一行业结构，因此总量情景不会改变行业排序；任务参数敏感性才会改变行业结构。

## 下一步最有价值的校准

下一步应把就业驱动、设备驱动和场址驱动的任务拆开，避免所有AI需求都随“采用企业数”同比例增长；同时为四个自动校准行业寻找企业或工厂级任务量案例，检验当前相对强度，而不是继续用行业政策或案例名单替代强度数据。
`;
await fs.writeFile(path.join(resultDir, "manufacturing_topdown_31sector_findings.md"), note, "utf8");

console.log(JSON.stringify({
  industryCount: rows.length,
  rawBottomupGwh: rawTotalGwh,
  weightSum: rows.reduce((sum, row) => sum + row.manufacturing_weight, 0),
  lowerTwhSum: rows.reduce((sum, row) => sum + row.lower_8twh_allocation_twh, 0),
  centralTwhSum: rows.reduce((sum, row) => sum + row.central_14twh_allocation_twh, 0),
  upperTwhSum: rows.reduce((sum, row) => sum + row.upper_28twh_allocation_twh, 0),
  directParameterWeight: directShare,
  internallyCalibratedWeight: calibratedShare,
  top10: top10.map(row => ({ code: row.industry_code, weight: row.manufacturing_weight, centralTwh: row.central_14twh_allocation_twh })),
}, null, 2));

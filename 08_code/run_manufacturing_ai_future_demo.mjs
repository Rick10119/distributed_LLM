import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const outputDir = path.join(root, "outputs", "manufacturing_ai_future_demo");
const dataDir = path.join(root, "02_data");
const resultsDir = path.join(root, "05_results");
const figureDir = path.join(resultsDir, "figures");

const national2023 = 0.209347;
const years = [2023, 2026, 2027, 2030];
const stateMultipliers = { S1: 0.15, S2: 1.0, S3: 3.0, S4: 7.0 };

const industries = [
  { code: "C14", name: "食品制造业", firms: 10075, adoption2023: 0.2037, depthFactor: 0.90, diffusionSpeed: 0.90, intensityExponent: 0.90, archetype: "食品与冷链；质检、供应链和经营任务" },
  { code: "C17", name: "纺织业", firms: 20858, adoption2023: 0.1567, depthFactor: 0.80, diffusionSpeed: 0.80, intensityExponent: 0.85, archetype: "轻工劳动密集；视觉、设计和订单任务" },
  { code: "C26", name: "化学原料和化学制品制造业", firms: 25489, adoption2023: 0.2296, depthFactor: 1.05, diffusionSpeed: 1.00, intensityExponent: 1.05, archetype: "连续流程；设备诊断、工艺优化和安全任务" },
  { code: "C36", name: "汽车制造业", firms: 18899, adoption2023: 0.2909, depthFactor: 1.15, diffusionSpeed: 1.10, intensityExponent: 1.10, archetype: "离散装备；视觉质检、排程、研发和供应链任务" },
  { code: "C39", name: "计算机、通信和其他电子设备制造业", firms: 27776, adoption2023: 0.3278, depthFactor: 1.25, diffusionSpeed: 1.15, intensityExponent: 1.15, archetype: "电子洁净制造；高频视觉、良率和工艺优化任务" },
];

const scenarios = {
  slow: {
    name: "较慢扩散/效率较快",
    nationalAdoption: { 2023: national2023, 2026: 0.30, 2027: 0.331079, 2030: 0.418037 },
    states: {
      2023: [0.60, 0.30, 0.09, 0.01], 2026: [0.50, 0.35, 0.13, 0.02],
      2027: [0.46, 0.36, 0.15, 0.03], 2030: [0.30, 0.40, 0.24, 0.06],
    },
    intensity: { 2023: 1.00, 2026: 1.15, 2027: 1.25, 2030: 1.60 },
    efficiency: { 2023: 1.00, 2026: 1.60, 2027: 1.90, 2030: 3.50 },
  },
  base: {
    name: "中间情景",
    nationalAdoption: { 2023: national2023, 2026: 0.32, 2027: 0.360440, 2030: 0.480046 },
    states: {
      2023: [0.60, 0.30, 0.09, 0.01], 2026: [0.42, 0.38, 0.17, 0.03],
      2027: [0.34, 0.39, 0.22, 0.05], 2030: [0.15, 0.35, 0.35, 0.15],
    },
    intensity: { 2023: 1.00, 2026: 1.35, 2027: 1.55, 2030: 2.50 },
    efficiency: { 2023: 1.00, 2026: 1.50, 2027: 1.75, 2030: 3.00 },
  },
  fast: {
    name: "较快扩散/服务深化较快",
    nationalAdoption: { 2023: national2023, 2026: 0.35, 2027: 0.404088, 2030: 0.567981 },
    states: {
      2023: [0.60, 0.30, 0.09, 0.01], 2026: [0.35, 0.38, 0.22, 0.05],
      2027: [0.25, 0.37, 0.29, 0.09], 2030: [0.05, 0.20, 0.45, 0.30],
    },
    intensity: { 2023: 1.00, 2026: 1.60, 2027: 2.00, 2030: 4.00 },
    efficiency: { 2023: 1.00, 2026: 1.35, 2027: 1.50, 2030: 2.20 },
  },
};

function odds(p) { return p / (1 - p); }
function invOdds(o) { return o / (1 + o); }
function adoptionFor(industry, scenario, year) {
  if (year === 2023) return industry.adoption2023;
  const nationalMultiplier = odds(scenario.nationalAdoption[year]) / odds(national2023);
  return invOdds(odds(industry.adoption2023) * Math.pow(nationalMultiplier, industry.diffusionSpeed));
}
function maturityAverage(shares) {
  return shares.reduce((total, share, i) => total + share * stateMultipliers[`S${i + 1}`], 0);
}
function csvEscape(v) {
  const s = String(v ?? "");
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}
function toCsv(rows, headers) {
  return [headers.join(","), ...rows.map(r => headers.map(h => csvEscape(r[h])).join(","))].join("\n") + "\n";
}

const parameterRows = [];
for (const industry of industries) {
  parameterRows.push({
    record_type: "industry", scenario: "all", year: "", industry_code: industry.code,
    industry_name_cn: industry.name, parameter: "industry_inputs", value: "",
    unit: "", s1_share: "", s2_share: "", s3_share: "", s4_share: "",
    observed_firms_2023: industry.firms, observed_any_ai_rate_2023: industry.adoption2023,
    depth_factor: industry.depthFactor, diffusion_speed: industry.diffusionSpeed,
    intensity_growth_exponent: industry.intensityExponent,
    evidence_status: "firms_and_adoption_observed; factors_assumed",
    source_url: "https://www.stats.gov.cn/sj/pcsj/jjpc/5jp/zk/left.htm",
    notes: industry.archetype,
  });
}
for (const [scenarioId, scenario] of Object.entries(scenarios)) {
  for (const year of years) {
    const shares = scenario.states[year];
    parameterRows.push({
      record_type: "scenario_year", scenario: scenarioId, year, industry_code: "ALL",
      industry_name_cn: "全部示范行业", parameter: scenario.name,
      value: scenario.nationalAdoption[year], unit: "fraction",
      s1_share: shares[0], s2_share: shares[1], s3_share: shares[2], s4_share: shares[3],
      observed_firms_2023: "", observed_any_ai_rate_2023: "", depth_factor: "",
      diffusion_speed: "", intensity_growth_exponent: "",
      service_intensity_multiplier: scenario.intensity[year],
      compute_efficiency_multiplier: scenario.efficiency[year],
      evidence_status: year === 2023 ? "observed_anchor_plus_assumed_state_mix" : "scenario_assumption",
      source_url: year === 2026 ? "https://wap.miit.gov.cn/jgsj/ghs/gzdt/art/2026/art_a132d7b17b774c6b9ff72d7a0befe158.html" : "",
      notes: "状态份额均为采用企业内部条件分布；S2=1 服务当量，S1=0.15、S3=3、S4=7。",
    });
  }
}

const results = [];
for (const industry of industries) {
  for (const [scenarioId, scenario] of Object.entries(scenarios)) {
    const temp = [];
    for (const year of years) {
      const adoption = adoptionFor(industry, scenario, year);
      const shares = scenario.states[year];
      const maturity = maturityAverage(shares) * industry.depthFactor;
      const intensity = Math.pow(scenario.intensity[year], industry.intensityExponent);
      const service = industry.firms * adoption * maturity * intensity;
      const compute = service / scenario.efficiency[year];
      temp.push({
        industry_code: industry.code, industry_name_cn: industry.name, scenario: scenarioId,
        scenario_name_cn: scenario.name, year, observed_firms_2023: industry.firms,
        any_ai_adoption_rate: adoption, adopting_firms: industry.firms * adoption,
        s1_share: shares[0], s2_share: shares[1], s3_share: shares[2], s4_share: shares[3],
        industry_depth_factor: industry.depthFactor,
        maturity_service_multiplier: maturity, service_intensity_multiplier: intensity,
        compute_efficiency_multiplier: scenario.efficiency[year],
        service_demand_units: service, compute_resource_units: compute,
        service_demand_index_2023_100: 0, compute_resource_index_2023_100: 0,
        evidence_status: "conditional_scenario_demo",
      });
    }
    const serviceBase = temp[0].service_demand_units;
    const computeBase = temp[0].compute_resource_units;
    for (const row of temp) {
      row.service_demand_index_2023_100 = 100 * row.service_demand_units / serviceBase;
      row.compute_resource_index_2023_100 = 100 * row.compute_resource_units / computeBase;
      results.push(row);
    }
  }
}

const parameterHeaders = [
  "record_type", "scenario", "year", "industry_code", "industry_name_cn", "parameter", "value", "unit",
  "s1_share", "s2_share", "s3_share", "s4_share", "observed_firms_2023", "observed_any_ai_rate_2023",
  "depth_factor", "diffusion_speed", "intensity_growth_exponent", "service_intensity_multiplier",
  "compute_efficiency_multiplier", "evidence_status", "source_url", "notes",
];
const resultHeaders = Object.keys(results[0]);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(figureDir, { recursive: true });
await fs.writeFile(path.join(dataDir, "china_manufacturing_ai_future_scenario_parameters.csv"), toCsv(parameterRows, parameterHeaders), "utf8");
await fs.writeFile(path.join(resultsDir, "manufacturing_ai_future_demo_results.csv"), toCsv(results, resultHeaders), "utf8");

const workbook = Workbook.create();
const readme = workbook.worksheets.add("说明");
const parameters = workbook.worksheets.add("参数");
const calc = workbook.worksheets.add("计算结果");
const dashboard = workbook.worksheets.add("结果概览");
for (const sheet of [readme, parameters, calc, dashboard]) sheet.showGridLines = false;

readme.getRange("A1:H1").merge();
readme.getRange("A1").values = [["中国制造业未来 AI 需求情景 Demo"]];
readme.getRange("A1:H1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF", size: 16 }, rowHeight: 30 };
readme.getRange("A3:B11").values = [
  ["模型用途", "展示典型行业 2023—2030 年 AI 服务需求与计算资源需求的条件变化，不是点预测。"],
  ["行业", "食品、纺织、化工、汽车、电子设备。"],
  ["采用率", "2023 使用行业观测；未来按全国慢/中/快路径，以行业扩散速度调整 odds。"],
  ["成熟度", "S1 试点、S2 少数功能常态使用、S3 多功能整合、S4 生产闭环。"],
  ["服务需求", "采用企业数 × 成熟度服务当量 × 行业内使用强度。"],
  ["计算资源", "服务需求 ÷ 计算效率；用于说明效率进步可能抵消部分服务增长。"],
  ["单位", "service/compute units 均为归一化当量，不是 token、GPU·h 或电量。"],
  ["证据边界", "行业企业数与 2023 采用率为观测；成熟度、扩散速度、强度和效率均为透明情景假设。"],
  ["更新时间", "2026-08-09"],
];
readme.getRange("A3:A11").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
readme.getRange("A3:B11").format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
readme.getRange("A3:A11").format.columnWidth = 18;
readme.getRange("B3:B11").format.columnWidth = 82;
readme.getRange("B3:B11").format.wrapText = true;
readme.getRange("A3:B11").format.rowHeight = 34;

const paramMatrix = [parameterHeaders, ...parameterRows.map(r => parameterHeaders.map(h => r[h] ?? ""))];
parameters.getRangeByIndexes(0, 0, paramMatrix.length, parameterHeaders.length).values = paramMatrix;
parameters.getRangeByIndexes(0, 0, 1, parameterHeaders.length).format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" }, wrapText: true };
parameters.freezePanes.freezeRows(1);
parameters.getRangeByIndexes(1, 8, paramMatrix.length - 1, 4).format.numberFormat = "0.0%";
parameters.getRangeByIndexes(1, 13, paramMatrix.length - 1, 1).format.numberFormat = "0.0%";
parameters.getUsedRange().format.autofitColumns();
parameters.getRange("E:E").format.columnWidth = 28;
parameters.getRange("U:V").format.columnWidth = 48;
parameters.getRange("U:V").format.wrapText = true;
parameters.tables.add(`A1:V${paramMatrix.length}`, true, "FutureScenarioParameters");

const calcHeaders = resultHeaders;
const derivedHeaders = new Set([
  "adopting_firms", "maturity_service_multiplier", "service_demand_units",
  "compute_resource_units", "service_demand_index_2023_100", "compute_resource_index_2023_100",
]);
const calcRows = results.map(r => calcHeaders.map(h => derivedHeaders.has(h) ? null : r[h]));
calc.getRangeByIndexes(0, 0, 1, calcHeaders.length).values = [calcHeaders];
calc.getRangeByIndexes(1, 0, calcRows.length, calcHeaders.length).values = calcRows;
for (let i = 0; i < calcRows.length; i++) {
  const row = i + 2;
  const baseRow = Math.floor(i / 4) * 4 + 2;
  calc.getRange(`H${row}`).formulas = [[`=F${row}*G${row}`]];
  calc.getRange(`N${row}`).formulas = [[`=(I${row}*0.15+J${row}+K${row}*3+L${row}*7)*M${row}`]];
  calc.getRange(`Q${row}`).formulas = [[`=H${row}*N${row}*O${row}`]];
  calc.getRange(`R${row}`).formulas = [[`=Q${row}/P${row}`]];
  calc.getRange(`S${row}`).formulas = [[`=Q${row}/Q$${baseRow}*100`]];
  calc.getRange(`T${row}`).formulas = [[`=R${row}/R$${baseRow}*100`]];
}
calc.getRangeByIndexes(0, 0, 1, calcHeaders.length).format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" }, wrapText: true };
calc.freezePanes.freezeRows(1);
calc.getRangeByIndexes(1, 6, calcRows.length, 1).format.numberFormat = "0.0%";
calc.getRangeByIndexes(1, 8, calcRows.length, 4).format.numberFormat = "0.0%";
calc.getRangeByIndexes(1, 7, calcRows.length, 1).format.numberFormat = "#,##0";
calc.getRangeByIndexes(1, 12, calcRows.length, 4).format.numberFormat = "0.00";
calc.getRangeByIndexes(1, 16, calcRows.length, 2).format.numberFormat = "#,##0.0";
calc.getRangeByIndexes(1, 18, calcRows.length, 2).format.numberFormat = "0.0";
calc.getUsedRange().format.autofitColumns();
calc.getRange("B:B").format.columnWidth = 30;
calc.tables.add(`A1:U${calcRows.length + 1}`, true, "FutureDemoResults");

dashboard.getRange("A1:N1").merge();
dashboard.getRange("A1").values = [["典型制造行业未来 AI 需求变化（中间情景）"]];
dashboard.getRange("A1:N1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF", size: 15 }, rowHeight: 28 };
dashboard.getRange("A3:F8").values = [["行业", "2023采用率", "2030采用率", "2030服务指数", "2030计算指数", "解释"]];
const base2030 = results.filter(r => r.scenario === "base" && r.year === 2030);
const summaryRows = base2030.map(r => [
  r.industry_name_cn, industries.find(x => x.code === r.industry_code).adoption2023,
  r.any_ai_adoption_rate, r.service_demand_index_2023_100, r.compute_resource_index_2023_100,
  r.compute_resource_index_2023_100 < r.service_demand_index_2023_100 ? "效率抵消部分服务增长" : "计算资源与服务同步增长",
]);
dashboard.getRange("A4:F8").values = summaryRows.map(r => [r[0], null, null, null, null, r[5]]);
for (let i = 0; i < base2030.length; i++) {
  const resultIndex = results.indexOf(base2030[i]);
  const resultRow = resultIndex + 2;
  const dashboardRow = i + 4;
  const industry = industries.find(x => x.code === base2030[i].industry_code);
  dashboard.getRange(`B${dashboardRow}:E${dashboardRow}`).formulas = [[
    `=${industry.adoption2023}`,
    `='计算结果'!G${resultRow}`,
    `='计算结果'!S${resultRow}`,
    `='计算结果'!T${resultRow}`,
  ]];
}
dashboard.getRange("A3:F3").format = { fill: "#5B9BD5", font: { bold: true, color: "#FFFFFF" } };
dashboard.getRange("A3:F8").format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
dashboard.getRange("B4:C8").format.numberFormat = "0.0%";
dashboard.getRange("D4:E8").format.numberFormat = "0";
dashboard.getRange("A:A").format.columnWidth = 31;
dashboard.getRange("B:E").format.columnWidth = 15;
dashboard.getRange("F:F").format.columnWidth = 27;

const base = results.filter(r => r.scenario === "base");
const serviceChartTable = [["年份", ...industries.map(i => i.name)]];
const computeChartTable = [["年份", ...industries.map(i => i.name)]];
for (const year of years) {
  serviceChartTable.push([year, ...industries.map(i => base.find(r => r.industry_code === i.code && r.year === year).service_demand_index_2023_100)]);
  computeChartTable.push([year, ...industries.map(i => base.find(r => r.industry_code === i.code && r.year === year).compute_resource_index_2023_100)]);
}
dashboard.getRange("A11:F15").values = serviceChartTable.map((r, i) => i === 0 ? r : [r[0], null, null, null, null, null]);
dashboard.getRange("H11:M15").values = computeChartTable.map((r, i) => i === 0 ? r : [r[0], null, null, null, null, null]);
for (let yi = 0; yi < years.length; yi++) {
  for (let ii = 0; ii < industries.length; ii++) {
    const resultIndex = results.findIndex(r => r.scenario === "base" && r.year === years[yi] && r.industry_code === industries[ii].code);
    const resultRow = resultIndex + 2;
    dashboard.getCell(11 + yi, 1 + ii).formulas = [[`='计算结果'!S${resultRow}`]];
    dashboard.getCell(11 + yi, 8 + ii).formulas = [[`='计算结果'!T${resultRow}`]];
  }
}
dashboard.getRange("A11:F11").format = { fill: "#D9EAF7", font: { bold: true } };
dashboard.getRange("H11:M11").format = { fill: "#E2F0D9", font: { bold: true } };
dashboard.getRange("B12:F15").format.numberFormat = "0";
dashboard.getRange("I12:M15").format.numberFormat = "0";

const serviceChart = dashboard.charts.add("line", dashboard.getRange("A11:F15"));
serviceChart.title = "AI 服务需求指数（2023=100）";
serviceChart.hasLegend = true;
serviceChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
serviceChart.yAxis = { numberFormatCode: "0", min: 0 };
serviceChart.setPosition("A18", "G35");

const computeChart = dashboard.charts.add("line", dashboard.getRange("H11:M15"));
computeChart.title = "计算资源需求指数（2023=100）";
computeChart.hasLegend = true;
computeChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
computeChart.yAxis = { numberFormatCode: "0", min: 0 };
computeChart.setPosition("H18", "N35");

dashboard.getRange("A37:N39").merge();
dashboard.getRange("A37").values = [["说明：服务需求指数反映采用企业数量、成熟度和使用频率；计算资源指数进一步扣除硬件与算法效率。两者均为条件情景，不是全国 GPU、电量或投资预测。"]];
dashboard.getRange("A37:N39").format = { fill: "#FFF2CC", font: { color: "#7F6000" }, wrapText: true, rowHeight: 24 };

const inspect = await workbook.inspect({ kind: "table", range: "结果概览!A1:N39", include: "values,formulas", tableMaxRows: 39, tableMaxCols: 14, maxChars: 9000 });
console.log(inspect.ndjson);
const formulaInspect = await workbook.inspect({ kind: "formula", sheetId: "计算结果", range: "H1:T18", maxChars: 5000, options: { maxResults: 100 } });
console.log(formulaInspect.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula errors" });
console.log(errors.ndjson);

const renderJobs = [
  ["说明", "A1:H11", "preview_readme.png"],
  ["参数", "A1:V18", "preview_parameters.png"],
  ["计算结果", "A1:U18", "preview_calculations.png"],
  ["结果概览", "A1:N39", "preview_dashboard.png"],
];
for (const [sheetName, range, fileName] of renderJobs) {
  const preview = await workbook.render({ sheetName, range, scale: 1.5, format: "png" });
  await fs.writeFile(path.join(outputDir, fileName), new Uint8Array(await preview.arrayBuffer()));
  if (sheetName === "结果概览") {
    await fs.writeFile(path.join(figureDir, "manufacturing_ai_future_demo_dashboard.png"), new Uint8Array(await preview.arrayBuffer()));
  }
}
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(outputDir, "manufacturing_ai_future_demo.xlsx"));

const baseFindings = base2030.map(r => ({
  code: r.industry_code, name: r.industry_name_cn,
  adoption: r.any_ai_adoption_rate,
  serviceIndex: r.service_demand_index_2023_100,
  computeIndex: r.compute_resource_index_2023_100,
}));
console.log(JSON.stringify({ outputDir, baseFindings }, null, 2));

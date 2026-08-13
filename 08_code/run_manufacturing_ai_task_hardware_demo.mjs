import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const dataDir = path.join(root, "02_data");
const resultDir = path.join(root, "05_results");
const figureDir = path.join(resultDir, "figures");
const outputDir = path.join(root, "outputs", "manufacturing_ai_task_hardware_demo");

const years = [2023, 2026, 2027, 2030];
const efficiency = { 2023: 1.0, 2026: 1.5, 2027: 1.75, 2030: 3.0 };
const nationalAdoption = { 2023: 0.209347, 2026: 0.32, 2027: 0.360440, 2030: 0.480046 };
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

function odds(p) { return p / (1 - p); }
function invOdds(o) { return o / (1 + o); }
function adoptionPath(adoption2023, diffusionSpeed) {
  return Object.fromEntries(years.map(year => {
    if (year === 2023) return [year, adoption2023];
    const multiplier = odds(nationalAdoption[year]) / odds(nationalAdoption[2023]);
    return [year, invOdds(odds(adoption2023) * Math.pow(multiplier, diffusionSpeed))];
  }));
}

const industryInputs = [
  { code:"C13", name:"农副食品加工业", firms:25581, adoption2023:0.1858, diffusionSpeed:0.85, edge2030:4, mult:{office:0.75,agent:1.1,vision:1.2,maintenance:0.8,scheduling:1.1,simulation:0.4} },
  { code:"C14", name:"食品制造业", firms:10075, adoption2023:0.2037, diffusionSpeed:0.90, edge2030:3, mult:{office:0.8,agent:1.1,vision:1.0,maintenance:0.8,scheduling:1.1,simulation:0.5} },
  { code:"C17", name:"纺织业", firms:20858, adoption2023:0.1567, diffusionSpeed:0.80, edge2030:4, mult:{office:0.9,agent:1.0,vision:1.2,maintenance:0.7,scheduling:0.9,simulation:0.4} },
  { code:"C22", name:"造纸和纸制品业", firms:7859, adoption2023:0.1792, diffusionSpeed:0.95, edge2030:3, mult:{office:0.65,agent:0.8,vision:0.7,maintenance:1.4,scheduling:1.4,simulation:0.9} },
  { code:"C26", name:"化学原料和化学制品制造业", firms:25489, adoption2023:0.2296, diffusionSpeed:1.00, edge2030:2, mult:{office:0.7,agent:0.8,vision:0.5,maintenance:1.5,scheduling:1.3,simulation:1.5} },
  { code:"C27", name:"医药制造业", firms:9563, adoption2023:0.2587, diffusionSpeed:1.05, edge2030:5, mult:{office:1.0,agent:1.1,vision:1.2,maintenance:1.0,scheduling:1.2,simulation:1.6} },
  { code:"C31", name:"黑色金属冶炼和压延加工业", firms:6203, adoption2023:0.1640, diffusionSpeed:0.90, edge2030:3, mult:{office:0.55,agent:0.7,vision:0.6,maintenance:1.6,scheduling:1.4,simulation:1.1} },
  { code:"C34", name:"通用设备制造业", firms:34915, adoption2023:0.2384, diffusionSpeed:1.05, edge2030:6, mult:{office:0.9,agent:1.0,vision:1.4,maintenance:1.3,scheduling:1.1,simulation:1.4} },
  { code:"C36", name:"汽车制造业", firms:18899, adoption2023:0.2909, diffusionSpeed:1.10, edge2030:6, mult:{office:1.0,agent:1.0,vision:1.0,maintenance:1.0,scheduling:1.0,simulation:1.0} },
  { code:"C37", name:"铁路、船舶、航空航天和其他运输设备制造业", firms:6164, adoption2023:0.2146, diffusionSpeed:1.00, edge2030:6, mult:{office:0.9,agent:1.0,vision:1.3,maintenance:1.2,scheduling:1.1,simulation:1.8} },
  { code:"C38", name:"电气机械和器材制造业", firms:34242, adoption2023:0.2391, diffusionSpeed:1.05, edge2030:7, mult:{office:0.9,agent:1.0,vision:1.6,maintenance:1.1,scheduling:1.1,simulation:1.4} },
  { code:"C39", name:"计算机、通信和其他电子设备制造业", firms:27776, adoption2023:0.3278, diffusionSpeed:1.15, edge2030:8, mult:{office:0.9,agent:0.9,vision:2.0,maintenance:1.0,scheduling:1.0,simulation:1.2} },
  { code:"C40", name:"仪器仪表制造业", firms:7069, adoption2023:0.3319, diffusionSpeed:1.10, edge2030:6, mult:{office:1.0,agent:1.0,vision:1.5,maintenance:1.0,scheduling:1.0,simulation:1.6} },
];
const industries = industryInputs.map(industry => ({...industry, adoption: adoptionPath(industry.adoption2023, industry.diffusionSpeed)}));

const tasks = [
  { id:"office", name:"办公知识服务", hardware:"中央通用加速器", c36h2030:10, growth:4, note:"RAG、文档、知识检索和员工助手" },
  { id:"agent", name:"业务流程 Agent", hardware:"中央通用加速器", c36h2030:12, growth:8, note:"ERP/MES/QMS/采购销售等多步工作流" },
  { id:"vision", name:"机器视觉与异常复核", hardware:"边缘视觉设备+中央VLM", c36h2030:4, growth:6, note:"普通视觉在边缘；这里只将疑难复核计入GPU小时" },
  { id:"maintenance", name:"预测维护与设备诊断", hardware:"中央/边缘加速器等效", c36h2030:8, growth:8, note:"聚合窗口、异常解释、知识库和工单" },
  { id:"scheduling", name:"生产排程与优化", hardware:"中央加速器等效", c36h2030:7, growth:7, note:"排程重算、代理模型和工艺参数优化" },
  { id:"simulation", name:"研发仿真与数字孪生", hardware:"中央加速器等效", c36h2030:16, growth:10, note:"运行型数字孪生、设计验证和仿真作业" },
];

function csvEscape(value) {
  const s = String(value ?? "");
  return /[",\n]/.test(s) ? `"${s.replaceAll('"','""')}"` : s;
}
function toCsv(rows, headers) {
  return [headers.join(","), ...rows.map(r => headers.map(h => csvEscape(r[h])).join(","))].join("\n") + "\n";
}

const parameterRows = [];
for (const task of tasks) {
  parameterRows.push({record_type:"task",id:task.id,name_cn:task.name,year:2030,value:task.c36h2030,unit:"L20_equivalent_GPU_h/adopter/day",hardware_class:task.hardware,evidence_status:"prototype_anchor_plus_task_split_assumption",source:"05_results/manufacturing_ai_extended_load_screen.md",notes:`C36六任务合计57；${task.note}; 2023—2030单企业增长倍数=${task.growth}`});
  parameterRows.push({record_type:"task_growth",id:`${task.id}_growth_2023_2030`,name_cn:`${task.name}-单采用企业服务量增长`,year:2030,value:task.growth,unit:"times_2023",hardware_class:task.hardware,evidence_status:"scenario_assumption",source:"02_data/china_manufacturing_ai_future_scenarios.md",notes:"用于从2030锚点反推2023、2026和2027年每采用企业任务量。"});
}
for (const industry of industries) {
  parameterRows.push({record_type:"industry_stock",id:`${industry.code}_firms_2023`,name_cn:`${industry.name}-规上企业数`,year:2023,value:industry.firms,unit:"firms",hardware_class:"not_applicable",evidence_status:"observed_baseline",source:"02_data/china_manufacturing_ai_evidence_update.md",notes:"当前Demo固定使用2023年规上企业存量，未建模企业净进入。"});
  parameterRows.push({record_type:"industry_diffusion",id:`${industry.code}_diffusion_speed`,name_cn:`${industry.name}-扩散速度`,year:2030,value:industry.diffusionSpeed,unit:"odds_growth_exponent",hardware_class:"not_applicable",evidence_status:"scenario_assumption",source:"02_data/china_manufacturing_ai_future_scenarios.md",notes:"将全国中间情景的odds增长调整为行业路径。"});
  for (const year of years) parameterRows.push({record_type:"industry_adoption",id:`${industry.code}_adoption_${year}`,name_cn:`${industry.name}-AI采用率`,year,value:industry.adoption[year],unit:"fraction_of_above_size_firms",hardware_class:"not_applicable",evidence_status:year===2023?"observed_baseline":"conditional_middle_scenario",source:year===2023?"02_data/china_manufacturing_ai_evidence_update.md":"02_data/china_manufacturing_ai_future_scenarios.md",notes:"任意AI采用率，不等于六类任务各自的独立渗透率。"});
  for (const task of tasks) {
    parameterRows.push({record_type:"industry_task_multiplier",id:`${industry.code}_${task.id}`,name_cn:`${industry.name}-${task.name}`,year:2030,value:industry.mult[task.id],unit:"multiplier",hardware_class:task.hardware,evidence_status:"scenario_assumption",source:"02_data/china_manufacturing_ai_adoption_workload_report.md",notes:"表示相对于C36标准采用企业的任务适用性和深度，不是行业实测。"});
  }
  parameterRows.push({record_type:"edge_vision",id:`${industry.code}_edge`,name_cn:`${industry.name}-2030边缘视觉点位`,year:2030,value:industry.edge2030,unit:"edge_devices/adopter",hardware_class:"边缘视觉设备",evidence_status:"scenario_assumption",source:"05_results/manufacturing_ai_extended_load_screen.md",notes:"C36原型扩展为6个点位；其他行业按任务特征调整。"});
}
for (const year of years) {
  parameterRows.push({record_type:"hardware_efficiency",id:`eff_${year}`,name_cn:`${year}计算效率`,year,value:efficiency[year],unit:"times_2023",hardware_class:"全部中央加速器",evidence_status:year===2023?"normalisation":"scenario_assumption",source:"02_data/china_manufacturing_ai_future_scenarios.md",notes:"用于将L20等效服务小时转换为未来加速器小时。"});
}
for (const [id, value, unit, note] of [
  ["gpus_per_server",hardware.gpusPerServer,"GPUs/server","双加速器服务器"],
  ["target_utilisation",hardware.targetUtilisation,"fraction","池化容量目标利用率"],
  ["reserve_factor",hardware.reserveFactor,"multiplier","聚合容量备用系数"],
  ["server_idle_kw",hardware.serverIdleKw,"kW/server","继承原双L20服务器筛查"],
  ["server_full_kw",hardware.serverFullKw,"kW/server","继承原双L20服务器筛查"],
  ["pooled_pue",hardware.pooledPue,"ratio","区域/集团池化机房示范值"],
  ["edge_active_kw",hardware.edgeActiveKw,"kW/device","边缘视觉设备工作功率"],
  ["edge_idle_kw",hardware.edgeIdleKw,"kW/device","边缘视觉设备空闲功率"],
  ["edge_active_hours",hardware.edgeActiveHours,"hours/day","边缘视觉设备每日工作时数"],
  ["edge_aux_factor",hardware.edgeAuxFactor,"ratio","边缘设备辅助用电系数"],
]) parameterRows.push({record_type:"hardware",id,name_cn:note,year:"",value,unit,hardware_class:id.startsWith("edge_")?"边缘视觉设备":"中央服务器",evidence_status:"screening_assumption",source:id.startsWith("edge_")?"05_results/manufacturing_ai_extended_load_screen.md":"05_results/group_ai_center_findings.md",notes:"用于容量和设施电力下限筛查。"});

const taskRows = [];
const summaryRows = [];
for (const industry of industries) {
  for (const year of years) {
    const adopting = industry.firms * industry.adoption[year];
    const exponent = (2030 - year) / 7;
    for (const task of tasks) {
      const perAdopter = task.c36h2030 * industry.mult[task.id] / Math.pow(task.growth, exponent);
      const l20eq = adopting * perAdopter;
      const actual = l20eq / efficiency[year];
      taskRows.push({industry_code:industry.code,industry_name_cn:industry.name,year,task_id:task.id,task_name_cn:task.name,hardware_class:task.hardware,above_size_firms_2023:industry.firms,any_ai_adoption_rate:industry.adoption[year],adopting_firms:adopting,per_adopter_l20eq_gpu_h_day:perAdopter,sector_l20eq_gpu_h_day:l20eq,hardware_efficiency_vs_2023:efficiency[year],future_accelerator_gpu_h_day:actual,evidence_status:"conditional_screening_scenario"});
    }
    const group = taskRows.slice(taskRows.length - tasks.length);
    const l20eqTotal = group.reduce((s,r)=>s+r.sector_l20eq_gpu_h_day,0);
    const actualTotal = group.reduce((s,r)=>s+r.future_accelerator_gpu_h_day,0);
    const pooledServers = Math.ceil(actualTotal / (24 * hardware.gpusPerServer * hardware.targetUtilisation) * hardware.reserveFactor);
    const dynamicKwPerGpu = (hardware.serverFullKw - hardware.serverIdleKw) / hardware.gpusPerServer;
    const centralAverageKw = (pooledServers * hardware.serverIdleKw + actualTotal * dynamicKwPerGpu / 24) * hardware.pooledPue;
    const centralFullKw = pooledServers * hardware.serverFullKw * hardware.pooledPue;
    const stationPerAdopter = industry.edge2030 / Math.pow(tasks.find(t=>t.id==="vision").growth, exponent);
    const edgeDevices = adopting * stationPerAdopter;
    const edgeKwhDayPerDevice = (hardware.edgeActiveKw * hardware.edgeActiveHours + hardware.edgeIdleKw * (24-hardware.edgeActiveHours)) * hardware.edgeAuxFactor;
    const edgeAverageKw = edgeDevices * edgeKwhDayPerDevice / 24;
    const totalAverageKw = centralAverageKw + edgeAverageKw;
    summaryRows.push({industry_code:industry.code,industry_name_cn:industry.name,year,above_size_firms_2023:industry.firms,any_ai_adoption_rate:industry.adoption[year],adopting_firms:adopting,sector_l20eq_gpu_h_day:l20eqTotal,future_accelerator_gpu_h_day:actualTotal,pooled_2gpu_server_groups:pooledServers,central_average_facility_kw:centralAverageKw,central_installed_full_load_kw:centralFullKw,edge_devices:edgeDevices,edge_average_kw:edgeAverageKw,total_average_ai_facility_kw:totalAverageKw,annual_ai_electricity_gwh:totalAverageKw*8760/1e6,evidence_status:"pooled_capacity_lower_bound_scenario"});
  }
}

await fs.mkdir(outputDir,{recursive:true});
await fs.mkdir(figureDir,{recursive:true});
const paramHeaders = ["record_type","id","name_cn","year","value","unit","hardware_class","evidence_status","source","notes"];
const taskHeaders = Object.keys(taskRows[0]);
const summaryHeaders = Object.keys(summaryRows[0]);
await fs.writeFile(path.join(dataDir,"china_manufacturing_ai_task_hardware_demo_parameters.csv"),toCsv(parameterRows,paramHeaders),"utf8");
await fs.writeFile(path.join(resultDir,"manufacturing_ai_task_hardware_demo_task_results.csv"),toCsv(taskRows,taskHeaders),"utf8");
await fs.writeFile(path.join(resultDir,"manufacturing_ai_task_hardware_demo_sector_summary.csv"),toCsv(summaryRows,summaryHeaders),"utf8");

const wb=Workbook.create();
const readme=wb.worksheets.add("说明");
const params=wb.worksheets.add("参数");
const taskCalc=wb.worksheets.add("任务计算");
const sectorCalc=wb.worksheets.add("行业汇总");
const dash=wb.worksheets.add("结果概览");
for(const s of [readme,params,taskCalc,sectorCalc,dash]) s.showGridLines=false;

readme.getRange("A1:H1").merge();
readme.getRange("A1").values=[["十三行业六任务：GPU、服务器和电力负荷 Demo"]];
readme.getRange("A1:H1").format={fill:"#17365D",font:{bold:true,color:"#FFFFFF",size:16},rowHeight:30};
readme.getRange("A3:B12").values=[
  ["计算边界","先计算L20等效服务小时，再除以未来效率得到实际加速器小时。"],
  ["六类任务","办公知识、业务Agent、机器视觉、预测维护、生产排程、研发仿真。"],
  ["视觉边界","普通视觉使用边缘设备功率；只有复杂异常复核进入中央GPU小时。"],
  ["服务器数量","表示跨企业充分池化且按65%目标利用率、10%备用配置的容量下限。"],
  ["电力","中央服务器采用双加速器服务器功率包络和PUE=1.4；边缘视觉单列。"],
  ["企业边界","没有为每个企业强制配置整机，因此不能解释成本地部署采购量。"],
  ["年份","2023、2026、2027、2030中间情景。"],
  ["证据","十三行业规上企业数和2023采用率为观测；扩散、任务拆分、增长、效率与硬件参数为筛查假设。"],
  ["输出","任务级GPU小时、行业池化服务器组、平均设施功率、满载容量和年电量。"],
  ["更新时间","2026-08-09"],
];
readme.getRange("A3:A12").format={fill:"#D9EAF7",font:{bold:true,color:"#17365D"}};
readme.getRange("A3:A12").format.columnWidth=18; readme.getRange("B3:B12").format.columnWidth=82;
readme.getRange("B3:B12").format.wrapText=true; readme.getRange("A3:B12").format.rowHeight=32;

const pm=[paramHeaders,...parameterRows.map(r=>paramHeaders.map(h=>r[h]??""))];
params.getRangeByIndexes(0,0,pm.length,paramHeaders.length).values=pm;
params.getRangeByIndexes(0,0,1,paramHeaders.length).format={fill:"#17365D",font:{bold:true,color:"#FFFFFF"},wrapText:true};
params.freezePanes.freezeRows(1); params.getUsedRange().format.autofitColumns(); params.getRange("C:C").format.columnWidth=34; params.getRange("I:J").format.columnWidth=48; params.getRange("I:J").format.wrapText=true;
params.tables.add(`A1:J${pm.length}`,true,"TaskHardwareParameters");

const taskInputHeaders=taskHeaders;
const taskMatrix=taskRows.map(r=>taskInputHeaders.map(h=>["sector_l20eq_gpu_h_day","future_accelerator_gpu_h_day"].includes(h)?null:r[h]));
taskCalc.getRangeByIndexes(0,0,1,taskInputHeaders.length).values=[taskInputHeaders];
taskCalc.getRangeByIndexes(1,0,taskMatrix.length,taskInputHeaders.length).values=taskMatrix;
for(let i=0;i<taskRows.length;i++){
  const row=i+2;
  taskCalc.getRange(`K${row}`).formulas=[[`=I${row}*J${row}`]];
  taskCalc.getRange(`M${row}`).formulas=[[`=K${row}/L${row}`]];
}
taskCalc.getRangeByIndexes(0,0,1,taskInputHeaders.length).format={fill:"#17365D",font:{bold:true,color:"#FFFFFF"},wrapText:true};
taskCalc.freezePanes.freezeRows(1); taskCalc.getRange("H:H").format.numberFormat="0.0%"; taskCalc.getRange("I:I").format.numberFormat="#,##0"; taskCalc.getRange("J:M").format.numberFormat="#,##0.00"; taskCalc.getUsedRange().format.autofitColumns(); taskCalc.getRange("B:B").format.columnWidth=28; taskCalc.getRange("E:F").format.columnWidth=25;
taskCalc.tables.add(`A1:N${taskMatrix.length+1}`,true,"TaskHardwareResults");

const sh=summaryHeaders;
const sm=summaryRows.map(r=>sh.map(h=>["sector_l20eq_gpu_h_day","future_accelerator_gpu_h_day","pooled_2gpu_server_groups","central_average_facility_kw","central_installed_full_load_kw","edge_average_kw","total_average_ai_facility_kw","annual_ai_electricity_gwh"].includes(h)?null:r[h]));
sectorCalc.getRangeByIndexes(0,0,1,sh.length).values=[sh]; sectorCalc.getRangeByIndexes(1,0,sm.length,sh.length).values=sm;
for(let i=0;i<summaryRows.length;i++){
  const row=i+2, taskStart=i*6+2, taskEnd=taskStart+5;
  sectorCalc.getRange(`G${row}`).formulas=[[`=SUM('任务计算'!K${taskStart}:K${taskEnd})`]];
  sectorCalc.getRange(`H${row}`).formulas=[[`=SUM('任务计算'!M${taskStart}:M${taskEnd})`]];
  sectorCalc.getRange(`I${row}`).formulas=[[`=ROUNDUP(H${row}/(24*2*0.65)*1.1,0)`]];
  sectorCalc.getRange(`J${row}`).formulas=[[`=(I${row}*0.42+H${row}*0.44/24)*1.4`]];
  sectorCalc.getRange(`K${row}`).formulas=[[`=I${row}*1.3*1.4`]];
  sectorCalc.getRange(`M${row}`).formulas=[[`=L${row}*(0.04*16+0.01*8)*1.15/24`]];
  sectorCalc.getRange(`N${row}`).formulas=[[`=J${row}+M${row}`]];
  sectorCalc.getRange(`O${row}`).formulas=[[`=N${row}*8760/1000000`]];
}
sectorCalc.getRangeByIndexes(0,0,1,sh.length).format={fill:"#17365D",font:{bold:true,color:"#FFFFFF"},wrapText:true};
sectorCalc.freezePanes.freezeRows(1); sectorCalc.getRange("E:E").format.numberFormat="0.0%"; sectorCalc.getRange("F:I").format.numberFormat="#,##0"; sectorCalc.getRange("J:N").format.numberFormat="#,##0.0"; sectorCalc.getRange("O:O").format.numberFormat="#,##0.00"; sectorCalc.getUsedRange().format.autofitColumns(); sectorCalc.getRange("B:B").format.columnWidth=30;
sectorCalc.tables.add(`A1:P${sm.length+1}`,true,"SectorHardwareSummary");

const rows2030=summaryRows.filter(r=>r.year===2030);
const rows2030Ranked=[...rows2030].sort((a,b)=>b.total_average_ai_facility_kw-a.total_average_ai_facility_kw);
dash.getRange("A1:N1").merge(); dash.getRange("A1").values=[["2030年十三个制造行业 AI 计算与电力规模（中间情景、池化容量下限）"]]; dash.getRange("A1:N1").format={fill:"#17365D",font:{bold:true,color:"#FFFFFF",size:15},rowHeight:28};
for(const [range,label,fill] of [["A3:B3","覆盖行业","#D9EAF7"],["D3:E3","平均设施负荷","#E2F0D9"],["G3:H3","年用电","#FCE4D6"],["J3:K3","池化双GPU服务器组","#E4DFEC"]]){dash.getRange(range).merge();dash.getRange(range.split(":")[0]).values=[[label]];dash.getRange(range).format={fill,font:{bold:true,color:"#17365D"},horizontalAlignment:"center"};}
for(const range of ["A4:B4","D4:E4","G4:H4","J4:K4"]) dash.getRange(range).merge();
dash.getRange("A4").values=[[industries.length]]; dash.getRange("D4").formulas=[[`=SUM(F7:F${6+industries.length})`]]; dash.getRange("G4").formulas=[[`=SUM(G7:G${6+industries.length})`]]; dash.getRange("J4").formulas=[[`=SUM(E7:E${6+industries.length})`]];
dash.getRange("A4:K4").format={font:{bold:true,color:"#17365D",size:14},horizontalAlignment:"center"}; dash.getRange("D4").format.numberFormat="#,##0.0 \"MW\""; dash.getRange("G4").format.numberFormat="#,##0.0 \"GWh\""; dash.getRange("J4").format.numberFormat="#,##0";

dash.getRange("A6:G6").values=[["行业","采用企业","L20等效GPUh/日","未来GPUh/日","双GPU服务器组","平均设施MW","年用电GWh"]]; dash.getRange("A6:G6").format={fill:"#5B9BD5",font:{bold:true,color:"#FFFFFF"}};
dash.getRange(`A7:G${6+industries.length}`).values=rows2030Ranked.map(r=>[r.industry_name_cn,null,null,null,null,null,null]);
for(let i=0;i<rows2030Ranked.length;i++){
  const idx=summaryRows.indexOf(rows2030Ranked[i])+2, row=i+7;
  dash.getRange(`B${row}:G${row}`).formulas=[[
    `='行业汇总'!F${idx}`,`='行业汇总'!G${idx}`,`='行业汇总'!H${idx}`,`='行业汇总'!I${idx}`,`='行业汇总'!N${idx}/1000`,`='行业汇总'!O${idx}`
  ]];
}
dash.getRange(`A6:G${6+industries.length}`).format.borders={preset:"inside",style:"thin",color:"#D9E2F3"}; dash.getRange(`B7:E${6+industries.length}`).format.numberFormat="#,##0"; dash.getRange(`F7:G${6+industries.length}`).format.numberFormat="#,##0.0"; dash.getRange("A:A").format.columnWidth=40; dash.getRange("B:G").format.columnWidth=17;

dash.getRange("I6:J10").values=[["年份","十三行业平均负荷MW"],...years.map(y=>[y,null])];
for(let yi=0;yi<years.length;yi++){
  const sourceRows=summaryRows.map((r,i)=>r.year===years[yi]?i+2:null).filter(Boolean);
  dash.getRange(`J${7+yi}`).formulas=[[`=SUM(${sourceRows.map(row=>`'行业汇总'!N${row}`).join(",")})/1000`]];
}
dash.getRange("I6:J6").format={fill:"#E2F0D9",font:{bold:true}}; dash.getRange("J7:J10").format.numberFormat="#,##0.0";
dash.getRange(`I13:J${13+industries.length}`).values=[["行业","平均设施MW"],...rows2030Ranked.map(()=>[null,null])];
for(let i=0;i<rows2030Ranked.length;i++){const row=i+14,source=i+7;dash.getRange(`I${row}:J${row}`).formulas=[[`=A${source}`,`=F${source}`]];}
dash.getRange("I13:J13").format={fill:"#D9EAF7",font:{bold:true}};dash.getRange(`J14:J${13+industries.length}`).format.numberFormat="#,##0.0";dash.getRange("I:I").format.columnWidth=34;dash.getRange("J:J").format.columnWidth=18;
const c1=dash.charts.add("bar",dash.getRange(`I13:J${13+industries.length}`)); c1.title="2030年各行业AI设施平均负荷（MW）"; c1.hasLegend=false; c1.yAxis={numberFormatCode:"#,##0.0",min:0}; c1.setPosition("A22","G43");
const c2=dash.charts.add("line",dash.getRange("I6:J10")); c2.title="十三行业AI设施总平均负荷（MW）"; c2.hasLegend=false; c2.xAxis={axisType:"textAxis"}; c2.yAxis={numberFormatCode:"#,##0.0",min:0}; c2.setPosition("H22","N43");

dash.getRange("A45:N47").merge(); dash.getRange("A45").values=[["边界：服务器组是跨企业充分池化后的连续需求装箱结果，不包含每厂整机最小采购和N+1离散冗余；L20等效小时用于服务量核算，不表示所有任务实际运行在L20上。"]]; dash.getRange("A45:N47").format={fill:"#FFF2CC",font:{color:"#7F6000"},wrapText:true,rowHeight:24};

console.log((await wb.inspect({kind:"table",range:"结果概览!A1:N47",include:"values,formulas",tableMaxRows:47,tableMaxCols:14,maxChars:12000})).ndjson);
console.log((await wb.inspect({kind:"formula",sheetId:"行业汇总",range:`G1:O${summaryRows.length+1}`,maxChars:5000,options:{maxResults:120}})).ndjson);
console.log((await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",options:{useRegex:true,maxResults:100},summary:"formula errors"})).ndjson);

const renderJobs=[["说明","A1:H12","preview_readme.png"],["参数","A1:J24","preview_parameters.png"],["任务计算","A1:N24","preview_tasks.png"],["行业汇总",`A1:P${summaryRows.length+1}`,"preview_sectors.png"],["结果概览","A1:N47","preview_dashboard.png"]];
for(const [sheetName,range,file] of renderJobs){const b=await wb.render({sheetName,range,scale:1.5,format:"png"});await fs.writeFile(path.join(outputDir,file),new Uint8Array(await b.arrayBuffer()));if(sheetName==="结果概览")await fs.writeFile(path.join(figureDir,"manufacturing_ai_task_hardware_demo_dashboard.png"),new Uint8Array(await b.arrayBuffer()));}
const xlsx=await SpreadsheetFile.exportXlsx(wb); await xlsx.save(path.join(outputDir,"manufacturing_ai_task_hardware_demo.xlsx"));
console.log(JSON.stringify({summary2030:rows2030},null,2));

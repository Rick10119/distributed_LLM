import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "/Users/rick/Documents/research/distributed_LLM/outputs/019fef47_land_materials";
await fs.mkdir(outputDir, { recursive: true });

const wb = Workbook.create();
const summary = wb.worksheets.add("Summary");
const inputs = wb.worksheets.add("Inputs");
const results = wb.worksheets.add("Calculations");
const cross = wb.worksheets.add("Cross-check");
const evidence = wb.worksheets.add("Evidence");
const definitions = wb.worksheets.add("Definitions");

const navy = "#17324D";
const blue = "#2F75B5";
const paleBlue = "#D9EAF7";
const paleGreen = "#E2F0D9";
const paleYellow = "#FFF2CC";
const paleGray = "#F2F2F2";
const red = "#C00000";
const white = "#FFFFFF";

function title(sheet, range, text) {
  sheet.getRange(range).merge();
  sheet.getRange(range.split(":")[0]).values = [[text]];
  sheet.getRange(range).format = {
    fill: navy,
    font: { bold: true, color: white, size: 15 },
    verticalAlignment: "center",
  };
  sheet.getRange(range).format.rowHeight = 30;
}

function header(sheet, range) {
  sheet.getRange(range).format = {
    fill: blue,
    font: { bold: true, color: white },
    wrapText: true,
    verticalAlignment: "center",
  };
}

function section(sheet, range) {
  sheet.getRange(range).format = {
    fill: paleBlue,
    font: { bold: true, color: navy },
  };
}

title(inputs, "A1:I1", "Land, Building and Construction-Material Inputs");
inputs.getRange("A3:B6").values = [
  ["Global input", "Value"],
  ["Common installed IT capacity (MW-IT)", 1555.9755],
  ["Model version", "v0.6.0 / II_1host common-capacity comparison"],
  ["Boundary", "On-site building and land only; no power-generation land or materials"],
];
header(inputs, "A3:B3");
inputs.getRange("B4").format.numberFormat = "0.0000";

inputs.getRange("A8:I12").values = [
  ["Case ID", "Deployment case", "Capacity (MW-IT)", "GFA intensity (m2/MW-IT)", "Site intensity (m2/MW-IT)", "New-build fraction", "New-land-conversion fraction", "Status", "Interpretation"],
  ["SMALL_REUSE", "Small distributed AI in existing factories", 1555.9755, null, null, 0, 0, "boundary scenario", "New structural shell and new land are zero by scenario definition; occupied existing indoor area and retrofit materials remain NR."],
  ["CN_LARGE", "China large cloud / concentrated facility", 1555.9755, 1100, 313, 1, 1, "screening proxy", "GFA base is a Chinese project-sample midpoint; site intensity is a single China Unicom project proxy, not a national mean."],
  ["US_LARGE", "US large cloud / concentrated facility", 1555.9755, 764, 1175, 1, 1, "single-project proxy", "Microsoft SJDC04 multistory greenfield design proxy; not a US portfolio mean."],
  ["NOTE", "Alternative realization", null, null, null, null, null, "scenario rule", "Set new-land-conversion fraction to 0 for existing-campus expansion, while retaining new-build fraction at 1; set both to 0 for existing-capacity reuse."],
];
header(inputs, "A8:I8");
inputs.getRange("C9:G11").format.numberFormat = "0.000";
inputs.getRange("D9:G11").format.fill = paleYellow;

inputs.getRange("A15:G18").values = [
  ["Archetype ID", "Structural archetype", "Concrete (m3/m2 new GFA)", "Rebar (kg/m2 new GFA)", "Structural steel (kg/m2 new GFA)", "Evidence status", "Use"],
  ["STEEL_FRAME", "Three-storey steel-frame industrial building proxy", 0.13, 19.96, 93.67, "official cost-indicator project proxy", "Low-concrete / steel-frame sensitivity; not a data-centre average."],
  ["RC_FRAME", "Multistorey reinforced-concrete industrial building midpoint", 0.5125, 70, 0, "mean of two official project indicators", "RC-frame sensitivity. Mean of 0.472/0.553 m3 concrete and 74/66 kg rebar per m2."],
  ["REUSE", "Existing building shell reuse", 0, 0, 0, "boundary scenario", "Counts no new structural shell. Retrofit steel, pads, supports and MEP materials remain outside this first estimate."],
];
header(inputs, "A15:G15");
inputs.getRange("C16:E18").format.numberFormat = "0.000";
inputs.getRange("C16:E18").format.fill = paleYellow;

inputs.getRange("A21:E25").values = [
  ["Top-down parameter", "Low", "Base", "High", "Unit"],
  ["Hyperscale concrete", 275, 1500, 2500, "m3/MW"],
  ["Hyperscale construction steel", null, 275, null, "t/MW"],
  ["Reference facility capacity", 200, 200, 200, "MW"],
  ["Reference facility floor area", 60000, 93000, 120000, "m2"],
];
header(inputs, "A21:E21");
inputs.getRange("B22:D25").format.numberFormat = "#,##0.0";
inputs.getRange("B22:D25").format.fill = paleYellow;

title(results, "A1:N1", "Formula-driven Land and Material Calculations");
results.getRange("A3:N10").values = [
  ["Case ID", "Deployment case", "Archetype ID", "Capacity MW-IT", "Required GFA m2", "New GFA m2", "Total site area m2", "New land conversion m2", "Concrete m3", "Rebar t", "Structural steel t", "Total construction steel t", "Method", "Boundary note"],
  ["SMALL_REUSE", "Small distributed AI in existing factories", "REUSE", null, null, null, null, null, null, null, null, null, "GFA × material intensity", "Existing indoor area is NR; only new structural shell is counted here."],
  ["CN_LARGE", "China large cloud / concentrated facility", "STEEL_FRAME", null, null, null, null, null, null, null, null, null, "GFA × material intensity", "Greenfield screen; change new-land fraction in Inputs for campus expansion."],
  ["CN_LARGE", "China large cloud / concentrated facility", "RC_FRAME", null, null, null, null, null, null, null, null, null, "GFA × material intensity", "Same land result as steel-frame case; only structural material archetype changes."],
  ["US_LARGE", "US large cloud / concentrated facility", "STEEL_FRAME", null, null, null, null, null, null, null, null, null, "GFA × material intensity", "Greenfield screen using SJDC04 GFA and site proxies."],
  ["US_LARGE", "US large cloud / concentrated facility", "RC_FRAME", null, null, null, null, null, null, null, null, null, "GFA × material intensity", "Counterfactual common RC structure for cross-country comparability."],
  ["CN_CAMPUS", "China large cloud / existing-campus expansion", "RC_FRAME", null, null, null, null, null, null, null, null, null, "GFA × material intensity", "Illustrates new building without new land conversion."],
  ["US_EXISTING", "US cloud / existing capacity", "REUSE", null, null, null, null, null, null, null, null, null, "GFA × material intensity", "Existing capacity boundary: no new shell or land; physical occupied space is not zero."],
];
header(results, "A3:N3");

const rowSpecs = [
  { row: 4, inputRow: 9, archetypeRow: 18 },
  { row: 5, inputRow: 10, archetypeRow: 16 },
  { row: 6, inputRow: 10, archetypeRow: 17 },
  { row: 7, inputRow: 11, archetypeRow: 16 },
  { row: 8, inputRow: 11, archetypeRow: 17 },
  { row: 9, inputRow: 10, archetypeRow: 17, landOverride: 0 },
  { row: 10, inputRow: 11, archetypeRow: 18, buildOverride: 0, landOverride: 0 },
];

for (const spec of rowSpecs) {
  const r = spec.row;
  const input = spec.inputRow;
  const arch = spec.archetypeRow;
  results.getRange(`D${r}`).formulas = [[`=Inputs!C${input}`]];
  results.getRange(`E${r}`).formulas = [[`=IF(Inputs!D${input}="","NR",D${r}*Inputs!D${input})`]];
  const buildFraction = spec.buildOverride ?? `Inputs!F${input}`;
  results.getRange(`F${r}`).formulas = [[`=IF(${buildFraction}=0,0,D${r}*Inputs!D${input}*${buildFraction})`]];
  results.getRange(`G${r}`).formulas = [[`=IF(Inputs!E${input}="","NR",D${r}*Inputs!E${input})`]];
  const landFraction = spec.landOverride ?? `Inputs!G${input}`;
  results.getRange(`H${r}`).formulas = [[`=IF(${landFraction}=0,0,D${r}*Inputs!E${input}*${landFraction})`]];
  results.getRange(`I${r}`).formulas = [[`=F${r}*Inputs!C${arch}`]];
  results.getRange(`J${r}`).formulas = [[`=F${r}*Inputs!D${arch}/1000`]];
  results.getRange(`K${r}`).formulas = [[`=F${r}*Inputs!E${arch}/1000`]];
  results.getRange(`L${r}`).formulas = [[`=J${r}+K${r}`]];
}
results.getRange("D4:L10").format.numberFormat = "#,##0.0";

title(cross, "A1:H1", "Top-down Hyperscale Material Cross-check");
cross.getRange("A3:H6").values = [
  ["Metric", "Low", "Base", "High", "Unit", "Formula basis", "Use", "Do not combine with"],
  ["Concrete for common capacity", null, null, null, "m3", "Installed MW × m3/MW", "Facility/campus scale plausibility range", "GFA-based concrete totals"],
  ["Construction steel for common capacity", null, null, null, "t", "Installed MW × t/MW", "Order-of-magnitude check; steel split is unavailable", "GFA-based rebar + structural steel totals"],
  ["Reference-facility count equivalent", null, null, null, "200-MW equivalents", "Installed MW / 200 MW", "Interpretation only; not a real integer site count", "Actual project count"],
];
header(cross, "A3:H3");
cross.getRange("B4:D4").formulas = [["=Inputs!$B$4*Inputs!B22", "=Inputs!$B$4*Inputs!C22", "=Inputs!$B$4*Inputs!D22"]];
cross.getRange("B5:D5").formulas = [["=\"NR\"", "=Inputs!$B$4*Inputs!C23", "=\"NR\""]];
cross.getRange("B6:D6").formulas = [["=Inputs!$B$4/Inputs!B24", "=Inputs!$B$4/Inputs!C24", "=Inputs!$B$4/Inputs!D24"]];
cross.getRange("B4:D6").format.numberFormat = "#,##0.0";

title(evidence, "A1:H1", "Evidence Register");
evidence.getRange("A3:H10").values = [
  ["Evidence ID", "Geography", "Facility / archetype", "Parameter", "Value", "Unit", "Source", "Use limitation"],
  ["CN_GFA_BASE", "China", "Large data-centre project sample", "GFA intensity", 1100, "m2/MW-IT", "02_data/china_us_hyperscale_cloud_physical_parameters_report.md", "Scenario midpoint, not national mean"],
  ["CN_SITE_PROXY", "China", "China Unicom JJJ north", "Site intensity", 313, "m2/MW-IT", "02_data/china_us_hyperscale_cloud_physical_parameters_report.md", "Single project proxy"],
  ["US_SJDC04", "United States", "Microsoft SJDC04", "GFA / site intensity", "764 / 1175", "m2/MW-IT", "https://efiling.energy.ca.gov/GetDocument.aspx?DocumentContentId=101543&tn=264764", "Single multistorey greenfield design"],
  ["US_SJDC04_CONCRETE", "United States", "SJDC04 building construction", "Concrete deliveries", 6160, "yd3 for one building phase", "https://efiling.energy.ca.gov/GetDocument.aspx?DocumentContentId=80138&tn=245949", "Construction-model input; does not disclose rebar or structural-steel mass"],
  ["CN_STEEL_FACTORY", "China", "Three-storey steel-frame factory", "Concrete / rebar / structural steel", "0.13 / 19.96 / 93.67", "m3/m2; kg/m2; kg/m2", "https://zjj.baoji.gov.cn/col1025/col1078/202510/P020251015316477903739.pdf", "Industrial-building proxy, not data-centre sample"],
  ["CN_RC_FACTORY", "China", "Two multistorey RC industrial projects", "Concrete / rebar midpoint", "0.5125 / 70", "m3/m2; kg/m2", "https://jsj.zs.gov.cn/attachment/0/404/404975/1611411.pdf", "Mean of two project indicators; not data-centre sample"],
  ["ICEF_200MW", "Global", "200-MW reference data centre", "Concrete / construction steel", "55,000-500,000 m3; 275 t/MW steel", "facility total", "https://icef.go.jp/wp-content/themes/icef_new/pdf/roadmap/2025/CHAPTER%20%E2%85%A2%20%E2%80%93%203.%20EMBODIED%20GREENHOUSE%20GAS%20EMISSIONS%20%28SCOPE%203%29.pdf", "Broad order-of-magnitude cross-check; public steel data are scarce"],
];
header(evidence, "A3:H3");
evidence.getRange("E4:E10").format.numberFormat = "#,##0.0";

title(definitions, "A1:D1", "Definitions and Boundary Rules");
definitions.getRange("A3:D12").values = [
  ["Term", "Definition", "Reported separately from", "Rule"],
  ["Gross floor area (GFA)", "All building floor area used by IT, power, cooling, operations and internal circulation.", "Building footprint and site area", "Do not substitute white space or rack footprint."],
  ["Building footprint", "Ground projection of buildings.", "GFA", "Requires floor count or direct project evidence; not estimated in this MVP."],
  ["Site area", "Buildings plus outdoor cooling, substation, backup power, roads, fire separation and landscaping.", "New land conversion", "A brownfield or existing campus can have site occupancy without new conversion."],
  ["New land conversion", "Land newly converted from another use for the AI facility and supporting infrastructure.", "Total site area", "Equals site area only in an explicitly defined greenfield scenario."],
  ["New GFA", "Newly built or expanded floor area beyond reusable existing space.", "Total occupied existing space", "Only new GFA drives structural material quantities in this MVP."],
  ["Concrete", "Ready-mix or structural concrete volume.", "Cement mass", "Cement cannot be inferred without a mix design; this workbook does not estimate cement."],
  ["Rebar", "Reinforcing steel embedded in concrete.", "Structural steel", "Keep separate unless the source reports only total construction steel."],
  ["Structural steel", "Steel framing, beams, columns and related structural members.", "Server racks and equipment metal", "IT hardware and MEP equipment are outside this building-shell estimate."],
  ["Reuse baseline", "Existing factory room/building and existing industrial land are used.", "Zero physical occupancy", "Zero new shell and land is a scenario boundary, not proof that every factory can host the load."],
];
header(definitions, "A3:D3");

title(summary, "A1:H1", "Land and Construction-Material MVP — Decision Summary");
summary.getRange("A3:H3").values = [["Case", "New GFA (m2)", "Total site (m2)", "New land (m2)", "Concrete (m3)", "Rebar (t)", "Structural steel (t)", "Interpretation"]];
header(summary, "A3:H3");
summary.getRange("A4:H8").values = [
  ["Small distributed / existing factory reuse", null, null, null, null, null, null, "Baseline counts no new building shell or land; existing occupied area and retrofit materials are NR."],
  ["China large cloud / steel-frame proxy", null, null, null, null, null, null, "Greenfield site screen; project proxies, not a national average."],
  ["China large cloud / RC-frame proxy", null, null, null, null, null, null, "Same GFA and site; structural archetype sensitivity."],
  ["US large cloud / steel-frame proxy", null, null, null, null, null, null, "SJDC04 GFA/site proxy; steel-frame material proxy is Chinese industrial-building evidence."],
  ["US large cloud / RC-frame proxy", null, null, null, null, null, null, "Counterfactual common structure for comparable material intensity."],
];
const calcRows = [4, 5, 6, 7, 8];
for (let i = 0; i < calcRows.length; i++) {
  const sr = 4 + i;
  const cr = calcRows[i];
  summary.getRange(`B${sr}:G${sr}`).formulas = [[`=Calculations!F${cr}`, `=Calculations!G${cr}`, `=Calculations!H${cr}`, `=Calculations!I${cr}`, `=Calculations!J${cr}`, `=Calculations!K${cr}`]];
}
summary.getRange("B4:G8").format.numberFormat = "#,##0.0";
summary.getRange("A11:H15").values = [
  ["Key finding", "Value", "Unit", "Meaning", null, null, null, null],
  ["Common-capacity top-down concrete base", null, "m3", "ICEF facility/campus-scale check; do not add to GFA-based concrete.", null, null, null, null],
  ["Common-capacity top-down concrete range", null, "m3", "Wide structural/site uncertainty.", null, null, null, null],
  ["Common-capacity top-down construction steel", null, "t", "ICEF screening estimate; rebar/structural split unavailable.", null, null, null, null],
  ["Primary unresolved quantity", "Small-server occupied indoor area and retrofit materials", "NR", "Requires factory AI-room bills of quantities or rack/power-linked layouts.", null, null, null, null],
];
section(summary, "A11:H11");
summary.getRange("B12").formulas = [["='Cross-check'!C4"]];
summary.getRange("B13").formulas = [["=ROUND('Cross-check'!B4/1000000,3)&\"–\"&ROUND('Cross-check'!D4/1000000,3)&\" million\""]];
summary.getRange("B14").formulas = [["='Cross-check'!C5"]];
summary.getRange("B12:B14").format.numberFormat = "#,##0.0";
summary.getRange("A18:H21").values = [
  ["How to use", "1. Edit yellow cells in Inputs. 2. Use Calculations for GFA-based estimates. 3. Use Cross-check only as a separate plausibility range. 4. Never add the two methods.", null, null, null, null, null, null],
  ["Recommended baseline", "Small distributed: reuse scenario. Large cloud: report existing capacity, campus expansion and greenfield separately; do not collapse them into an assumed portfolio share.", null, null, null, null, null, null],
  ["Cement", "Not calculated. A cement mass requires concrete mix designs and supplementary-cementitious-material shares for each project.", null, null, null, null, null, null],
  ["Status", "Screening MVP; land proxies and structural archetypes are not national averages.", null, null, null, null, null, null],
];
summary.getRange("A18:A21").format = { fill: paleGreen, font: { bold: true, color: navy } };
summary.getRange("B18:H21").merge(true);

for (const sheet of [summary, inputs, results, cross, evidence, definitions]) {
  sheet.getUsedRange().format.verticalAlignment = "top";
  sheet.getUsedRange().format.wrapText = true;
  sheet.freezePanes.freezeRows(3);
}

summary.getRange("A1:H21").format.borders = { color: "#D9E1F2", style: "continuous", weight: 1 };
summary.getRange("A4:A8").format.font = { bold: true, color: navy };
summary.getRange("A21:H21").format.font = { italic: true, color: red };

summary.getRange("A1:H21").format.autofitColumns();
summary.getRange("A1:H21").format.autofitRows();
inputs.getRange("A1:I25").format.autofitColumns();
inputs.getRange("A1:I25").format.autofitRows();
results.getRange("A1:N10").format.autofitColumns();
results.getRange("A1:N10").format.autofitRows();
cross.getRange("A1:H6").format.autofitColumns();
cross.getRange("A1:H6").format.autofitRows();
evidence.getRange("A1:H10").format.autofitColumns();
evidence.getRange("A1:H10").format.autofitRows();
definitions.getRange("A1:D12").format.autofitColumns();
definitions.getRange("A1:D12").format.autofitRows();

summary.getRange("A:A").format.columnWidth = 34;
summary.getRange("B:B").format.columnWidth = 24;
summary.getRange("C:G").format.columnWidth = 16;
summary.getRange("H:H").format.columnWidth = 52;
summary.getRange("B4:G8").format.horizontalAlignment = "right";
summary.getRange("A3:H3").format.rowHeight = 34;
inputs.getRange("B:B").format.columnWidth = 42;
inputs.getRange("H:I").format.columnWidth = 34;
results.getRange("B:B").format.columnWidth = 34;
results.getRange("N:N").format.columnWidth = 44;
evidence.getRange("G:H").format.columnWidth = 48;
definitions.getRange("B:D").format.columnWidth = 48;

const inspect = await wb.inspect({ kind: "formula", sheetId: "Calculations", range: "D4:L10", maxChars: 12000, options: { maxResults: 100 } });
await fs.writeFile(`${outputDir}/formula_inspection.txt`, inspect.ndjson ?? String(inspect));

const formulaErrors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, maxChars: 6000 });
await fs.writeFile(`${outputDir}/formula_error_check.txt`, formulaErrors.ndjson ?? String(formulaErrors));

for (const sheetName of ["Summary", "Inputs", "Calculations", "Cross-check", "Evidence", "Definitions"]) {
  const preview = await wb.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${outputDir}/${sheetName}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(`${outputDir}/land_materials_mvp.xlsx`);

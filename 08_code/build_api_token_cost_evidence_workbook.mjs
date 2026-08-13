import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.env.API_TOKEN_PROJECT_ROOT || process.cwd();
const version = "api_token_cost_v1.1.0";
const accessDate = "2026-08-11";
const usdToCny = 6.7719;
const fxSource = "https://www.federalreserve.gov/releases/h10/current/";
const fxDate = "2026-07-24";

const headers = [
  "cost_case_version", "provider", "platform", "model_id", "model_version",
  "region", "currency", "observation_date", "valid_from", "valid_to",
  "service_tier", "input_per_mtoken", "cached_input_per_mtoken",
  "cache_write_per_mtoken", "output_per_mtoken", "batch_input_per_mtoken",
  "batch_cached_input_per_mtoken", "batch_output_per_mtoken",
  "long_context_threshold_tokens", "long_input_per_mtoken",
  "long_cached_input_per_mtoken", "long_cache_write_per_mtoken",
  "long_output_per_mtoken", "reasoning_token_billing", "batch_supported",
  "cache_rule", "active_for_baseline", "evidence_level", "model_status",
  "benchmark_role", "mainstream_representative",
  "source_url", "source_last_updated", "access_date", "limitations",
  "fx_to_cny", "fx_source_url", "fx_observation_date",
];

const common = {
  cost_case_version: version,
  observation_date: accessDate,
  access_date: accessDate,
  evidence_level: "A",
  model_status: "model_ready_price_only",
  fx_source_url: fxSource,
  fx_observation_date: fxDate,
};

const prices = [
  {
    ...common, provider: "Alibaba Cloud", platform: "Model Studio",
    model_id: "qwen3.7-max-2026-06-08", model_version: "2026-06-08",
    region: "China (Beijing)", currency: "CNY", valid_from: "2026-06-08",
    valid_to: "", service_tier: "standard_list_price", input_per_mtoken: 12,
    cached_input_per_mtoken: 1.2, cache_write_per_mtoken: 15,
    output_per_mtoken: 36, batch_input_per_mtoken: 6,
    batch_cached_input_per_mtoken: "", batch_output_per_mtoken: 18,
    long_context_threshold_tokens: "", long_input_per_mtoken: "",
    long_cached_input_per_mtoken: "", long_cache_write_per_mtoken: "",
    long_output_per_mtoken: "", reasoning_token_billing: "thinking tokens billed as output",
    batch_supported: true,
    cache_rule: "explicit cache hit 10%, write 125%; implicit hit 20%; batch and cache cannot combine",
    active_for_baseline: true,
    benchmark_role: "mainstream_representative", mainstream_representative: true,
    source_url: "https://help.aliyun.com/en/model-studio/model-pricing;https://help.aliyun.com/en/model-studio/context-cache;https://help.aliyun.com/en/model-studio/batch-inference",
    source_last_updated: "page current at access",
    limitations: "Use non-promotional list price. Snapshot is pay-as-you-go; qwen3.8-max-preview Token Plan is not substituted.",
    fx_to_cny: 1,
  },
  {
    ...common, provider: "DeepSeek", platform: "DeepSeek API",
    model_id: "deepseek-v4-flash", model_version: "DeepSeek-V4-Flash-0731",
    region: "Global API", currency: "USD", valid_from: "2026-07-31", valid_to: "",
    service_tier: "standard", input_per_mtoken: 0.14,
    cached_input_per_mtoken: 0.0028, cache_write_per_mtoken: "",
    output_per_mtoken: 0.28, batch_input_per_mtoken: "",
    batch_cached_input_per_mtoken: "", batch_output_per_mtoken: "",
    long_context_threshold_tokens: "", long_input_per_mtoken: "",
    long_cached_input_per_mtoken: "", long_cache_write_per_mtoken: "",
    long_output_per_mtoken: "", reasoning_token_billing: "thinking content included in output usage",
    batch_supported: false, cache_rule: "automatic disk cache; hit and miss priced separately",
    active_for_baseline: true,
    benchmark_role: "lower_bound_lightweight", mainstream_representative: false,
    source_url: "https://api-docs.deepseek.com/quick_start/pricing/",
    source_last_updated: "page current at access",
    limitations: "Provider warns that prices may change and a price increase is planned; version by observation date.",
    fx_to_cny: usdToCny,
  },
  {
    ...common, provider: "DeepSeek", platform: "DeepSeek API",
    model_id: "deepseek-v4-pro", model_version: "DeepSeek-V4-Pro",
    region: "Global API", currency: "USD", valid_from: "2026-07-31", valid_to: "",
    service_tier: "standard", input_per_mtoken: 0.435,
    cached_input_per_mtoken: 0.003625, cache_write_per_mtoken: "",
    output_per_mtoken: 0.87, batch_input_per_mtoken: "",
    batch_cached_input_per_mtoken: "", batch_output_per_mtoken: "",
    long_context_threshold_tokens: "", long_input_per_mtoken: "",
    long_cached_input_per_mtoken: "", long_cache_write_per_mtoken: "",
    long_output_per_mtoken: "", reasoning_token_billing: "thinking content included in output usage",
    batch_supported: false, cache_rule: "automatic disk cache; hit and miss priced separately",
    active_for_baseline: true,
    benchmark_role: "mainstream_representative", mainstream_representative: true,
    source_url: "https://api-docs.deepseek.com/quick_start/pricing/",
    source_last_updated: "page current at access",
    limitations: "Responses API availability differs by model. Provider warns that prices may change.",
    fx_to_cny: usdToCny,
  },
  ...[
    ["gpt-5.6-sol", 5, 0.5, 6.25, 30, 2.5, 0.25, 15, 10, 1, 12.5, 45, "upper_capability_sensitivity", false],
    ["gpt-5.6-terra", 2, 0.2, 2.5, 12, 1, 0.1, 6, 4, 0.4, 5, 18, "mainstream_representative", true],
    ["gpt-5.6-luna", 0.2, 0.02, 0.25, 1.2, 0.1, 0.01, 0.6, 0.4, 0.04, 0.5, 1.8, "lower_bound_lightweight", false],
  ].map(([model, input, cached, write, output, bInput, bCached, bOutput, lInput, lCached, lWrite, lOutput, benchmarkRole, representative]) => ({
    ...common, provider: "OpenAI", platform: "OpenAI API", model_id: model,
    model_version: "current catalog entry", region: "Global default routing", currency: "USD",
    valid_from: "2026-08-11", valid_to: "", service_tier: "standard_short_context",
    input_per_mtoken: input, cached_input_per_mtoken: cached,
    cache_write_per_mtoken: write, output_per_mtoken: output,
    batch_input_per_mtoken: bInput, batch_cached_input_per_mtoken: bCached,
    batch_output_per_mtoken: bOutput, long_context_threshold_tokens: 272000,
    long_input_per_mtoken: lInput, long_cached_input_per_mtoken: lCached,
    long_cache_write_per_mtoken: lWrite, long_output_per_mtoken: lOutput,
    reasoning_token_billing: "reasoning tokens billed as output tokens",
    batch_supported: true, cache_rule: "cache write 125% of uncached input; cached input priced separately",
    active_for_baseline: true,
    benchmark_role: benchmarkRole, mainstream_representative: representative,
    source_url: "https://developers.openai.com/api/docs/pricing",
    source_last_updated: "page current at access",
    limitations: "Regional processing can add 10%; tool calls and adjacent services are excluded from the base token bill.",
    fx_to_cny: usdToCny,
  })),
  {
    ...common, provider: "Anthropic", platform: "Claude API", model_id: "claude-sonnet-5",
    model_version: "launch pricing", region: "Global default routing", currency: "USD",
    valid_from: "2026-08-01", valid_to: "2026-08-31", service_tier: "standard",
    input_per_mtoken: 2, cached_input_per_mtoken: 0.2, cache_write_per_mtoken: 2.5,
    output_per_mtoken: 10, batch_input_per_mtoken: 1,
    batch_cached_input_per_mtoken: 0.1, batch_output_per_mtoken: 5,
    long_context_threshold_tokens: "", long_input_per_mtoken: "",
    long_cached_input_per_mtoken: "", long_cache_write_per_mtoken: "",
    long_output_per_mtoken: "", reasoning_token_billing: "all output usage billed at output rate",
    batch_supported: true,
    cache_rule: "5m write 125%, 1h write 200%, hit 10%; modifiers can stack with batch",
    active_for_baseline: true,
    benchmark_role: "mainstream_representative", mainstream_representative: true,
    source_url: "https://platform.claude.com/docs/en/about-claude/pricing",
    source_last_updated: "page current at access",
    limitations: "Introductory price expires 2026-08-31; US-only inference adds 10%. cache_write field stores the 5-minute write price.",
    fx_to_cny: usdToCny,
  },
  {
    ...common, provider: "Anthropic", platform: "Claude API", model_id: "claude-sonnet-5",
    model_version: "standard pricing", region: "Global default routing", currency: "USD",
    valid_from: "2026-09-01", valid_to: "", service_tier: "standard",
    input_per_mtoken: 3, cached_input_per_mtoken: 0.3, cache_write_per_mtoken: 3.75,
    output_per_mtoken: 15, batch_input_per_mtoken: 1.5,
    batch_cached_input_per_mtoken: 0.15, batch_output_per_mtoken: 7.5,
    long_context_threshold_tokens: "", long_input_per_mtoken: "",
    long_cached_input_per_mtoken: "", long_cache_write_per_mtoken: "",
    long_output_per_mtoken: "", reasoning_token_billing: "all output usage billed at output rate",
    batch_supported: true,
    cache_rule: "5m write 125%, 1h write 200%, hit 10%; modifiers can stack with batch",
    active_for_baseline: false,
    benchmark_role: "dated_future_price", mainstream_representative: false,
    source_url: "https://platform.claude.com/docs/en/about-claude/pricing",
    source_last_updated: "page current at access",
    limitations: "Future effective price retained as a dated sensitivity row; not used for 2026-08-11 baseline.",
    fx_to_cny: usdToCny,
  },
  ...[
    ["gemini-3.5-flash", 1.5, 0.15, 9, 0.75, 0.075, 4.5, "mainstream_representative", true],
    ["gemini-3.1-flash-lite", 0.25, 0.025, 1.5, 0.125, 0.0125, 0.75, "lower_bound_lightweight", false],
  ].map(([model, input, cached, output, bInput, bCached, bOutput, benchmarkRole, representative]) => ({
    ...common, provider: "Google", platform: "Gemini Developer API", model_id: model,
    model_version: "current catalog entry", region: "Paid tier", currency: "USD",
    valid_from: "2026-08-05", valid_to: "", service_tier: "standard",
    input_per_mtoken: input, cached_input_per_mtoken: cached,
    cache_write_per_mtoken: "", output_per_mtoken: output,
    batch_input_per_mtoken: bInput, batch_cached_input_per_mtoken: bCached,
    batch_output_per_mtoken: bOutput, long_context_threshold_tokens: "",
    long_input_per_mtoken: "", long_cached_input_per_mtoken: "",
    long_cache_write_per_mtoken: "", long_output_per_mtoken: "",
    reasoning_token_billing: "thinking tokens included in output token charge",
    batch_supported: true, cache_rule: "cached input plus separate cache storage charge",
    active_for_baseline: true,
    benchmark_role: benchmarkRole, mainstream_representative: representative,
    source_url: "https://ai.google.dev/gemini-api/docs/pricing",
    source_last_updated: "2026-08-05",
    limitations: "Cache storage, grounding, tools and enterprise provisioned throughput are excluded from base token bill.",
    fx_to_cny: usdToCny,
  })),
];

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows) {
  return [headers.join(","), ...rows.map((row) => headers.map((key) => csvEscape(row[key] ?? "")).join(","))].join("\n") + "\n";
}

const rawDir = path.join(root, "02_data", "raw", "curated");
const processedDir = path.join(root, "02_data", "processed", "api_token_cost");
await fs.mkdir(rawDir, { recursive: true });
await fs.mkdir(processedDir, { recursive: true });
await fs.writeFile(path.join(rawDir, "api_token_price_evidence_v1_1.csv"), toCsv(prices), "utf8");
await fs.writeFile(
  path.join(processedDir, "api_token_prices_v1_1.csv"),
  toCsv(prices.filter((row) => row.evidence_level === "A" && row.model_status === "model_ready_price_only")),
  "utf8",
);

const workbook = Workbook.create();
const priceSheet = workbook.worksheets.add("API Prices");
priceSheet.showGridLines = false;
priceSheet.getRangeByIndexes(0, 0, prices.length + 1, headers.length).values = [
  headers,
  ...prices.map((row) => headers.map((key) => row[key] ?? "")),
];
priceSheet.freezePanes.freezeRows(1);
priceSheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
  fill: "#17365D", font: { bold: true, color: "#FFFFFF" }, wrapText: true,
};
priceSheet.getUsedRange().format.autofitColumns();
priceSheet.getUsedRange().format.autofitRows();
priceSheet.getRange("A:AJ").format.columnWidth = 18;
priceSheet.getRange("AD:AD").format.columnWidth = 52;
priceSheet.getRange("AG:AG").format.columnWidth = 52;

const noteSheet = workbook.worksheets.add("Audit Notes");
noteSheet.showGridLines = false;
const notes = [
  ["Item", "Audit disposition"],
  ["Scope", "Official list prices only. Mainstream representatives are compared with IF, IG, II and the existing GPU-IaaS cases. Quality equivalence, workload token volume, adjacent services and enterprise discounts remain separate."],
  ["Representative rule", "One balanced or general-purpose current model per mainstream provider: Qwen Max, DeepSeek Pro, GPT-5.6 Terra, Claude Sonnet 5 and Gemini 3.5 Flash. Lightweight/Flash/Lite models remain lower-bound sensitivities."],
  ["Qwen", "The research report's qwen3.8-max pay-as-you-go row was not reproduced. Use the dated qwen3.7-max snapshot; qwen3.8-max-preview belongs to Token Plan documentation."],
  ["DeepSeek", "V4 Flash and V4 Pro prices reproduced on the official pricing page; provider warns that prices will change."],
  ["OpenAI", "Current official pricing page used. Terra and Luna prices differ from earlier search snippets; the opened pricing page is authoritative for this archive."],
  ["Anthropic", "Launch and post-launch Sonnet 5 prices are stored as separate dated rows."],
  ["Google", "The report's gemini-3.6-flash row was not reproduced. Current catalog entries gemini-3.5-flash and gemini-3.1-flash-lite are used."],
  ["Excluded", "Doubao, Azure provisioned throughput and Bedrock provisioned throughput are not in the active table because directly comparable public prices were not confirmed in this pass."],
  ["FX", `USD rows use ${usdToCny} CNY/USD from the Federal Reserve H.10 observation on ${fxDate}; update independently from token prices.`],
];
noteSheet.getRangeByIndexes(0, 0, notes.length, 2).values = notes;
noteSheet.getRange("A1:B1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" } };
noteSheet.getRange(`A1:B${notes.length}`).format.wrapText = true;
noteSheet.getRange("A:A").format.columnWidth = 20;
noteSheet.getRange("B:B").format.columnWidth = 95;
noteSheet.getUsedRange().format.autofitRows();

const totalComparisonPath = path.join(root, "05_results", "v0.6.0", "result", "api_token_cost", "mainstream_total_cost_comparison.csv");
try {
  const comparisonCsv = await fs.readFile(totalComparisonPath, "utf8");
  const rawComparisonRows = comparisonCsv.trim().split(/\r?\n/).map((line, rowIndex) =>
    line.split(",").map((value, columnIndex) => {
      const clean = rowIndex === 0 && columnIndex === 0 ? value.replace(/^\uFEFF/, "") : value;
      return rowIndex > 0 && [1, 5, 6, 7, 8, 9, 10].includes(columnIndex) ? Number(clean) : clean;
    })
  );
  const comparisonRows = [
    ["方案", "年总成本（十亿元）", "方案组", "厂商", "模型/计费方式", "相对IF差额（十亿元）", "相对IF", "成本口径", "结果状态"],
    ...rawComparisonRows.slice(1).map((row) => [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[11], row[12]]),
  ];
  const resultSheet = workbook.worksheets.add("Total Comparison");
  resultSheet.getRangeByIndexes(0, 0, comparisonRows.length, comparisonRows[0].length).values = comparisonRows;
  resultSheet.showGridLines = false;
  resultSheet.freezePanes.freezeRows(1);
  const used = resultSheet.getUsedRange();
  used.format.autofitColumns();
  used.format.autofitRows();
  resultSheet.getRange("A1:I1").format = {
    fill: "#17365D", font: { bold: true, color: "#FFFFFF" }, wrapText: true,
  };
  resultSheet.getRange("A:A").format.columnWidth = 42;
  resultSheet.getRange("B:B").format.columnWidth = 20;
  resultSheet.getRange("C:E").format.columnWidth = 24;
  resultSheet.getRange("F:F").format.columnWidth = 22;
  resultSheet.getRange("B2:B11").format.numberFormat = "#,##0.000";
  resultSheet.getRange("F2:F11").format.numberFormat = "+#,##0.000;-#,##0.000;0.000";
  resultSheet.getRange("G2:G11").format.numberFormat = "+0.0%;-0.0%;0.0%";
  resultSheet.getRange("H:I").format.columnWidth = 44;
  resultSheet.getRange("H1:I11").format = { wrapText: true, horizontalAlignment: "left" };
} catch (error) {
  if (error?.code !== "ENOENT") throw error;
}

const workbookOutput = await SpreadsheetFile.exportXlsx(workbook);
await workbookOutput.save(path.join(processedDir, `${version}.xlsx`));

const previewDir = process.env.API_TOKEN_PREVIEW_DIR || "/private/tmp/api_token_cost_preview";
await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of workbook.worksheets.items.map((sheet) => sheet.name)) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  const safeName = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(path.join(previewDir, `${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const inspection = await workbook.inspect({
  kind: "table", range: "'API Prices'!A1:AL12", include: "values,formulas",
  tableMaxRows: 12, tableMaxCols: 38, maxChars: 14000,
});
console.log(inspection.ndjson);
const resultInspection = await workbook.inspect({
  kind: "table", range: "'Total Comparison'!A1:I11", include: "values,formulas",
  tableMaxRows: 11, tableMaxCols: 9, maxChars: 12000,
});
console.log(resultInspection.ndjson);
const formulaErrors = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan",
});
console.log(formulaErrors.ndjson);

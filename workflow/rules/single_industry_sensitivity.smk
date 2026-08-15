"""One-at-a-time sensitivity workflow for the industry selected in its registry."""

import yaml

SINGLE_INDUSTRY_OAT_REGISTRY = "config/sensitivity/single_industry_oat_v1.yaml"
SINGLE_INDUSTRY_OAT = yaml.safe_load(open(SINGLE_INDUSTRY_OAT_REGISTRY, encoding="utf-8"))
SINGLE_INDUSTRY_OAT_ROOT = SINGLE_INDUSTRY_OAT["output_root"]
SINGLE_INDUSTRY_CODE = SINGLE_INDUSTRY_OAT["industry"]
SINGLE_INDUSTRY_ARCHITECTURES = SINGLE_INDUSTRY_OAT["architectures"]
SINGLE_INDUSTRY_CASES = [
    factor_id + "__" + case_name
    for factor_id, factor in SINGLE_INDUSTRY_OAT["factors"].items()
    for case_name in factor["cases"]
]
SINGLE_INDUSTRY_REFERENCE = SINGLE_INDUSTRY_OAT["reference_case"]["case_id"]
SINGLE_INDUSTRY_RUN_CASES = [SINGLE_INDUSTRY_REFERENCE] + SINGLE_INDUSTRY_CASES
SINGLE_INDUSTRY_CONFIGS = expand(SINGLE_INDUSTRY_OAT_ROOT + "/configs/{case_id}.yaml", case_id=SINGLE_INDUSTRY_RUN_CASES)
SINGLE_INDUSTRY_SUMMARIES = expand(
    SINGLE_INDUSTRY_OAT_ROOT + "/model/{case_id}/" + SINGLE_INDUSTRY_CODE + "/{scenario}/summary.csv",
    case_id=SINGLE_INDUSTRY_RUN_CASES,
    scenario=SINGLE_INDUSTRY_ARCHITECTURES,
)
SINGLE_INDUSTRY_VALIDATIONS = expand(
    SINGLE_INDUSTRY_OAT_ROOT + "/result/{case_id}/" + SINGLE_INDUSTRY_CODE + "/{scenario}/validated.done.json",
    case_id=SINGLE_INDUSTRY_RUN_CASES,
    scenario=SINGLE_INDUSTRY_ARCHITECTURES,
)
SINGLE_INDUSTRY_CASE_RESULTS = SINGLE_INDUSTRY_OAT_ROOT + "/result/case_results.csv"
SINGLE_INDUSTRY_FACTOR_RESULTS = SINGLE_INDUSTRY_OAT_ROOT + "/result/factor_results.csv"
SINGLE_INDUSTRY_DONE = SINGLE_INDUSTRY_OAT_ROOT + "/result/validated.done.json"
SINGLE_INDUSTRY_PROFILE = config["paths"].get(
    "hourly_industry_profiles",
    "02_data/processed/core/manufacturing_31sector_real_weeks.csv",
)
SINGLE_INDUSTRY_COMMON_INPUTS = [
    path for path in COMMON_INPUTS
    if path != config["paths"]["hourly_industry_profiles"]
] + [SINGLE_INDUSTRY_PROFILE]
GRID_HYBRID_REGISTRY = "config/sensitivity/single_industry_grid_hybrid_v1.yaml"
GRID_HYBRID = yaml.safe_load(open(GRID_HYBRID_REGISTRY, encoding="utf-8"))
GRID_HYBRID_ROOT = GRID_HYBRID["output_root"]
GRID_HYBRID_INDUSTRY = GRID_HYBRID["industry"]
GRID_HYBRID_ARCHITECTURES = GRID_HYBRID["architectures"]
GRID_HYBRID_CASES = [
    factor_id + "__" + case_name
    for factor_id, factor in GRID_HYBRID["factors"].items()
    for case_name in factor["cases"]
]
GRID_HYBRID_SUMMARIES = expand(
    GRID_HYBRID_ROOT + "/model/{case_id}/" + GRID_HYBRID_INDUSTRY + "/{scenario}/summary.csv",
    case_id=GRID_HYBRID_CASES,
    scenario=GRID_HYBRID_ARCHITECTURES,
)
GRID_HYBRID_VALIDATIONS = expand(
    GRID_HYBRID_ROOT + "/result/{case_id}/" + GRID_HYBRID_INDUSTRY + "/{scenario}/validated.done.json",
    case_id=GRID_HYBRID_CASES,
    scenario=GRID_HYBRID_ARCHITECTURES,
)
GRID_HYBRID_COMPARISON = GRID_HYBRID_ROOT + "/result/comparison.csv"
GRID_HYBRID_FINDINGS = GRID_HYBRID_ROOT + "/result/findings.md"
GRID_HYBRID_DONE = GRID_HYBRID_ROOT + "/result/validated.done.json"
GROUP_MULTISITE_REGISTRY = "config/sensitivity/c36_group_multisite_continuous_v1.yaml"
GROUP_MULTISITE = yaml.safe_load(open(GROUP_MULTISITE_REGISTRY, encoding="utf-8"))
GROUP_MULTISITE_ROOT = GROUP_MULTISITE["output_root"]
GROUP_MULTISITE_SUMMARY = GROUP_MULTISITE_ROOT + "/summary.csv"
GROUP_MULTISITE_HOURLY = GROUP_MULTISITE_ROOT + "/hourly.csv"
GROUP_MULTISITE_LINEAGE = GROUP_MULTISITE_ROOT + "/curve_lineage.csv"
GROUP_MULTISITE_ALIGNMENT = GROUP_MULTISITE_ROOT + "/load_alignment_value.csv"
GROUP_MULTISITE_METADATA = GROUP_MULTISITE_ROOT + "/metadata.json"
C33_TYPICAL_DAY_REGISTRY = "config/sensitivity/c33_typical_day_horizon_v1.yaml"
C33_TYPICAL_DAY = yaml.safe_load(open(C33_TYPICAL_DAY_REGISTRY, encoding="utf-8"))
C33_TYPICAL_DAY_ROOT = C33_TYPICAL_DAY["output_root"]
C33_MEASURED_WEEK_REGISTRY = "config/sensitivity/c33_measured_week_horizon_v1.yaml"
C33_MEASURED_WEEK = yaml.safe_load(open(C33_MEASURED_WEEK_REGISTRY, encoding="utf-8"))
C33_MEASURED_WEEK_ROOT = C33_MEASURED_WEEK["output_root"]
C33_MEASURED_WEEK_OUTPUTS = [
    C33_MEASURED_WEEK_ROOT + "/summary.csv",
    C33_MEASURED_WEEK_ROOT + "/hourly.csv",
    C33_MEASURED_WEEK_ROOT + "/curve_lineage.csv",
    C33_MEASURED_WEEK_ROOT + "/load_alignment_value.csv",
    C33_MEASURED_WEEK_ROOT + "/metadata.json",
]
C33_TYPICAL_DAY_OUTPUTS = [
    C33_TYPICAL_DAY_ROOT + "/summary.csv",
    C33_TYPICAL_DAY_ROOT + "/hourly.csv",
    C33_TYPICAL_DAY_ROOT + "/curve_lineage.csv",
    C33_TYPICAL_DAY_ROOT + "/load_alignment_value.csv",
    C33_TYPICAL_DAY_ROOT + "/metadata.json",
]
C33_HORIZON_COMPARISON = C33_TYPICAL_DAY_ROOT + "/horizon_comparison.csv"
C33_HORIZON_FINDINGS = C33_TYPICAL_DAY_ROOT + "/findings.md"
C33_HORIZON_DONE = C33_TYPICAL_DAY_ROOT + "/validated.done.json"
NATIONAL_CLOUD_CONFIG = "config/sensitivity/national_cloud_center_v1.yaml"
NATIONAL_CLOUD_ROOT = "05_results/sensitivity/v0.8.0/national_cloud_center_v1"
NATIONAL_CLOUD_SUMMARY = NATIONAL_CLOUD_ROOT + "/summary.csv"
NATIONAL_CLOUD_HOURLY = NATIONAL_CLOUD_ROOT + "/hourly.csv"
NATIONAL_CLOUD_RESOLVED = NATIONAL_CLOUD_ROOT + "/resolved_config.yaml"
NATIONAL_GRID_COMPARISON = "05_results/sensitivity/v0.8.0/national_grid_capacity_comparison.csv"
NATIONAL_GRID_COMPARISON_DONE = "05_results/sensitivity/v0.8.0/national_grid_capacity_comparison.validated.done.json"
NATIONAL_OAT_REGISTRY = "config/sensitivity/national_oat_extension_v1.yaml"
NATIONAL_OAT = yaml.safe_load(open(NATIONAL_OAT_REGISTRY, encoding="utf-8"))
NATIONAL_OAT_ROOT = NATIONAL_OAT["output_root"]
NATIONAL_OAT_INDUSTRIES = NATIONAL_OAT["selected_industries"]
NATIONAL_OAT_ARCHITECTURES = NATIONAL_OAT.get("solve_architectures", NATIONAL_OAT["architectures"])
NATIONAL_OAT_CASES = [
    factor_id + "__" + case_name
    for factor_id, factor in NATIONAL_OAT["factors"].items()
    for case_name in factor["cases"]
]
NATIONAL_OAT_NO_SHIFT_CASES = ["PHY03__no_shift"]
NATIONAL_OAT_HIGH_CASES = ["PHY01__low", "PHY01__high", "PHY02__efficient"]
NATIONAL_OAT_CONFIG = NATIONAL_OAT_ROOT + "/configs/{case_id}.yaml"
NATIONAL_OAT_INDUSTRY_ROOT = NATIONAL_OAT_ROOT + "/model/{case_id}/{industry}"
NATIONAL_OAT_INDUSTRY_SUMMARY = NATIONAL_OAT_INDUSTRY_ROOT + "/summary.csv"
NATIONAL_OAT_INDUSTRY_HOURLY = NATIONAL_OAT_INDUSTRY_ROOT + "/hourly.csv"
NATIONAL_OAT_INDUSTRY_LINEAGE = NATIONAL_OAT_INDUSTRY_ROOT + "/curve_lineage.csv"
NATIONAL_OAT_INDUSTRY_ALIGNMENT = NATIONAL_OAT_INDUSTRY_ROOT + "/load_alignment_value.csv"
NATIONAL_OAT_INDUSTRY_METADATA = NATIONAL_OAT_INDUSTRY_ROOT + "/metadata.json"
NATIONAL_OAT_SUMMARY = NATIONAL_OAT_ROOT + "/result/{case_id}/core_scenarios.csv"
NATIONAL_OAT_ALIGNMENT = NATIONAL_OAT_ROOT + "/result/{case_id}/ig_1host_load_alignment.csv"
NATIONAL_OAT_LINEAGE = NATIONAL_OAT_ROOT + "/result/{case_id}/curve_lineage.csv"
NATIONAL_OAT_NATIONAL_DONE = NATIONAL_OAT_ROOT + "/result/{case_id}/validated.done.json"
NATIONAL_OAT_CLOUD = NATIONAL_OAT_ROOT + "/cloud/{case_id}/summary.csv"
NATIONAL_OAT_COMPARISON = NATIONAL_OAT_ROOT + "/result/{case_id}/grid_capacity_comparison.csv"
NATIONAL_OAT_COMPARISON_DONE = NATIONAL_OAT_ROOT + "/result/{case_id}/grid_capacity_comparison.validated.done.json"


rule single_industry_sensitivity:
    input:
        SINGLE_INDUSTRY_CASE_RESULTS,
        SINGLE_INDUSTRY_FACTOR_RESULTS,
        SINGLE_INDUSTRY_DONE,


rule single_industry_grid_hybrid_sensitivity:
    input:
        GRID_HYBRID_COMPARISON,
        GRID_HYBRID_FINDINGS,
        GRID_HYBRID_DONE,


rule single_industry_group_multisite_sensitivity:
    input:
        GROUP_MULTISITE_SUMMARY,
        GROUP_MULTISITE_HOURLY,
        GROUP_MULTISITE_LINEAGE,
        GROUP_MULTISITE_ALIGNMENT,
        GROUP_MULTISITE_METADATA,


rule c33_typical_day_horizon_test:
    input:
        C33_MEASURED_WEEK_OUTPUTS,
        C33_TYPICAL_DAY_OUTPUTS,
        C33_HORIZON_COMPARISON,
        C33_HORIZON_FINDINGS,
        C33_HORIZON_DONE,


rule run_c33_typical_day_horizon_test:
    input:
        registry=C33_TYPICAL_DAY_REGISTRY,
        defaults="config/defaults.yaml",
        run_config="config/runs/c33_typical_day_group_test.yaml",
        archive=config["paths"]["eweld_archive"],
        script="08_code/run_group_multisite_continuous_test.py",
    output:
        summary=C33_TYPICAL_DAY_ROOT + "/summary.csv",
        hourly=C33_TYPICAL_DAY_ROOT + "/hourly.csv",
        lineage=C33_TYPICAL_DAY_ROOT + "/curve_lineage.csv",
        alignment=C33_TYPICAL_DAY_ROOT + "/load_alignment_value.csv",
        metadata=C33_TYPICAL_DAY_ROOT + "/metadata.json",
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults {input.defaults} --config {input.run_config} "
        "--experiment {input.registry} --output-dir " + C33_TYPICAL_DAY_ROOT


rule run_c33_measured_week_horizon_reference:
    input:
        registry=C33_MEASURED_WEEK_REGISTRY,
        defaults="config/defaults.yaml",
        run_config="config/runs/all_industries_core.yaml",
        archive=config["paths"]["eweld_archive"],
        script="08_code/run_group_multisite_continuous_test.py",
    output:
        summary=C33_MEASURED_WEEK_ROOT + "/summary.csv",
        hourly=C33_MEASURED_WEEK_ROOT + "/hourly.csv",
        lineage=C33_MEASURED_WEEK_ROOT + "/curve_lineage.csv",
        alignment=C33_MEASURED_WEEK_ROOT + "/load_alignment_value.csv",
        metadata=C33_MEASURED_WEEK_ROOT + "/metadata.json",
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults {input.defaults} --config {input.run_config} "
        "--experiment {input.registry} --output-dir " + C33_MEASURED_WEEK_ROOT


rule summarize_c33_horizon_comparison:
    input:
        week=C33_MEASURED_WEEK_ROOT + "/summary.csv",
        week_metadata=C33_MEASURED_WEEK_ROOT + "/metadata.json",
        day=C33_TYPICAL_DAY_ROOT + "/summary.csv",
        script="08_code/summarize_c33_horizon_comparison.py",
    output:
        comparison=C33_HORIZON_COMPARISON,
        findings=C33_HORIZON_FINDINGS,
        done=C33_HORIZON_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --week {input.week} --week-metadata {input.week_metadata} "
        "--day {input.day} --output {output.comparison} "
        "--findings {output.findings} --done {output.done}"


rule run_single_industry_group_multisite_sensitivity:
    input:
        registry=GROUP_MULTISITE_REGISTRY,
        defaults="config/defaults.yaml",
        run_config="config/runs/all_industries_core.yaml",
        archive="02_data/raw_load_profiles/eweld/EWELD.zip",
        routing="config/compute_hardware/cpu_gpu_routing_v1.yaml",
        groups=config["paths"]["representative_group_report"],
        script="08_code/run_group_multisite_continuous_test.py",
    output:
        summary=GROUP_MULTISITE_SUMMARY,
        hourly=GROUP_MULTISITE_HOURLY,
        lineage=GROUP_MULTISITE_LINEAGE,
        alignment=GROUP_MULTISITE_ALIGNMENT,
        metadata=GROUP_MULTISITE_METADATA,
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults {input.defaults} --config {input.run_config} "
        "--experiment {input.registry} --output-dir " + GROUP_MULTISITE_ROOT


rule national_cloud_center:
    input:
        common=SINGLE_INDUSTRY_COMMON_INPUTS,
        cloud_config=NATIONAL_CLOUD_CONFIG,
        script="08_code/run_national_cloud_center.py",
    output:
        summary=NATIONAL_CLOUD_SUMMARY,
        hourly=NATIONAL_CLOUD_HOURLY,
        resolved=NATIONAL_CLOUD_RESOLVED,
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults config/defaults.yaml --run-config {input.cloud_config} "
        "--cloud-config {input.cloud_config} --summary-output {output.summary} "
        "--hourly-output {output.hourly} --resolved-config-output {output.resolved}"


rule national_grid_capacity_comparison:
    input:
        national=GROUP_CORE_NATIONAL_SUMMARY,
        national_validation=GROUP_CORE_NATIONAL_DONE,
        cloud=NATIONAL_CLOUD_SUMMARY,
        script="08_code/summarize_national_grid_capacity_comparison.py",
    output:
        table=NATIONAL_GRID_COMPARISON,
        done=NATIONAL_GRID_COMPARISON_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --national-summary {input.national} --cloud-summary {input.cloud} "
        "--output {output.table} --done-output {output.done}"


rule national_no_shift_sensitivity:
    input:
        expand(NATIONAL_OAT_COMPARISON_DONE, case_id=NATIONAL_OAT_NO_SHIFT_CASES),


rule national_high_impact_sensitivity:
    input:
        expand(NATIONAL_OAT_COMPARISON_DONE, case_id=NATIONAL_OAT_HIGH_CASES),


rule materialize_national_oat_case:
    input:
        registry=NATIONAL_OAT_REGISTRY,
        script="08_code/materialize_sensitivity_case.py",
    output:
        NATIONAL_OAT_CONFIG,
    wildcard_constraints:
        case_id="|".join(NATIONAL_OAT_CASES),
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --registry {input.registry} --case-id {wildcards.case_id} --output {output}"


rule run_national_oat_industry:
    input:
        common=COMMON_INPUTS,
        case_config=NATIONAL_OAT_CONFIG,
        experiment=GROUP_CORE_REGISTRY,
        archive=config["paths"]["eweld_archive"],
        script="08_code/run_group_multisite_continuous_test.py",
    output:
        summary=NATIONAL_OAT_INDUSTRY_SUMMARY,
        hourly=NATIONAL_OAT_INDUSTRY_HOURLY,
        lineage=NATIONAL_OAT_INDUSTRY_LINEAGE,
        alignment=NATIONAL_OAT_INDUSTRY_ALIGNMENT,
        metadata=NATIONAL_OAT_INDUSTRY_METADATA,
    wildcard_constraints:
        case_id="|".join(NATIONAL_OAT_CASES),
        industry="|".join(NATIONAL_OAT_INDUSTRIES),
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults config/defaults.yaml --config {input.case_config} "
        "--experiment {input.experiment} --industry {wildcards.industry} "
        "--architectures " + " ".join(NATIONAL_OAT_ARCHITECTURES) + " "
        "--output-dir " + NATIONAL_OAT_ROOT + "/model/{wildcards.case_id}/{wildcards.industry}"


rule combine_national_oat_case:
    input:
        summaries=lambda wildcards: expand(
            NATIONAL_OAT_INDUSTRY_SUMMARY,
            case_id=[wildcards.case_id],
            industry=NATIONAL_OAT_INDUSTRIES,
        ),
        alignments=lambda wildcards: expand(
            NATIONAL_OAT_INDUSTRY_ALIGNMENT,
            case_id=[wildcards.case_id],
            industry=NATIONAL_OAT_INDUSTRIES,
        ),
        lineages=lambda wildcards: expand(
            NATIONAL_OAT_INDUSTRY_LINEAGE,
            case_id=[wildcards.case_id],
            industry=NATIONAL_OAT_INDUSTRIES,
        ),
        metadata=lambda wildcards: expand(
            NATIONAL_OAT_INDUSTRY_METADATA,
            case_id=[wildcards.case_id],
            industry=NATIONAL_OAT_INDUSTRIES,
        ),
        script="08_code/summarize_group_multisite_core.py",
    output:
        summary=NATIONAL_OAT_SUMMARY,
        alignment=NATIONAL_OAT_ALIGNMENT,
        lineage=NATIONAL_OAT_LINEAGE,
        done=NATIONAL_OAT_NATIONAL_DONE,
    wildcard_constraints:
        case_id="|".join(NATIONAL_OAT_CASES),
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --root " + NATIONAL_OAT_ROOT + "/model/{wildcards.case_id} "
        "--industries " + " ".join(NATIONAL_OAT_INDUSTRIES) + " "
        "--architectures " + " ".join(NATIONAL_OAT_ARCHITECTURES) + " "
        "--summary-output {output.summary} --alignment-output {output.alignment} "
        "--lineage-output {output.lineage} --done-output {output.done}"


rule run_national_oat_cloud_case:
    input:
        common=SINGLE_INDUSTRY_COMMON_INPUTS,
        case_config=NATIONAL_OAT_CONFIG,
        cloud_config=NATIONAL_CLOUD_CONFIG,
        script="08_code/run_national_cloud_center.py",
    output:
        summary=NATIONAL_OAT_CLOUD,
        hourly=NATIONAL_OAT_ROOT + "/cloud/{case_id}/hourly.csv",
        resolved=NATIONAL_OAT_ROOT + "/cloud/{case_id}/resolved_config.yaml",
    wildcard_constraints:
        case_id="|".join(NATIONAL_OAT_CASES),
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults config/defaults.yaml --run-config {input.case_config} "
        "--cloud-config {input.cloud_config} --summary-output {output.summary} "
        "--hourly-output {output.hourly} --resolved-config-output {output.resolved}"


rule compare_national_oat_grid_capacity:
    input:
        national=NATIONAL_OAT_SUMMARY,
        national_validation=NATIONAL_OAT_NATIONAL_DONE,
        cloud=NATIONAL_OAT_CLOUD,
        script="08_code/summarize_national_grid_capacity_comparison.py",
    output:
        table=NATIONAL_OAT_COMPARISON,
        done=NATIONAL_OAT_COMPARISON_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --national-summary {input.national} --cloud-summary {input.cloud} "
        "--output {output.table} --done-output {output.done}"


rule materialize_single_industry_oat_case:
    input:
        registry=SINGLE_INDUSTRY_OAT_REGISTRY,
        script="08_code/materialize_sensitivity_case.py",
    output:
        SINGLE_INDUSTRY_OAT_ROOT + "/configs/{case_id}.yaml",
    wildcard_constraints:
        case_id="|".join(SINGLE_INDUSTRY_RUN_CASES),
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --registry {input.registry} --case-id {wildcards.case_id} --output {output}"


rule run_single_industry_oat_case:
    input:
        common=SINGLE_INDUSTRY_COMMON_INPUTS,
        case_config=SINGLE_INDUSTRY_OAT_ROOT + "/configs/{case_id}.yaml",
        script="08_code/run_core_scenario.py",
    params:
        baseline=MODEL_OUTPUT_ROOT + "/" + SINGLE_INDUSTRY_CODE + "/baseline/summary.json",
    output:
        summary=SINGLE_INDUSTRY_OAT_ROOT + "/model/{case_id}/" + SINGLE_INDUSTRY_CODE + "/{scenario}/summary.csv",
        hourly=SINGLE_INDUSTRY_OAT_ROOT + "/model/{case_id}/" + SINGLE_INDUSTRY_CODE + "/{scenario}/hourly.csv",
        resolved=SINGLE_INDUSTRY_OAT_ROOT + "/model/{case_id}/" + SINGLE_INDUSTRY_CODE + "/{scenario}/resolved_config.yaml",
    wildcard_constraints:
        case_id="|".join(SINGLE_INDUSTRY_RUN_CASES),
        scenario="|".join(SINGLE_INDUSTRY_ARCHITECTURES),
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults config/defaults.yaml --config {input.case_config} "
        "--industry " + SINGLE_INDUSTRY_CODE + " --scenario {wildcards.scenario} "
        "--baseline-summary {params.baseline} --summary-output {output.summary} "
        "--hourly-output {output.hourly} --resolved-config-output {output.resolved}"


rule validate_single_industry_oat_case:
    input:
        config=SINGLE_INDUSTRY_OAT_ROOT + "/configs/{case_id}.yaml",
        summary=SINGLE_INDUSTRY_OAT_ROOT + "/model/{case_id}/" + SINGLE_INDUSTRY_CODE + "/{scenario}/summary.csv",
        hourly=SINGLE_INDUSTRY_OAT_ROOT + "/model/{case_id}/" + SINGLE_INDUSTRY_CODE + "/{scenario}/hourly.csv",
        script="08_code/validate_core_scenario.py",
    params:
        baseline=MODEL_OUTPUT_ROOT + "/" + SINGLE_INDUSTRY_CODE + "/baseline/summary.json",
    output:
        SINGLE_INDUSTRY_OAT_ROOT + "/result/{case_id}/" + SINGLE_INDUSTRY_CODE + "/{scenario}/validated.done.json",
    wildcard_constraints:
        case_id="|".join(SINGLE_INDUSTRY_RUN_CASES),
        scenario="|".join(SINGLE_INDUSTRY_ARCHITECTURES),
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --defaults config/defaults.yaml --config {input.config} "
        "--industry " + SINGLE_INDUSTRY_CODE + " --scenario {wildcards.scenario} "
        "--summary {input.summary} --hourly {input.hourly} --baseline-summary {params.baseline} --output {output}"


rule summarize_single_industry_oat:
    input:
        registry=SINGLE_INDUSTRY_OAT_REGISTRY,
        configs=expand(SINGLE_INDUSTRY_OAT_ROOT + "/configs/{case_id}.yaml", case_id=SINGLE_INDUSTRY_CASES),
        summaries=expand(
            SINGLE_INDUSTRY_OAT_ROOT + "/model/{case_id}/" + SINGLE_INDUSTRY_CODE + "/{scenario}/summary.csv",
            case_id=SINGLE_INDUSTRY_CASES,
            scenario=SINGLE_INDUSTRY_ARCHITECTURES,
        ),
        reference=expand(
            SINGLE_INDUSTRY_OAT_ROOT + "/model/" + SINGLE_INDUSTRY_REFERENCE + "/" + SINGLE_INDUSTRY_CODE + "/{scenario}/summary.csv",
            scenario=SINGLE_INDUSTRY_ARCHITECTURES,
        ),
        validations=SINGLE_INDUSTRY_VALIDATIONS,
        script="08_code/summarize_single_industry_oat.py",
    output:
        cases=SINGLE_INDUSTRY_CASE_RESULTS,
        factors=SINGLE_INDUSTRY_FACTOR_RESULTS,
        done=SINGLE_INDUSTRY_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --registry {input.registry} --reference-summaries {input.reference} "
        "--case-summaries {input.summaries} --case-configs {input.configs} "
        "--case-output {output.cases} --factor-output {output.factors} --done-output {output.done}"


rule materialize_grid_hybrid_case:
    input:
        registry=GRID_HYBRID_REGISTRY,
        script="08_code/materialize_sensitivity_case.py",
    output:
        GRID_HYBRID_ROOT + "/configs/{case_id}.yaml",
    wildcard_constraints:
        case_id="|".join(GRID_HYBRID_CASES),
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --registry {input.registry} --case-id {wildcards.case_id} --output {output}"


rule run_grid_hybrid_baseline:
    input:
        common=SINGLE_INDUSTRY_COMMON_INPUTS,
        case_config=GRID_HYBRID_ROOT + "/configs/{case_id}.yaml",
        script="08_code/run_core_baseline.py",
    output:
        summary=GRID_HYBRID_ROOT + "/model/{case_id}/" + GRID_HYBRID_INDUSTRY + "/baseline/summary.json",
        hourly=GRID_HYBRID_ROOT + "/model/{case_id}/" + GRID_HYBRID_INDUSTRY + "/baseline/hourly.csv",
        resolved=GRID_HYBRID_ROOT + "/model/{case_id}/" + GRID_HYBRID_INDUSTRY + "/baseline/resolved_config.yaml",
    wildcard_constraints:
        case_id="|".join(GRID_HYBRID_CASES),
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults config/defaults.yaml --config {input.case_config} "
        "--industry " + GRID_HYBRID_INDUSTRY + " --summary-output {output.summary} "
        "--hourly-output {output.hourly} --resolved-config-output {output.resolved}"


rule run_grid_hybrid_case:
    input:
        common=SINGLE_INDUSTRY_COMMON_INPUTS,
        case_config=GRID_HYBRID_ROOT + "/configs/{case_id}.yaml",
        baseline=GRID_HYBRID_ROOT + "/model/{case_id}/" + GRID_HYBRID_INDUSTRY + "/baseline/summary.json",
        script="08_code/run_core_scenario.py",
    output:
        summary=GRID_HYBRID_ROOT + "/model/{case_id}/" + GRID_HYBRID_INDUSTRY + "/{scenario}/summary.csv",
        hourly=GRID_HYBRID_ROOT + "/model/{case_id}/" + GRID_HYBRID_INDUSTRY + "/{scenario}/hourly.csv",
        resolved=GRID_HYBRID_ROOT + "/model/{case_id}/" + GRID_HYBRID_INDUSTRY + "/{scenario}/resolved_config.yaml",
    wildcard_constraints:
        case_id="|".join(GRID_HYBRID_CASES),
        scenario="|".join(GRID_HYBRID_ARCHITECTURES),
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults config/defaults.yaml --config {input.case_config} "
        "--industry " + GRID_HYBRID_INDUSTRY + " --scenario {wildcards.scenario} "
        "--baseline-summary {input.baseline} --summary-output {output.summary} "
        "--hourly-output {output.hourly} --resolved-config-output {output.resolved}"


rule validate_grid_hybrid_case:
    input:
        config=GRID_HYBRID_ROOT + "/configs/{case_id}.yaml",
        summary=GRID_HYBRID_ROOT + "/model/{case_id}/" + GRID_HYBRID_INDUSTRY + "/{scenario}/summary.csv",
        hourly=GRID_HYBRID_ROOT + "/model/{case_id}/" + GRID_HYBRID_INDUSTRY + "/{scenario}/hourly.csv",
        baseline=GRID_HYBRID_ROOT + "/model/{case_id}/" + GRID_HYBRID_INDUSTRY + "/baseline/summary.json",
        script="08_code/validate_core_scenario.py",
    output:
        GRID_HYBRID_ROOT + "/result/{case_id}/" + GRID_HYBRID_INDUSTRY + "/{scenario}/validated.done.json",
    wildcard_constraints:
        case_id="|".join(GRID_HYBRID_CASES),
        scenario="|".join(GRID_HYBRID_ARCHITECTURES),
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --defaults config/defaults.yaml --config {input.config} "
        "--industry " + GRID_HYBRID_INDUSTRY + " --scenario {wildcards.scenario} "
        "--summary {input.summary} --hourly {input.hourly} --baseline-summary {input.baseline} --output {output}"


rule summarize_grid_hybrid_sensitivity:
    input:
        registry=GRID_HYBRID_REGISTRY,
        summaries=GRID_HYBRID_SUMMARIES,
        validations=GRID_HYBRID_VALIDATIONS,
        script="08_code/summarize_single_industry_grid_hybrid.py",
    output:
        comparison=GRID_HYBRID_COMPARISON,
        findings=GRID_HYBRID_FINDINGS,
        done=GRID_HYBRID_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --registry {input.registry} --inputs {input.summaries} "
        "--output {output.comparison} --findings-output {output.findings} --done-output {output.done}"

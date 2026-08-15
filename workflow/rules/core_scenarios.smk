"""Reproducible core-scenario workflow."""

import yaml

INDUSTRIES = config["selected_industries"]
SCENARIOS = config["selected_scenarios"]
RUN_CONFIG = config["run_config_path"]
MODEL_VERSION = config["model_version"]
SCENARIO_REGISTRY_PATH = config.get("scenario_registry_path", "config/scenarios/mainline.yaml")
SCENARIO_REGISTRY = yaml.safe_load(open(SCENARIO_REGISTRY_PATH, encoding="utf-8"))
COUNTRY_CASES = SCENARIO_REGISTRY["countries"]
ENABLED_COUNTRIES = COUNTRY_CASES["enabled"]
if set(ENABLED_COUNTRIES) != {"china", "us"}:
    raise ValueError("The mainline result package currently requires countries.enabled = [china, us]")
HARDWARE_CASE = SCENARIO_REGISTRY["compute_hardware"]
RESOURCE_CASE = SCENARIO_REGISTRY["resource_footprint"]
DEPLOYMENT_CONFIG = "config/deployment_core.yaml"
DEPLOYMENT = yaml.safe_load(open(DEPLOYMENT_CONFIG, encoding="utf-8"))
CORE_ARCHITECTURE = DEPLOYMENT["core_architecture"]
if CORE_ARCHITECTURE not in SCENARIOS:
    raise ValueError("deployment.core_architecture must be one of selected_scenarios")
VERSION_OUTPUT_ROOT = config["paths"]["results_root"] + "/" + MODEL_VERSION
MODEL_OUTPUT_ROOT = VERSION_OUTPUT_ROOT + "/model"
RESULT_OUTPUT_ROOT = VERSION_OUTPUT_ROOT + "/result"
GROUP_CORE_REGISTRY = "config/scenarios/group_multisite_core_v1.yaml"
GROUP_CORE_ROOT = RESULT_OUTPUT_ROOT + "/group_architecture_core"
GROUP_CORE_SUMMARIES = expand(GROUP_CORE_ROOT + "/{industry}/summary.csv", industry=INDUSTRIES)
GROUP_CORE_HOURLY = expand(GROUP_CORE_ROOT + "/{industry}/hourly.csv", industry=INDUSTRIES)
GROUP_CORE_LINEAGES = expand(GROUP_CORE_ROOT + "/{industry}/curve_lineage.csv", industry=INDUSTRIES)
GROUP_CORE_ALIGNMENTS = expand(GROUP_CORE_ROOT + "/{industry}/load_alignment_value.csv", industry=INDUSTRIES)
GROUP_CORE_METADATA = expand(GROUP_CORE_ROOT + "/{industry}/metadata.json", industry=INDUSTRIES)
GROUP_CORE_NATIONAL_ROOT = GROUP_CORE_ROOT + "/national"
GROUP_CORE_NATIONAL_SUMMARY = GROUP_CORE_NATIONAL_ROOT + "/core_scenarios.csv"
GROUP_CORE_NATIONAL_ALIGNMENT = GROUP_CORE_NATIONAL_ROOT + "/ig_1host_load_alignment.csv"
GROUP_CORE_NATIONAL_LINEAGE = GROUP_CORE_NATIONAL_ROOT + "/curve_lineage.csv"
GROUP_CORE_NATIONAL_DONE = GROUP_CORE_NATIONAL_ROOT + "/validated.done.json"
NATIONAL_CLOUD_CONFIG = "config/sensitivity/national_cloud_center_v1.yaml"
NATIONAL_CLOUD_ROOT = "05_results/sensitivity/v0.8.0/national_cloud_center_v1"
NATIONAL_CLOUD_SUMMARY = NATIONAL_CLOUD_ROOT + "/summary.csv"
GROUP_CORE_TARGETS = (
    [
        GROUP_CORE_NATIONAL_SUMMARY,
        GROUP_CORE_NATIONAL_ALIGNMENT,
        GROUP_CORE_NATIONAL_LINEAGE,
        GROUP_CORE_NATIONAL_DONE,
    ]
    if len(INDUSTRIES) == 31
    else []
)

CORE_SOURCES = [
    "08_code/core/capacity.py",
    "08_code/core/config.py",
    "08_code/core/data.py",
    "08_code/core/io.py",
    "08_code/core/model.py",
    "08_code/core/representative_group.py",
]

RAW_SERVICE_INPUTS = [
    config["paths"]["raw_effective_service_growth"],
    config["paths"]["raw_effective_service_template_fallbacks"],
]

UPSTREAM_SERVICE_INPUTS = [
    config["paths"]["upstream_31sector_bottomup"],
    config["paths"]["upstream_task_templates"],
]

MODEL_READY_SERVICE = config["paths"]["model_ready_task_service"]
MODEL_READY_SUMMARY = config["paths"]["model_ready_service_summary"]
MODEL_READY_LINEAGE = config["paths"]["model_ready_service_lineage"]
MODEL_READY_VALIDATION = config["paths"]["model_ready_service_validation"]
MODEL_READY_FINDINGS = config["paths"]["model_ready_service_findings"]
MODEL_READY_DONE = config["paths"]["model_ready_service_done"]
LEGACY_HOURLY_PROFILES = config["paths"]["legacy_hourly_industry_profiles"]
DAILY_HOURLY_PROFILES = config["paths"]["daily_hourly_industry_profiles"]
MODEL_READY_HOURLY_PROFILES = config["paths"]["hourly_industry_profiles"]
DAILY_HOURLY_LINEAGE = "02_data/processed/core/manufacturing_31sector_base_load_profiles.lineage.json"
MODEL_READY_HOURLY_LINEAGE = config["paths"]["hourly_industry_profiles_lineage"]
RAW_US_MECS = config["paths"]["raw_us_manufacturing_mecs"]
ROOFTOP_CROSSWALK = config["paths"]["industry_rooftop_crosswalk"]
MODEL_READY_ROOFTOP = config["paths"]["model_ready_industry_rooftop"]
MODEL_READY_ROOFTOP_LINEAGE = config["paths"]["model_ready_industry_rooftop_lineage"]
RAW_COMPUTE_EFFICIENCY = config["paths"]["raw_compute_efficiency_micro_scenarios"]
MODEL_READY_COMPUTE_EFFICIENCY = config["paths"]["model_ready_compute_efficiency"]
MODEL_READY_COMPUTE_EFFICIENCY_LINEAGE = config["paths"]["model_ready_compute_efficiency_lineage"]
RAW_MODEL_LIFECYCLE_PARAMETERS = config["paths"]["raw_model_lifecycle_parameters"]
RAW_MODEL_HARDWARE_MATRIX = config["paths"]["raw_model_hardware_matrix"]
MODEL_READY_MODEL_LIFECYCLE = config["paths"]["model_ready_model_lifecycle"]
ENTERPRISE_AI_COST_PARAMETERS = config["paths"]["enterprise_ai_cost_parameters"]
API_TOKEN_PRICES = config["paths"]["api_token_prices"]
LAND_SPACE_PARAMETERS = RESOURCE_CASE["space"]["parameter_file"]
LAND_MATERIAL_PARAMETERS = RESOURCE_CASE["materials"]["parameter_file"]
LAND_MATERIAL_CROSSCHECK_PARAMETERS = RESOURCE_CASE["materials"]["crosscheck_file"]
WATER_PARAMETERS = RESOURCE_CASE["water"]["parameter_file"]

COMMON_INPUTS = [
    "config/defaults.yaml",
    RUN_CONFIG,
    config["paths"]["representative_group_report"],
    config["paths"]["hourly_industry_profiles"],
    config["paths"]["hourly_industry_profiles_lineage"],
    MODEL_READY_SERVICE,
    MODEL_READY_DONE,
    MODEL_READY_COMPUTE_EFFICIENCY,
    MODEL_READY_COMPUTE_EFFICIENCY_LINEAGE,
    MODEL_READY_MODEL_LIFECYCLE,
    config["paths"]["flexibility_mapping"],
    config["paths"]["topdown_allocation"],
    config["paths"]["workload_shape_source"],
    config["paths"]["pv_profile_source"],
    config["paths"]["spot_price_source"],
    config["paths"]["battery_cost_source"],
    MODEL_READY_ROOFTOP,
    MODEL_READY_ROOFTOP_LINEAGE,
    HARDWARE_CASE["routing_config"],
] + CORE_SOURCES

VALIDATED = expand(
    RESULT_OUTPUT_ROOT + "/{industry}/{scenario}/validated.done.json",
    industry=INDUSTRIES,
    scenario=SCENARIOS,
)
COMBINED = expand(
    RESULT_OUTPUT_ROOT + "/{industry}/core_scenarios.csv",
    industry=INDUSTRIES,
)
FINDINGS = expand(
    RESULT_OUTPUT_ROOT + "/{industry}/{scenario}/findings.md",
    industry=INDUSTRIES,
    scenario=SCENARIOS,
)
NATIONAL_SUMMARY = RESULT_OUTPUT_ROOT + "/national/core_scenarios.csv"
NATIONAL_VALIDATION = RESULT_OUTPUT_ROOT + "/national/validated.done.json"
NATIONAL_FINDINGS = RESULT_OUTPUT_ROOT + "/national/findings.md"
INDUSTRY_COST_DIFFERENCE_ROOT = RESULT_OUTPUT_ROOT + "/national/industry_cost_differences"
INDUSTRY_COST_DIFFERENCE_DETAIL = INDUSTRY_COST_DIFFERENCE_ROOT + "/core_scenario_detail.csv"
INDUSTRY_COST_DRIVER_ASSOCIATIONS = INDUSTRY_COST_DIFFERENCE_ROOT + "/driver_associations.csv"
INDUSTRY_COST_GAP_DECOMPOSITION = INDUSTRY_COST_DIFFERENCE_ROOT + "/extreme_gap_decomposition.csv"
INDUSTRY_COST_DIFFERENCE_FINDINGS = INDUSTRY_COST_DIFFERENCE_ROOT + "/findings.md"
INDUSTRY_COST_DIFFERENCE_DONE = INDUSTRY_COST_DIFFERENCE_ROOT + "/validated.done.json"
MANUSCRIPT_FIGURE_ROOT = RESULT_OUTPUT_ROOT + "/manuscript_figures"
FIGURE1_METHOD_PNG = MANUSCRIPT_FIGURE_ROOT + "/figure1_method.png"
FIGURE1_METHOD_SVG = MANUSCRIPT_FIGURE_ROOT + "/figure1_method.svg"
FIGURE1_DATA = MANUSCRIPT_FIGURE_ROOT + "/figure1_demand_architecture_data.csv"
FIGURE1_PNG = MANUSCRIPT_FIGURE_ROOT + "/figure1_demand_architecture.png"
FIGURE1_PDF = MANUSCRIPT_FIGURE_ROOT + "/figure1_demand_architecture.pdf"
FIGURE1_SVG = MANUSCRIPT_FIGURE_ROOT + "/figure1_demand_architecture.svg"
FIGURE1_DONE = MANUSCRIPT_FIGURE_ROOT + "/figure1_demand_architecture.validated.done.json"
FIGURE2_DATA = MANUSCRIPT_FIGURE_ROOT + "/figure2_enterprise_cost_data.csv"
FIGURE2_PNG = MANUSCRIPT_FIGURE_ROOT + "/figure2_enterprise_cost.png"
FIGURE2_PDF = MANUSCRIPT_FIGURE_ROOT + "/figure2_enterprise_cost.pdf"
FIGURE2_SVG = MANUSCRIPT_FIGURE_ROOT + "/figure2_enterprise_cost.svg"
FIGURE2_DONE = MANUSCRIPT_FIGURE_ROOT + "/figure2_enterprise_cost.validated.done.json"
FIGURE3_DATA = MANUSCRIPT_FIGURE_ROOT + "/figure3_grid_capacity_data.csv"
FIGURE3_SVG = MANUSCRIPT_FIGURE_ROOT + "/figure3_grid_capacity.svg"
FIGURE3_PNG = MANUSCRIPT_FIGURE_ROOT + "/figure3_grid_capacity.png"
FIGURE3_DONE = MANUSCRIPT_FIGURE_ROOT + "/figure3_grid_capacity.validated.done.json"
FIGURE4_DATA = MANUSCRIPT_FIGURE_ROOT + "/figure4_resource_footprint_data.csv"
FIGURE4_SVG = MANUSCRIPT_FIGURE_ROOT + "/figure4_resource_footprint.svg"
FIGURE4_PNG = MANUSCRIPT_FIGURE_ROOT + "/figure4_resource_footprint.png"
FIGURE4_DONE = MANUSCRIPT_FIGURE_ROOT + "/figure4_resource_footprint.validated.done.json"
FIGURE5_DATA = MANUSCRIPT_FIGURE_ROOT + "/figure5_spatial_concentration_data.csv"
FIGURE5_CLOUD_DATA = MANUSCRIPT_FIGURE_ROOT + "/figure5_cloud_spatial_scenario_data.csv"
FIGURE5_SVG = MANUSCRIPT_FIGURE_ROOT + "/figure5_spatial_concentration.svg"
FIGURE5_PNG = MANUSCRIPT_FIGURE_ROOT + "/figure5_spatial_concentration.png"
NATIONAL_TARGETS = (
    [
        NATIONAL_SUMMARY,
        NATIONAL_VALIDATION,
        NATIONAL_FINDINGS,
        INDUSTRY_COST_DIFFERENCE_DETAIL,
        INDUSTRY_COST_DRIVER_ASSOCIATIONS,
        INDUSTRY_COST_GAP_DECOMPOSITION,
        INDUSTRY_COST_DIFFERENCE_FINDINGS,
        INDUSTRY_COST_DIFFERENCE_DONE,
    ]
    if len(INDUSTRIES) == 31
    else []
)
COMPUTE_EFFICIENCY_VALIDATION = RESULT_OUTPUT_ROOT + "/compute_efficiency/cases.csv"
COMPUTE_EFFICIENCY_FINDINGS = RESULT_OUTPUT_ROOT + "/compute_efficiency/findings.md"
COMPUTE_EFFICIENCY_DONE = RESULT_OUTPUT_ROOT + "/compute_efficiency/validated.done.json"
MODEL_LIFECYCLE_VALIDATION = RESULT_OUTPUT_ROOT + "/model_lifecycle/scenarios.csv"
MODEL_LIFECYCLE_FINDINGS = RESULT_OUTPUT_ROOT + "/model_lifecycle/findings.md"
MODEL_LIFECYCLE_DONE = RESULT_OUTPUT_ROOT + "/model_lifecycle/validated.done.json"
CLOUD_SUBSCRIPTION_COMPARISON = RESULT_OUTPUT_ROOT + "/cloud_subscription/comparison.csv"
CLOUD_SUBSCRIPTION_BREAK_EVEN = RESULT_OUTPUT_ROOT + "/cloud_subscription/break_even.csv"
CLOUD_SUBSCRIPTION_FINDINGS = RESULT_OUTPUT_ROOT + "/cloud_subscription/findings.md"
CLOUD_SUBSCRIPTION_DONE = RESULT_OUTPUT_ROOT + "/cloud_subscription/validated.done.json"
API_TOKEN_COST_COMPARISON = RESULT_OUTPUT_ROOT + "/api_token_cost/comparison.csv"
API_TOKEN_MAINSTREAM_COMPARISON = RESULT_OUTPUT_ROOT + "/api_token_cost/mainstream_total_cost_comparison.csv"
API_TOKEN_COST_TASK_DETAIL = RESULT_OUTPUT_ROOT + "/api_token_cost/task_token_demand.csv"
API_TOKEN_COST_FINDINGS = RESULT_OUTPUT_ROOT + "/api_token_cost/findings.md"
API_TOKEN_COST_DONE = RESULT_OUTPUT_ROOT + "/api_token_cost/validated.done.json"
US_COST_PARAMETERS = "02_data/processed/cost_benchmark/us_core_cost_parameters_v1.csv"
US_API_TOKEN_PRICES = "02_data/processed/cost_benchmark/us_api_token_prices_v1.csv"
US_OWNED_COST_CONFIG = COUNTRY_CASES["us"]["owned_cost_config"]
US_FULL_CLOUD_CONFIG = COUNTRY_CASES["us"]["full_cloud_cost_config"]
US_HETEROGENEOUS_COST_CONFIG = COUNTRY_CASES["us"]["heterogeneous_cpu_cost_config"]
US_OWNED_COST_ROOT = RESULT_OUTPUT_ROOT + "/cost_benchmark/us_core_cost_v1"
US_OWNED_COST = US_OWNED_COST_ROOT + "/us_owned_core_cost.csv"
US_OWNED_FINDINGS = US_OWNED_COST_ROOT + "/findings.md"
US_OWNED_DONE = US_OWNED_COST_ROOT + "/validated.done.json"
US_FULL_CLOUD_ROOT = RESULT_OUTPUT_ROOT + "/cost_benchmark/us_full_cloud_v1"
US_FULL_CLOUD_COMPARISON = US_FULL_CLOUD_ROOT + "/us_full_cloud_comparison.csv"
US_FULL_CLOUD_AUDIT = US_FULL_CLOUD_ROOT + "/us_full_cloud_all_provider_audit.csv"
US_LOCAL_CLOUD_TOTAL_COMPARISON = US_FULL_CLOUD_ROOT + "/us_local_cloud_total_comparison.csv"
COUNTRY_PRICE_ENVIRONMENT_SUMMARY = US_FULL_CLOUD_ROOT + "/country_price_environment_summary.csv"
US_FULL_CLOUD_FINDINGS = US_FULL_CLOUD_ROOT + "/findings.md"
US_FULL_CLOUD_DONE = US_FULL_CLOUD_ROOT + "/validated.done.json"
CPU_GPU_ROUTING_CONFIG = HARDWARE_CASE["routing_config"]
ACTIVE_ROUTING_CASE = HARDWARE_CASE["active_routing_case"]
ROUTING_CONFIG = yaml.safe_load(open(CPU_GPU_ROUTING_CONFIG, encoding="utf-8"))
if ROUTING_CONFIG["active_core_routing_case"] != ACTIVE_ROUTING_CASE:
    raise ValueError("Scenario registry and hardware routing config select different active cases")
SINGLE_INDUSTRY_HETEROGENEOUS_ROOT = RESULT_OUTPUT_ROOT + "/cost_benchmark/c36_heterogeneous_hardware_v1"
SINGLE_INDUSTRY_HETEROGENEOUS_COMPARISON = SINGLE_INDUSTRY_HETEROGENEOUS_ROOT + "/comparison.csv"
SINGLE_INDUSTRY_HETEROGENEOUS_ROUTING = SINGLE_INDUSTRY_HETEROGENEOUS_ROOT + "/routing_parameters.csv"
SINGLE_INDUSTRY_HETEROGENEOUS_FINDINGS = SINGLE_INDUSTRY_HETEROGENEOUS_ROOT + "/findings.md"
SINGLE_INDUSTRY_HETEROGENEOUS_DONE = SINGLE_INDUSTRY_HETEROGENEOUS_ROOT + "/validated.done.json"
SINGLE_INDUSTRY_HETEROGENEOUS_US_ROOT = SINGLE_INDUSTRY_HETEROGENEOUS_ROOT + "/us_cost_environment"
SINGLE_INDUSTRY_HETEROGENEOUS_US_COMPARISON = SINGLE_INDUSTRY_HETEROGENEOUS_US_ROOT + "/us_comparison.csv"
SINGLE_INDUSTRY_HETEROGENEOUS_US_CPU_SENSITIVITY = SINGLE_INDUSTRY_HETEROGENEOUS_US_ROOT + "/us_cpu_server_purchase_price_sensitivity.csv"
SINGLE_INDUSTRY_HETEROGENEOUS_US_FINDINGS = SINGLE_INDUSTRY_HETEROGENEOUS_US_ROOT + "/findings.md"
SINGLE_INDUSTRY_HETEROGENEOUS_US_DONE = SINGLE_INDUSTRY_HETEROGENEOUS_US_ROOT + "/validated.done.json"
HETEROGENEOUS_ROOT = RESULT_OUTPUT_ROOT + "/cost_benchmark/heterogeneous_hardware_v1"
CHINA_HETEROGENEOUS_ROOT = HETEROGENEOUS_ROOT + "/china_industry"
CHINA_HETEROGENEOUS_ARCHITECTURES = ["IF", "IG", "II_1host"]
CHINA_HETEROGENEOUS_COMPARISONS = expand(
    CHINA_HETEROGENEOUS_ROOT + "/{architecture}/{industry}/comparison.csv",
    architecture=CHINA_HETEROGENEOUS_ARCHITECTURES,
    industry=INDUSTRIES,
)
CHINA_HETEROGENEOUS_DONE = expand(
    CHINA_HETEROGENEOUS_ROOT + "/{architecture}/{industry}/validated.done.json",
    architecture=CHINA_HETEROGENEOUS_ARCHITECTURES,
    industry=INDUSTRIES,
)
CHINA_HETEROGENEOUS_NATIONAL = HETEROGENEOUS_ROOT + "/china_national/comparison.csv"
CHINA_HETEROGENEOUS_NATIONAL_FINDINGS = HETEROGENEOUS_ROOT + "/china_national/findings.md"
CHINA_HETEROGENEOUS_NATIONAL_DONE = HETEROGENEOUS_ROOT + "/china_national/validated.done.json"
US_HETEROGENEOUS_ROOT = HETEROGENEOUS_ROOT + "/us_naics3"
US_HETEROGENEOUS_DETAIL = US_HETEROGENEOUS_ROOT + "/us_naics3_comparison.csv"
US_HETEROGENEOUS_NATIONAL = US_HETEROGENEOUS_ROOT + "/us_national_comparison.csv"
US_HETEROGENEOUS_FINDINGS = US_HETEROGENEOUS_ROOT + "/findings.md"
US_HETEROGENEOUS_DONE = US_HETEROGENEOUS_ROOT + "/validated.done.json"
HETEROGENEOUS_COUNTRY_SUMMARY = HETEROGENEOUS_ROOT + "/country_comparison.csv"
HETEROGENEOUS_FINDINGS = HETEROGENEOUS_ROOT + "/findings.md"
HETEROGENEOUS_DONE = HETEROGENEOUS_ROOT + "/validated.done.json"
US_DEMAND_CONFIG = COUNTRY_CASES["us"]["demand_config"]
US_DEMAND_ROOT = RESULT_OUTPUT_ROOT + "/us_demand"
US_DEMAND_SERVICE = "02_data/processed/us_demand/us_manufacturing_ai_effective_service_2030.csv"
US_DEMAND_LINEAGE = "02_data/processed/us_demand/us_manufacturing_ai_demand_lineage.json"
US_DEMAND_TASK_SUMMARY = US_DEMAND_ROOT + "/us_national_task_summary.csv"
US_DEMAND_NAICS_SUMMARY = US_DEMAND_ROOT + "/us_naics3_task_summary.csv"
US_DEMAND_VALIDATION = US_DEMAND_ROOT + "/us_demand_validation.csv"
US_DEMAND_MACRO_ALIGNMENT = US_DEMAND_ROOT + "/us_macro_alignment.csv"
US_DEMAND_PARAMETER_AUDIT = US_DEMAND_ROOT + "/us_bottom_up_parameter_audit.csv"
US_DEMAND_COST_SENSITIVITY = US_DEMAND_ROOT + "/us_cost_ratio_sensitivity.csv"
US_DEMAND_LOCAL_COST = US_DEMAND_ROOT + "/us_local_cost.csv"
US_DEMAND_CLOUD_COST = US_DEMAND_ROOT + "/us_full_cloud_cost.csv"
US_DEMAND_COMPARISON = US_DEMAND_ROOT + "/comparison.csv"
US_DEMAND_FINDINGS = US_DEMAND_ROOT + "/findings.md"
US_DEMAND_DONE = US_DEMAND_ROOT + "/validated.done.json"
LOAD_ALIGNMENT_DETAIL = RESULT_OUTPUT_ROOT + "/load_alignment/industry_scenarios.csv"
LOAD_ALIGNMENT_SUMMARY = RESULT_OUTPUT_ROOT + "/load_alignment/architecture_summary.csv"
LOAD_ALIGNMENT_FINDINGS = RESULT_OUTPUT_ROOT + "/load_alignment/findings.md"
LOAD_ALIGNMENT_DONE = RESULT_OUTPUT_ROOT + "/load_alignment/validated.done.json"
FLEXIBILITY_ABLATION_COMPARISON = RESULT_OUTPUT_ROOT + "/flexibility_ablation/comparison.csv"
FLEXIBILITY_ABLATION_FINDINGS = RESULT_OUTPUT_ROOT + "/flexibility_ablation/findings.md"
FLEXIBILITY_ABLATION_DONE = RESULT_OUTPUT_ROOT + "/flexibility_ablation/validated.done.json"
TYPICAL_LOAD_STACKING_ROOT = RESULT_OUTPUT_ROOT + "/typical_industry_load_stacking"
TYPICAL_LOAD_STACKING_PROFILES = TYPICAL_LOAD_STACKING_ROOT + "/profiles.csv"
TYPICAL_LOAD_STACKING_SUMMARY = TYPICAL_LOAD_STACKING_ROOT + "/summary.csv"
TYPICAL_LOAD_STACKING_FIGURE = TYPICAL_LOAD_STACKING_ROOT + "/load_stacking.png"
TYPICAL_AI_LOAD_FIGURE = TYPICAL_LOAD_STACKING_ROOT + "/ai_load_only.png"
TYPICAL_LOAD_STACKING_FINDINGS = TYPICAL_LOAD_STACKING_ROOT + "/findings.md"
TYPICAL_LOAD_STACKING_DONE = TYPICAL_LOAD_STACKING_ROOT + "/validated.done.json"
SPOT_PV_ROOT = RESULT_OUTPUT_ROOT + "/industry_spot_price_pv"
SPOT_PV_HOURLY = SPOT_PV_ROOT + "/{industry}/hourly.csv"
SPOT_PV_SUMMARY = SPOT_PV_ROOT + "/{industry}/summary.csv"
SPOT_PV_FIGURE = SPOT_PV_ROOT + "/{industry}/comparison.png"
SPOT_PV_FINDINGS = SPOT_PV_ROOT + "/{industry}/findings.md"
SPOT_PV_INDUSTRY_DONE = SPOT_PV_ROOT + "/{industry}/validated.done.json"
SPOT_PV_ALL_CASES = SPOT_PV_ROOT + "/all_industry_cases.csv"
SPOT_PV_COMPARISON = SPOT_PV_ROOT + "/industry_comparison.csv"
SPOT_PV_ALL_FINDINGS = SPOT_PV_ROOT + "/findings.md"
SPOT_PV_ALL_DONE = SPOT_PV_ROOT + "/validated.done.json"
C40_SPOT_PV_SUMMARY = SPOT_PV_ROOT + "/C40/summary.csv"
BOLUN_PROGRESS_BRIEFING = RESULT_OUTPUT_ROOT + "/briefing/bolun_progress_briefing_2026-08-12.html"
BOLUN_SENSITIVITY_FACTOR_RESULTS = "05_results/sensitivity/v0.8.0/single_industry_oat/result/factor_results.csv"
BOLUN_CORE_GRID_COMPARISON = "05_results/sensitivity/v0.8.0/national_grid_capacity_comparison.csv"
BOLUN_NO_SHIFT_GRID_COMPARISON = "05_results/sensitivity/v0.8.0/national_oat_extension/result/PHY03__no_shift/grid_capacity_comparison.csv"
LAND_MATERIAL_ROOT = RESULT_OUTPUT_ROOT + "/resource_footprint_land_materials"
LAND_SPACE_RESULTS = "02_data/processed/resource_footprint/land_space_scenarios.csv"
LAND_MATERIAL_RESULTS = "02_data/processed/resource_footprint/building_material_scenarios.csv"
LAND_MATERIAL_CROSSCHECK = "02_data/processed/resource_footprint/hyperscale_material_crosscheck.csv"
LAND_MATERIAL_LINEAGE = "02_data/processed/resource_footprint/land_materials.lineage.json"
LAND_MATERIAL_FINDINGS = LAND_MATERIAL_ROOT + "/findings.md"
LAND_MATERIAL_DONE = LAND_MATERIAL_ROOT + "/validated.done.json"
ADDITIONAL_ANALYSIS_TARGETS = (
    [
        NATIONAL_CLOUD_SUMMARY,
        FIGURE1_DATA,
        FIGURE1_METHOD_PNG,
        FIGURE1_METHOD_SVG,
        FIGURE1_PNG,
        FIGURE1_PDF,
        FIGURE1_SVG,
        FIGURE1_DONE,
        FIGURE2_DATA,
        FIGURE2_PNG,
        FIGURE2_PDF,
        FIGURE2_SVG,
        FIGURE2_DONE,
        FIGURE3_DATA,
        FIGURE3_SVG,
        FIGURE3_PNG,
        FIGURE3_DONE,
        FIGURE4_DATA,
        FIGURE4_SVG,
        FIGURE4_PNG,
        FIGURE4_DONE,
        FIGURE5_DATA,
        FIGURE5_CLOUD_DATA,
        FIGURE5_SVG,
        FIGURE5_PNG,
    ]
    if len(INDUSTRIES) == 31
    else []
)

OPTIONAL_DIAGNOSTIC_TARGETS = (
    [
        FLEXIBILITY_ABLATION_COMPARISON,
        FLEXIBILITY_ABLATION_FINDINGS,
        FLEXIBILITY_ABLATION_DONE,
        SPOT_PV_ALL_CASES,
        SPOT_PV_COMPARISON,
        SPOT_PV_ALL_FINDINGS,
        SPOT_PV_ALL_DONE,
        BOLUN_PROGRESS_BRIEFING,
    ]
    if len(INDUSTRIES) == 31
    else []
)


rule all:
    input:
        MODEL_READY_ROOFTOP,
        MODEL_READY_ROOFTOP_LINEAGE,
        MODEL_READY_DONE,
        MODEL_READY_COMPUTE_EFFICIENCY,
        MODEL_READY_COMPUTE_EFFICIENCY_LINEAGE,
        MODEL_READY_MODEL_LIFECYCLE,
        COMPUTE_EFFICIENCY_VALIDATION,
        COMPUTE_EFFICIENCY_FINDINGS,
        COMPUTE_EFFICIENCY_DONE,
        MODEL_LIFECYCLE_VALIDATION if len(INDUSTRIES) == 31 else [],
        MODEL_LIFECYCLE_FINDINGS if len(INDUSTRIES) == 31 else [],
        MODEL_LIFECYCLE_DONE if len(INDUSTRIES) == 31 else [],
        RESULT_OUTPUT_ROOT + "/tests/unit_tests.done",
        GROUP_CORE_TARGETS,


rule extended_analysis:
    input:
        ADDITIONAL_ANALYSIS_TARGETS,


rule core:
    input:
        GROUP_CORE_TARGETS,


rule core_group_architectures:
    input:
        GROUP_CORE_TARGETS,


rule run_group_architecture_core_industry:
    input:
        registry=GROUP_CORE_REGISTRY,
        defaults="config/defaults.yaml",
        run_config=RUN_CONFIG,
        common=COMMON_INPUTS,
        archive=config["paths"]["eweld_archive"],
        script="08_code/run_group_multisite_continuous_test.py",
    output:
        summary=GROUP_CORE_ROOT + "/{industry}/summary.csv",
        hourly=GROUP_CORE_ROOT + "/{industry}/hourly.csv",
        lineage=GROUP_CORE_ROOT + "/{industry}/curve_lineage.csv",
        alignment=GROUP_CORE_ROOT + "/{industry}/load_alignment_value.csv",
        metadata=GROUP_CORE_ROOT + "/{industry}/metadata.json",
    wildcard_constraints:
        industry="|".join(INDUSTRIES),
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults {input.defaults} --config {input.run_config} "
        "--experiment {input.registry} --industry {wildcards.industry} "
        "--output-dir " + GROUP_CORE_ROOT + "/{wildcards.industry}"


rule summarize_group_architecture_core:
    input:
        summaries=GROUP_CORE_SUMMARIES,
        lineages=GROUP_CORE_LINEAGES,
        alignments=GROUP_CORE_ALIGNMENTS,
        metadata=GROUP_CORE_METADATA,
        script="08_code/summarize_group_multisite_core.py",
    output:
        summary=GROUP_CORE_NATIONAL_SUMMARY,
        alignment=GROUP_CORE_NATIONAL_ALIGNMENT,
        lineage=GROUP_CORE_NATIONAL_LINEAGE,
        done=GROUP_CORE_NATIONAL_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --root " + GROUP_CORE_ROOT + " "
        "--industries " + " ".join(INDUSTRIES) + " "
        "--summary-output {output.summary} --alignment-output {output.alignment} "
        "--lineage-output {output.lineage} --done-output {output.done}"


rule single_industry_heterogeneous_cost:
    input:
        SINGLE_INDUSTRY_HETEROGENEOUS_COMPARISON,
        SINGLE_INDUSTRY_HETEROGENEOUS_ROUTING,
        SINGLE_INDUSTRY_HETEROGENEOUS_FINDINGS,
        SINGLE_INDUSTRY_HETEROGENEOUS_DONE,
        SINGLE_INDUSTRY_HETEROGENEOUS_US_COMPARISON,
        SINGLE_INDUSTRY_HETEROGENEOUS_US_CPU_SENSITIVITY,
        SINGLE_INDUSTRY_HETEROGENEOUS_US_FINDINGS,
        SINGLE_INDUSTRY_HETEROGENEOUS_US_DONE,


rule china_heterogeneous_cost:
    input:
        CHINA_HETEROGENEOUS_COMPARISONS,
        CHINA_HETEROGENEOUS_DONE,
        CHINA_HETEROGENEOUS_NATIONAL,
        CHINA_HETEROGENEOUS_NATIONAL_FINDINGS,
        CHINA_HETEROGENEOUS_NATIONAL_DONE,


rule us_heterogeneous_cost:
    input:
        US_HETEROGENEOUS_DETAIL,
        US_HETEROGENEOUS_NATIONAL,
        US_HETEROGENEOUS_FINDINGS,
        US_HETEROGENEOUS_DONE,


rule heterogeneous_cost:
    input:
        CHINA_HETEROGENEOUS_NATIONAL_DONE,
        US_HETEROGENEOUS_DONE,
        HETEROGENEOUS_COUNTRY_SUMMARY,
        HETEROGENEOUS_DONE,


rule optional_diagnostics:
    input:
        OPTIONAL_DIAGNOSTIC_TARGETS,


rule prepare_core_base_load_profiles:
    input:
        source=LEGACY_HOURLY_PROFILES,
        script="08_code/prepare_core_base_load_profiles.py",
    output:
        profiles=DAILY_HOURLY_PROFILES,
        lineage=DAILY_HOURLY_LINEAGE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --input {input.source} --output {output.profiles} "
        "--lineage-output {output.lineage}"


rule prepare_core_real_week_profiles:
    input:
        archive=config["paths"]["eweld_archive"],
        daily=DAILY_HOURLY_PROFILES,
        crosswalk=config["paths"]["manufacturing_load_crosswalk"],
        script="08_code/prepare_eweld_representative_weeks.py",
    output:
        profiles=MODEL_READY_HOURLY_PROFILES,
        lineage=MODEL_READY_HOURLY_LINEAGE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --archive {input.archive} --daily-profile {input.daily} "
        "--crosswalk {input.crosswalk} --output {output.profiles} "
        "--lineage-output {output.lineage}"


rule prepare_industry_rooftop_parameters:
    input:
        crosswalk=ROOFTOP_CROSSWALK,
        mecs=RAW_US_MECS,
        script="08_code/prepare_industry_rooftop_parameters.py",
    output:
        parameters=MODEL_READY_ROOFTOP,
        lineage=MODEL_READY_ROOFTOP_LINEAGE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --crosswalk {input.crosswalk} --mecs {input.mecs} "
        "--output {output.parameters} --lineage-output {output.lineage}"


rule prepare_effective_service_data:
    input:
        industry_baseline=config["paths"]["upstream_31sector_bottomup"],
        task_templates=config["paths"]["upstream_task_templates"],
        growth=config["paths"]["raw_effective_service_growth"],
        fallbacks=config["paths"]["raw_effective_service_template_fallbacks"],
        script="08_code/prepare_effective_service_data.py",
    output:
        service=MODEL_READY_SERVICE,
        summary=MODEL_READY_SUMMARY,
        lineage=MODEL_READY_LINEAGE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --industry-baseline {input.industry_baseline} "
        "--task-templates {input.task_templates} --growth-scenarios {input.growth} "
        "--template-fallbacks {input.fallbacks} --output {output.service} "
        "--summary-output {output.summary} --lineage-output {output.lineage}"


rule prepare_compute_efficiency_data:
    input:
        source=RAW_COMPUTE_EFFICIENCY,
        script="08_code/prepare_compute_efficiency_data.py",
    output:
        table=MODEL_READY_COMPUTE_EFFICIENCY,
        lineage=MODEL_READY_COMPUTE_EFFICIENCY_LINEAGE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --input {input.source} --output {output.table} "
        "--lineage-output {output.lineage}"


rule prepare_model_lifecycle_data:
    input:
        parameters=RAW_MODEL_LIFECYCLE_PARAMETERS,
        hardware=RAW_MODEL_HARDWARE_MATRIX,
        script="08_code/prepare_model_lifecycle_data.py",
    output:
        MODEL_READY_MODEL_LIFECYCLE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --parameters {input.parameters} "
        "--hardware-matrix {input.hardware} --output {output}"


rule validate_compute_efficiency_cases:
    input:
        table=MODEL_READY_COMPUTE_EFFICIENCY,
        service=MODEL_READY_SERVICE,
        hourly=MODEL_READY_HOURLY_PROFILES,
        defaults="config/defaults.yaml",
        run_config="config/runs/all_industries_core.yaml",
        script="08_code/validate_compute_efficiency_cases.py",
    output:
        table=COMPUTE_EFFICIENCY_VALIDATION,
        findings=COMPUTE_EFFICIENCY_FINDINGS,
        done=COMPUTE_EFFICIENCY_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --defaults {input.defaults} --config {input.run_config} "
        "--table {input.table} --output {output.table} "
        "--findings-output {output.findings} --done-output {output.done}"


rule validate_model_lifecycle_results:
    input:
        summaries=GROUP_CORE_SUMMARIES,
        lifecycle=MODEL_READY_MODEL_LIFECYCLE,
        defaults="config/defaults.yaml",
        script="08_code/validate_model_lifecycle_results.py",
    output:
        table=MODEL_LIFECYCLE_VALIDATION,
        findings=MODEL_LIFECYCLE_FINDINGS,
        done=MODEL_LIFECYCLE_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --inputs {input.summaries} --lifecycle {input.lifecycle} "
        "--defaults {input.defaults} --model-version {MODEL_VERSION} "
        "--output {output.table} --findings-output {output.findings} "
        "--done-output {output.done}"


rule validate_effective_service_data:
    input:
        service=MODEL_READY_SERVICE,
        hourly=MODEL_READY_HOURLY_PROFILES,
        defaults="config/defaults.yaml",
        run_config="config/runs/all_industries_core.yaml",
        script="08_code/validate_effective_service_data.py",
    output:
        validation=MODEL_READY_VALIDATION,
        findings=MODEL_READY_FINDINGS,
        done=MODEL_READY_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --service-input {input.service} "
        "--defaults {input.defaults} --run-config {input.run_config} "
        "--output {output.validation} --findings-output {output.findings} "
        "--done-output {output.done}"


rule unit_tests:
    input:
        COMMON_INPUTS,
        "tests/test_core_config_and_scaling.py",
        "tests/test_core_land_material_footprint.py",
        "08_code/analyze_land_material_footprint.py",
        LAND_SPACE_PARAMETERS,
        LAND_MATERIAL_PARAMETERS,
        LAND_MATERIAL_CROSSCHECK_PARAMETERS,
    output:
        RESULT_OUTPUT_ROOT + "/tests/unit_tests.done",
    conda:
        "../envs/core_model.yaml"
    run:
        import subprocess
        from pathlib import Path

        subprocess.check_call(
            ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_core_*.py"]
        )
        done = Path(output[0])
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("ok\n", encoding="utf-8")


rule core_baseline:
    input:
        COMMON_INPUTS,
        script="08_code/run_core_baseline.py",
    output:
        summary=MODEL_OUTPUT_ROOT + "/{industry}/baseline/summary.json",
        hourly=MODEL_OUTPUT_ROOT + "/{industry}/baseline/hourly.csv",
        resolved=MODEL_OUTPUT_ROOT + "/{industry}/baseline/resolved_config.yaml",
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults config/defaults.yaml --config {RUN_CONFIG} "
        "--industry {wildcards.industry} --summary-output {output.summary} "
        "--hourly-output {output.hourly} --resolved-config-output {output.resolved}"


rule core_scenario:
    input:
        COMMON_INPUTS,
        script="08_code/run_core_scenario.py",
        baseline=MODEL_OUTPUT_ROOT + "/{industry}/baseline/summary.json",
    output:
        summary=MODEL_OUTPUT_ROOT + "/{industry}/{scenario}/summary.csv",
        hourly=MODEL_OUTPUT_ROOT + "/{industry}/{scenario}/hourly.csv",
        resolved=MODEL_OUTPUT_ROOT + "/{industry}/{scenario}/resolved_config.yaml",
    conda:
        "../envs/core_model.yaml"
    threads: 5
    shell:
        "python {input.script} --defaults config/defaults.yaml --config {RUN_CONFIG} "
        "--industry {wildcards.industry} --scenario {wildcards.scenario} "
        "--baseline-summary {input.baseline} --summary-output {output.summary} "
        "--hourly-output {output.hourly} "
        "--resolved-config-output {output.resolved}"


rule validate_core_scenario:
    input:
        summary=MODEL_OUTPUT_ROOT + "/{industry}/{scenario}/summary.csv",
        hourly=MODEL_OUTPUT_ROOT + "/{industry}/{scenario}/hourly.csv",
        baseline=MODEL_OUTPUT_ROOT + "/{industry}/baseline/summary.json",
        script="08_code/validate_core_scenario.py",
        defaults="config/defaults.yaml",
        run_config=RUN_CONFIG,
    output:
        RESULT_OUTPUT_ROOT + "/{industry}/{scenario}/validated.done.json",
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --defaults {input.defaults} --config {input.run_config} "
        "--industry {wildcards.industry} --scenario {wildcards.scenario} "
        "--summary {input.summary} --hourly {input.hourly} "
        "--baseline-summary {input.baseline} --output {output}"


rule analyze_core_scenario:
    input:
        summary=MODEL_OUTPUT_ROOT + "/{industry}/{scenario}/summary.csv",
        validation=RESULT_OUTPUT_ROOT + "/{industry}/{scenario}/validated.done.json",
        script="08_code/analyze_core_scenario.py",
    output:
        RESULT_OUTPUT_ROOT + "/{industry}/{scenario}/findings.md",
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --summary {input.summary} "
        "--validation {input.validation} --output {output}"


rule combine_core_summaries:
    input:
        summaries=lambda wildcards: expand(
            MODEL_OUTPUT_ROOT + "/{industry}/{scenario}/summary.csv",
            industry=wildcards.industry,
            scenario=SCENARIOS,
        ),
        validations=lambda wildcards: expand(
            RESULT_OUTPUT_ROOT + "/{industry}/{scenario}/validated.done.json",
            industry=wildcards.industry,
            scenario=SCENARIOS,
        ),
        script="08_code/combine_core_summaries.py",
    output:
        RESULT_OUTPUT_ROOT + "/{industry}/core_scenarios.csv",
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --inputs {input.summaries} --output {output}"


rule combine_national_core_summaries:
    input:
        summaries=expand(
            MODEL_OUTPUT_ROOT + "/{industry}/{scenario}/summary.csv",
            industry=INDUSTRIES,
            scenario=SCENARIOS,
        ),
        validations=expand(
            RESULT_OUTPUT_ROOT + "/{industry}/{scenario}/validated.done.json",
            industry=INDUSTRIES,
            scenario=SCENARIOS,
        ),
        script="08_code/combine_core_summaries.py",
    output:
        NATIONAL_SUMMARY,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --inputs {input.summaries} --output {output}"


rule validate_national_core_results:
    input:
        summary=NATIONAL_SUMMARY,
        script="08_code/validate_national_core_results.py",
    output:
        NATIONAL_VALIDATION,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --input {input.summary} --model-version {MODEL_VERSION} "
        "--output {output}"


rule analyze_national_core_results:
    input:
        validation=NATIONAL_VALIDATION,
        deployment_config=DEPLOYMENT_CONFIG,
        script="08_code/analyze_national_core_results.py",
    output:
        NATIONAL_FINDINGS,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --validation {input.validation} --output {output}"


rule analyze_core_industry_cost_differences:
    input:
        national=GROUP_CORE_NATIONAL_SUMMARY,
        validation=GROUP_CORE_NATIONAL_DONE,
        service=MODEL_READY_SERVICE,
        routing_config=HARDWARE_CASE["routing_config"],
        script="08_code/analyze_core_industry_cost_differences.py",
    output:
        detail=INDUSTRY_COST_DIFFERENCE_DETAIL,
        associations=INDUSTRY_COST_DRIVER_ASSOCIATIONS,
        decomposition=INDUSTRY_COST_GAP_DECOMPOSITION,
        findings=INDUSTRY_COST_DIFFERENCE_FINDINGS,
        done=INDUSTRY_COST_DIFFERENCE_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --national-input {input.national} --service-input {input.service} "
        "--routing-config {input.routing_config} --detail-output {output.detail} "
        "--association-output {output.associations} --decomposition-output {output.decomposition} "
        "--findings-output {output.findings} --done-output {output.done}"


rule core_industry_cost_differences:
    input:
        INDUSTRY_COST_DIFFERENCE_DETAIL,
        INDUSTRY_COST_DRIVER_ASSOCIATIONS,
        INDUSTRY_COST_GAP_DECOMPOSITION,
        INDUSTRY_COST_DIFFERENCE_FINDINGS,
        INDUSTRY_COST_DIFFERENCE_DONE,


rule build_figure1_method:
    input:
        script="08_code/build_figure1_method.py",
    output:
        png=FIGURE1_METHOD_PNG,
        svg=FIGURE1_METHOD_SVG,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --png-output {output.png} --svg-output {output.svg}"


rule build_figure1_demand_architecture:
    input:
        service=MODEL_READY_SERVICE,
        national=GROUP_CORE_NATIONAL_SUMMARY,
        routing_config=CPU_GPU_ROUTING_CONFIG,
        validation=GROUP_CORE_NATIONAL_DONE,
        script="08_code/build_figure1_demand_architecture.py",
    output:
        data=FIGURE1_DATA,
        png=FIGURE1_PNG,
        pdf=FIGURE1_PDF,
        svg=FIGURE1_SVG,
        done=FIGURE1_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --service-input {input.service} --national-input {input.national} "
        "--routing-config {input.routing_config} "
        "--model-version {MODEL_VERSION} --data-output {output.data} --png-output {output.png} "
        "--pdf-output {output.pdf} --svg-output {output.svg} --validation-output {output.done}"


rule build_figure2_enterprise_cost:
    input:
        national=GROUP_CORE_NATIONAL_SUMMARY,
        alignment=GROUP_CORE_NATIONAL_ALIGNMENT,
        validation=GROUP_CORE_NATIONAL_DONE,
        script="08_code/build_figure2_enterprise_cost.py",
    output:
        data=FIGURE2_DATA,
        png=FIGURE2_PNG,
        pdf=FIGURE2_PDF,
        svg=FIGURE2_SVG,
        done=FIGURE2_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --national-input {input.national} --alignment-input {input.alignment} "
        "--model-version {MODEL_VERSION} --data-output {output.data} --png-output {output.png} "
        "--pdf-output {output.pdf} --svg-output {output.svg} --validation-output {output.done}"


rule build_figure3_grid_capacity:
    input:
        summaries=GROUP_CORE_SUMMARIES,
        hourly=GROUP_CORE_HOURLY,
        script="08_code/build_figure3_grid_capacity.py",
    output:
        data=FIGURE3_DATA,
        svg=FIGURE3_SVG,
        png=FIGURE3_PNG,
        done=FIGURE3_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --core-root " + GROUP_CORE_ROOT + " "
        "--industries " + " ".join(INDUSTRIES) + " --model-version {MODEL_VERSION} "
        "--data-output {output.data} --svg-output {output.svg} "
        "--png-output {output.png} --validation-output {output.done}"


rule build_figure4_resource_footprint:
    input:
        scenario_registry=SCENARIO_REGISTRY_PATH,
        water=WATER_PARAMETERS,
        core=GROUP_CORE_NATIONAL_SUMMARY,
        cloud=NATIONAL_CLOUD_SUMMARY,
        script="08_code/build_figure4_resource_footprint.py",
    output:
        data=FIGURE4_DATA,
        svg=FIGURE4_SVG,
        png=FIGURE4_PNG,
        done=FIGURE4_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --scenario-registry {input.scenario_registry} --water-input {input.water} "
        "--core-input {input.core} --cloud-input {input.cloud} "
        "--model-version {MODEL_VERSION} --data-output {output.data} "
        "--svg-output {output.svg} --png-output {output.png} --validation-output {output.done}"


rule build_figure5_spatial_concentration:
    input:
        scenario_registry=SCENARIO_REGISTRY_PATH,
        routing_config=CPU_GPU_ROUTING_CONFIG,
        allocation=RESOURCE_CASE["spatial_water"]["province_allocation_file"],
        core=GROUP_CORE_NATIONAL_SUMMARY,
        scarcity=RESOURCE_CASE["spatial_water"]["scarcity_file"],
        cloud_share=RESOURCE_CASE["spatial_water"]["cloud_allocation_file"],
        map=RESOURCE_CASE["spatial_water"]["china_map_file"],
        deployment_config=DEPLOYMENT_CONFIG,
        script="08_code/build_figure5_spatial_concentration_draft.py",
    output:
        data=FIGURE5_DATA,
        cloud_data=FIGURE5_CLOUD_DATA,
        svg=FIGURE5_SVG,
        png=FIGURE5_PNG,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --scenario-registry {input.scenario_registry} "
        "--routing-config {input.routing_config} --allocation {input.allocation} "
        "--core-input {input.core} "
        "--scarcity {input.scarcity} --cloud-share {input.cloud_share} --map {input.map} --data-output {output.data} "
        "--cloud-data-output {output.cloud_data} --svg-output {output.svg} --png-output {output.png}"


rule analyze_land_material_footprint:
    input:
        national=NATIONAL_SUMMARY,
        scenario_registry=SCENARIO_REGISTRY_PATH,
        space_parameters=LAND_SPACE_PARAMETERS,
        material_parameters=LAND_MATERIAL_PARAMETERS,
        crosscheck_parameters=LAND_MATERIAL_CROSSCHECK_PARAMETERS,
        script="08_code/analyze_land_material_footprint.py",
    output:
        space=LAND_SPACE_RESULTS,
        materials=LAND_MATERIAL_RESULTS,
        crosscheck=LAND_MATERIAL_CROSSCHECK,
        lineage=LAND_MATERIAL_LINEAGE,
        findings=LAND_MATERIAL_FINDINGS,
        done=LAND_MATERIAL_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --national-summary {input.national} "
        "--scenario-registry {input.scenario_registry} "
        "--space-parameters {input.space_parameters} "
        "--material-parameters {input.material_parameters} "
        "--crosscheck-parameters {input.crosscheck_parameters} "
        "--space-output {output.space} --material-output {output.materials} "
        "--crosscheck-output {output.crosscheck} --lineage-output {output.lineage} "
        "--findings-output {output.findings} --done-output {output.done}"


rule analyze_cloud_subscription:
    input:
        summary=NATIONAL_SUMMARY,
        hourly=expand(
            MODEL_OUTPUT_ROOT + "/{industry}/II_1host/hourly.csv",
            industry=INDUSTRIES,
        ),
        prices=ENTERPRISE_AI_COST_PARAMETERS,
        defaults="config/defaults.yaml",
        run_config=RUN_CONFIG,
        script="08_code/analyze_cloud_subscription.py",
    output:
        comparison=CLOUD_SUBSCRIPTION_COMPARISON,
        break_even=CLOUD_SUBSCRIPTION_BREAK_EVEN,
        findings=CLOUD_SUBSCRIPTION_FINDINGS,
        done=CLOUD_SUBSCRIPTION_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --defaults {input.defaults} --config {input.run_config} "
        "--national-summary {input.summary} --hourly-inputs {input.hourly} "
        "--output {output.comparison} --break-even-output {output.break_even} "
        "--findings-output {output.findings} --done-output {output.done}"


rule analyze_api_token_cost:
    input:
        summary=NATIONAL_SUMMARY,
        service=MODEL_READY_SERVICE,
        workload=config["paths"]["raw_workload_parameters"],
        baseline="02_data/china_manufacturing_sector_baseline.csv",
        prices=API_TOKEN_PRICES,
        cloud_comparison=CLOUD_SUBSCRIPTION_COMPARISON,
        lifecycle_parameters=RAW_MODEL_LIFECYCLE_PARAMETERS,
        defaults="config/defaults.yaml",
        run_config=RUN_CONFIG,
        script="08_code/analyze_api_token_cost.py",
    output:
        comparison=API_TOKEN_COST_COMPARISON,
        mainstream_comparison=API_TOKEN_MAINSTREAM_COMPARISON,
        task_detail=API_TOKEN_COST_TASK_DETAIL,
        findings=API_TOKEN_COST_FINDINGS,
        done=API_TOKEN_COST_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --defaults {input.defaults} --config {input.run_config} "
        "--national-summary {input.summary} --service-input {input.service} "
        "--workload-input {input.workload} --industry-baseline {input.baseline} "
        "--api-prices {input.prices} --cloud-comparison {input.cloud_comparison} "
        "--lifecycle-parameters {input.lifecycle_parameters} "
        "--output {output.comparison} --mainstream-output {output.mainstream_comparison} "
        "--task-output {output.task_detail} --findings-output {output.findings} "
        "--done-output {output.done}"


rule analyze_us_owned_core_cost:
    input:
        national=NATIONAL_SUMMARY,
        parameters=US_COST_PARAMETERS,
        cost_config=US_OWNED_COST_CONFIG,
        script="08_code/analyze_us_owned_core_cost.py",
    output:
        cost=US_OWNED_COST,
        findings=US_OWNED_FINDINGS,
        done=US_OWNED_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --national-summary {input.national} "
        "--parameters {input.parameters} --cost-config {input.cost_config} "
        "--output {output.cost} --findings-output {output.findings} "
        "--done-output {output.done}"


rule analyze_us_full_cloud_cost:
    input:
        cloud_comparison=CLOUD_SUBSCRIPTION_COMPARISON,
        token_demand=API_TOKEN_COST_TASK_DETAIL,
        china_full_cloud=API_TOKEN_COST_COMPARISON,
        us_owned_cost=US_OWNED_COST,
        parameters=US_COST_PARAMETERS,
        api_prices=US_API_TOKEN_PRICES,
        cost_config=US_FULL_CLOUD_CONFIG,
        script="08_code/analyze_us_full_cloud_cost.py",
    output:
        comparison=US_FULL_CLOUD_COMPARISON,
        audit=US_FULL_CLOUD_AUDIT,
        total_comparison=US_LOCAL_CLOUD_TOTAL_COMPARISON,
        country_summary=COUNTRY_PRICE_ENVIRONMENT_SUMMARY,
        findings=US_FULL_CLOUD_FINDINGS,
        done=US_FULL_CLOUD_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --cloud-comparison {input.cloud_comparison} "
        "--token-demand {input.token_demand} --china-full-cloud {input.china_full_cloud} "
        "--us-owned-cost {input.us_owned_cost} --parameters {input.parameters} "
        "--api-prices {input.api_prices} --config {input.cost_config} "
        "--output-dir {US_FULL_CLOUD_ROOT}"


rule analyze_single_industry_heterogeneous_hardware:
    input:
        local_summary=MODEL_OUTPUT_ROOT + "/C36/" + CORE_ARCHITECTURE + "/summary.csv",
        token_demand=API_TOKEN_COST_TASK_DETAIL,
        defaults="config/defaults.yaml",
        run_config=RUN_CONFIG,
        screen_config=CPU_GPU_ROUTING_CONFIG,
        deployment_config=DEPLOYMENT_CONFIG,
        routing_research="02_data/manufacturing_ai_cpu_gpu_hardware_routing_research.md",
        cost_parameters=ENTERPRISE_AI_COST_PARAMETERS,
        script="08_code/run_single_industry_heterogeneous_hardware_screen.py",
    output:
        comparison=SINGLE_INDUSTRY_HETEROGENEOUS_COMPARISON,
        routing=SINGLE_INDUSTRY_HETEROGENEOUS_ROUTING,
        findings=SINGLE_INDUSTRY_HETEROGENEOUS_FINDINGS,
        done=SINGLE_INDUSTRY_HETEROGENEOUS_DONE,
    params:
        output_dir=lambda wildcards, output: os.path.dirname(output.comparison),
    log:
        SINGLE_INDUSTRY_HETEROGENEOUS_ROOT + "/workflow.log",
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --defaults {input.defaults} --run-config {input.run_config} "
        "--screen-config {input.screen_config} --token-demand {input.token_demand} "
        "--local-summary {input.local_summary} --output-dir {params.output_dir} "
        "> {log} 2>&1"


rule analyze_single_industry_heterogeneous_hardware_us_cost:
    input:
        china_screen=SINGLE_INDUSTRY_HETEROGENEOUS_COMPARISON,
        token_demand=API_TOKEN_COST_TASK_DETAIL,
        screen_config=CPU_GPU_ROUTING_CONFIG,
        us_parameters=US_COST_PARAMETERS,
        us_api_prices=US_API_TOKEN_PRICES,
        script="08_code/run_single_industry_heterogeneous_hardware_us_cost.py",
    output:
        comparison=SINGLE_INDUSTRY_HETEROGENEOUS_US_COMPARISON,
        cpu_sensitivity=SINGLE_INDUSTRY_HETEROGENEOUS_US_CPU_SENSITIVITY,
        findings=SINGLE_INDUSTRY_HETEROGENEOUS_US_FINDINGS,
        done=SINGLE_INDUSTRY_HETEROGENEOUS_US_DONE,
    params:
        output_dir=lambda wildcards, output: os.path.dirname(output.comparison),
    log:
        SINGLE_INDUSTRY_HETEROGENEOUS_US_ROOT + "/workflow.log",
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --screen-config {input.screen_config} "
        "--china-screen {input.china_screen} --token-demand {input.token_demand} "
        "--us-parameters {input.us_parameters} --us-api-prices {input.us_api_prices} "
        "--output-dir {params.output_dir} > {log} 2>&1"


rule analyze_china_industry_heterogeneous_hardware:
    input:
        local_summary=MODEL_OUTPUT_ROOT + "/{industry}/{architecture}/summary.csv",
        local_hourly=MODEL_OUTPUT_ROOT + "/{industry}/{architecture}/hourly.csv",
        token_demand=API_TOKEN_COST_TASK_DETAIL,
        defaults="config/defaults.yaml",
        run_config=RUN_CONFIG,
        screen_config=CPU_GPU_ROUTING_CONFIG,
        deployment_config=DEPLOYMENT_CONFIG,
        routing_research="02_data/manufacturing_ai_cpu_gpu_hardware_routing_research.md",
        cost_parameters=ENTERPRISE_AI_COST_PARAMETERS,
        script="08_code/run_single_industry_heterogeneous_hardware_screen.py",
    output:
        comparison=CHINA_HETEROGENEOUS_ROOT + "/{architecture}/{industry}/comparison.csv",
        routing=CHINA_HETEROGENEOUS_ROOT + "/{architecture}/{industry}/routing_parameters.csv",
        findings=CHINA_HETEROGENEOUS_ROOT + "/{architecture}/{industry}/findings.md",
        done=CHINA_HETEROGENEOUS_ROOT + "/{architecture}/{industry}/validated.done.json",
    params:
        output_dir=lambda wildcards, output: os.path.dirname(output.comparison),
    log:
        CHINA_HETEROGENEOUS_ROOT + "/{architecture}/{industry}/workflow.log",
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --industry {wildcards.industry} --owned-architecture {wildcards.architecture} --defaults {input.defaults} "
        "--run-config {input.run_config} --screen-config {input.screen_config} "
        "--token-demand {input.token_demand} --local-summary {input.local_summary} "
        "--output-dir {params.output_dir} > {log} 2>&1"


rule summarize_china_heterogeneous_hardware:
    input:
        comparisons=CHINA_HETEROGENEOUS_COMPARISONS,
        done=CHINA_HETEROGENEOUS_DONE,
        routing_config=CPU_GPU_ROUTING_CONFIG,
        deployment_config=DEPLOYMENT_CONFIG,
        script="08_code/summarize_china_heterogeneous_hardware.py",
    output:
        comparison=CHINA_HETEROGENEOUS_NATIONAL,
        findings=CHINA_HETEROGENEOUS_NATIONAL_FINDINGS,
        done=CHINA_HETEROGENEOUS_NATIONAL_DONE,
    log:
        HETEROGENEOUS_ROOT + "/china_national/workflow.log",
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --inputs {input.comparisons} --routing-config {input.routing_config} "
        "--output {output.comparison} "
        "--findings-output {output.findings} --done-output {output.done} > {log} 2>&1"


rule analyze_us_industry_heterogeneous_hardware:
    input:
        demand=US_DEMAND_SERVICE,
        national_task_summary=US_DEMAND_TASK_SUMMARY,
        demand_done=US_DEMAND_DONE,
        demand_config=US_DEMAND_CONFIG,
        routing_config=CPU_GPU_ROUTING_CONFIG,
        us_cost_config=US_HETEROGENEOUS_COST_CONFIG,
        us_parameters=US_COST_PARAMETERS,
        us_api_prices=US_API_TOKEN_PRICES,
        compute_efficiency=MODEL_READY_COMPUTE_EFFICIENCY,
        script="08_code/analyze_us_industry_heterogeneous_hardware.py",
    output:
        detail=US_HETEROGENEOUS_DETAIL,
        national=US_HETEROGENEOUS_NATIONAL,
        findings=US_HETEROGENEOUS_FINDINGS,
        done=US_HETEROGENEOUS_DONE,
    params:
        output_dir=lambda wildcards, output: os.path.dirname(output.detail),
    log:
        US_HETEROGENEOUS_ROOT + "/workflow.log",
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --demand {input.demand} --national-task-summary {input.national_task_summary} "
        "--demand-config {input.demand_config} --routing-config {input.routing_config} --us-cost-config {input.us_cost_config} "
        "--us-parameters {input.us_parameters} --us-api-prices {input.us_api_prices} "
        "--compute-efficiency {input.compute_efficiency} "
        "--output-dir {params.output_dir} > {log} 2>&1"


rule combine_heterogeneous_country_summary:
    input:
        china=CHINA_HETEROGENEOUS_NATIONAL,
        us=US_HETEROGENEOUS_NATIONAL,
        script="08_code/combine_heterogeneous_country_summary.py",
    output:
        comparison=HETEROGENEOUS_COUNTRY_SUMMARY,
        findings=HETEROGENEOUS_FINDINGS,
        done=HETEROGENEOUS_DONE,
    log:
        HETEROGENEOUS_ROOT + "/workflow.log",
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --china {input.china} --us {input.us} "
        "--output {output.comparison} --findings-output {output.findings} "
        "--done-output {output.done} > {log} 2>&1"


rule build_us_manufacturing_ai_demand:
    input:
        config=US_DEMAND_CONFIG,
        activity="02_data/raw/curated/us_manufacturing_activity_naics3_2022.csv",
        mops="02_data/raw/curated/us_manufacturing_ai_adoption_mops_2021.csv",
        btos="02_data/raw/curated/us_business_ai_functions_btos_2026.csv",
        mecs="02_data/raw/curated/us_manufacturing_mecs_2022.csv",
        berd="02_data/raw/curated/us_manufacturing_berd_2023.csv",
        task_parameters="02_data/processed/us_demand/us_task_driver_parameters_v0.1.csv",
        efficiency=MODEL_READY_COMPUTE_EFFICIENCY,
        us_cost_parameters=US_COST_PARAMETERS,
        us_api_prices=US_API_TOKEN_PRICES,
        script="08_code/build_us_manufacturing_ai_demand.py",
    output:
        service=US_DEMAND_SERVICE,
        lineage=US_DEMAND_LINEAGE,
        task_summary=US_DEMAND_TASK_SUMMARY,
        naics_summary=US_DEMAND_NAICS_SUMMARY,
        validation=US_DEMAND_VALIDATION,
        macro_alignment=US_DEMAND_MACRO_ALIGNMENT,
        parameter_audit=US_DEMAND_PARAMETER_AUDIT,
        cost_sensitivity=US_DEMAND_COST_SENSITIVITY,
        local_cost=US_DEMAND_LOCAL_COST,
        cloud_cost=US_DEMAND_CLOUD_COST,
        comparison=US_DEMAND_COMPARISON,
        findings=US_DEMAND_FINDINGS,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --config {input.config} --output-dir {US_DEMAND_ROOT}"


rule validate_us_manufacturing_ai_demand:
    input:
        config=US_DEMAND_CONFIG,
        service=US_DEMAND_SERVICE,
        task_summary=US_DEMAND_TASK_SUMMARY,
        validation=US_DEMAND_VALIDATION,
        local_cost=US_DEMAND_LOCAL_COST,
        cloud_cost=US_DEMAND_CLOUD_COST,
        script="08_code/validate_us_manufacturing_ai_demand.py",
    output:
        done=US_DEMAND_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --config {input.config} --service {input.service} --task-summary {input.task_summary} "
        "--validation {input.validation} --local-cost {input.local_cost} "
        "--cloud-cost {input.cloud_cost} --done-output {output.done}"


rule analyze_load_alignment:
    input:
        summaries=NATIONAL_SUMMARY,
        hourly=expand(
            MODEL_OUTPUT_ROOT + "/{industry}/{scenario}/hourly.csv",
            industry=INDUSTRIES,
            scenario=SCENARIOS,
        ),
        script="08_code/analyze_load_alignment.py",
    output:
        detail=LOAD_ALIGNMENT_DETAIL,
        summary=LOAD_ALIGNMENT_SUMMARY,
        findings=LOAD_ALIGNMENT_FINDINGS,
        done=LOAD_ALIGNMENT_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --summaries {input.summaries} "
        "--hourly-inputs {input.hourly} --model-version {MODEL_VERSION} "
        "--output {output.detail} --summary-output {output.summary} "
        "--findings-output {output.findings} --done-output {output.done}"


rule analyze_national_local_flexibility_ablation:
    input:
        summaries=expand(
            MODEL_OUTPUT_ROOT + "/{industry}/IF/summary.csv",
            industry=INDUSTRIES,
        ),
        hourly=expand(
            MODEL_OUTPUT_ROOT + "/{industry}/IF/hourly.csv",
            industry=INDUSTRIES,
        ),
        baselines=expand(
            MODEL_OUTPUT_ROOT + "/{industry}/baseline/summary.json",
            industry=INDUSTRIES,
        ),
        defaults="config/defaults.yaml",
        run_config=RUN_CONFIG,
        script="08_code/analyze_national_local_flexibility_ablation.py",
    output:
        comparison=FLEXIBILITY_ABLATION_COMPARISON,
        findings=FLEXIBILITY_ABLATION_FINDINGS,
        done=FLEXIBILITY_ABLATION_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --defaults {input.defaults} --config {input.run_config} "
        "--if-summaries {input.summaries} --flex-hourly-inputs {input.hourly} "
        "--baseline-summaries {input.baselines} --output {output.comparison} "
        "--findings-output {output.findings} --done-output {output.done}"


rule analyze_typical_industry_load_stacking:
    input:
        hourly=expand(
            MODEL_OUTPUT_ROOT + "/{industry}/{scenario}/hourly.csv",
            industry=["C14", "C17", "C26", "C36", "C39"],
            scenario=SCENARIOS,
        ),
        script="08_code/analyze_typical_industry_load_stacking.py",
    output:
        profiles=TYPICAL_LOAD_STACKING_PROFILES,
        summary=TYPICAL_LOAD_STACKING_SUMMARY,
        stacked=TYPICAL_LOAD_STACKING_FIGURE,
        ai_only=TYPICAL_AI_LOAD_FIGURE,
        findings=TYPICAL_LOAD_STACKING_FINDINGS,
        done=TYPICAL_LOAD_STACKING_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --hourly-inputs {input.hourly} --model-version {MODEL_VERSION} "
        "--profiles-output {output.profiles} --summary-output {output.summary} "
        "--stacked-figure-output {output.stacked} --ai-figure-output {output.ai_only} "
        "--findings-output {output.findings} --done-output {output.done}"


rule analyze_industry_spot_price_pv_test:
    input:
        common=COMMON_INPUTS,
        price=config["paths"]["spot_price_source"],
        script="08_code/analyze_industry_spot_price_pv_test.py",
    output:
        hourly=SPOT_PV_HOURLY,
        summary=SPOT_PV_SUMMARY,
        figure=SPOT_PV_FIGURE,
        findings=SPOT_PV_FINDINGS,
        done=SPOT_PV_INDUSTRY_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --defaults config/defaults.yaml --config {RUN_CONFIG} "
        "--price-input {input.price} --industry {wildcards.industry} "
        "--hourly-output {output.hourly} "
        "--summary-output {output.summary} --figure-output {output.figure} "
        "--findings-output {output.findings} --done-output {output.done}"


rule summarize_industry_spot_price_pv_tests:
    input:
        summaries=expand(SPOT_PV_SUMMARY, industry=INDUSTRIES),
        done=expand(SPOT_PV_INDUSTRY_DONE, industry=INDUSTRIES),
        script="08_code/summarize_industry_spot_price_pv_tests.py",
    output:
        cases=SPOT_PV_ALL_CASES,
        comparison=SPOT_PV_COMPARISON,
        findings=SPOT_PV_ALL_FINDINGS,
        done=SPOT_PV_ALL_DONE,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --defaults config/defaults.yaml --config {RUN_CONFIG} "
        "--summary-inputs {input.summaries} --cases-output {output.cases} "
        "--comparison-output {output.comparison} --findings-output {output.findings} "
        "--done-output {output.done}"


rule build_bolun_progress_briefing:
    input:
        method=FIGURE1_METHOD_SVG,
        figure1=FIGURE1_SVG,
        figure2=FIGURE2_SVG,
        figure3=FIGURE3_SVG,
        figure4=FIGURE4_SVG,
        figure5=FIGURE5_SVG,
        script="08_code/build_bolun_progress_briefing.py",
    output:
        BOLUN_PROGRESS_BRIEFING,
    conda:
        "../envs/core_model.yaml"
    shell:
        "python {input.script} --model-version {MODEL_VERSION} --method-svg {input.method} --figure1-svg {input.figure1} "
        "--figure2-svg {input.figure2} "
        "--figure3-svg {input.figure3} "
        "--figure4-svg {input.figure4} "
        "--figure5-svg {input.figure5} "
        "--output {output}"


wildcard_constraints:
    industry="C(?:1[3-9]|[234][0-9])",
    scenario="IF|IG|IG_1host|IG_multisite|II_1host|II_multihost",

ENV ?= pypsa
CORES ?= 5
CONFIG ?= config/runs/all_industries_core.yaml
TARGET ?= core
SNAKEMAKE_ARGS ?=

.PHONY: all results dry-run briefing extended-analysis sensitivity-smoke sensitivity-smoke-dry-run sensitivity-grid-hybrid sensitivity-grid-hybrid-dry-run sensitivity-group-multisite sensitivity-group-multisite-dry-run industry-cost-differences national-cloud-center national-grid-comparison national-no-shift-sensitivity national-high-impact-sensitivity national-extended-sensitivity

all: results

# Default workflow: only the 31-industry core scenarios and their validation.
results:
	conda run -n $(ENV) snakemake $(TARGET) --cores $(CORES) \
		--configfile $(CONFIG) \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_snakemake.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

# Inspect the jobs that would run without changing outputs.
dry-run:
	conda run -n $(ENV) snakemake $(TARGET) --cores $(CORES) \
		--configfile $(CONFIG) \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_snakemake_dryrun.XXXXXX)" \
		--dry-run $(SNAKEMAKE_ARGS)

# Figures and HTML are explicit because the briefing currently reads completed
# sensitivity outputs as well as the core results.
briefing:
	conda run -n $(ENV) snakemake build_bolun_progress_briefing --cores $(CORES) \
		--configfile $(CONFIG) \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_briefing.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

# Core post-processing, country comparison, and manuscript figures, excluding
# optional diagnostics and the sensitivity workflow.
extended-analysis:
	conda run -n $(ENV) snakemake extended_analysis --cores $(CORES) \
		--configfile $(CONFIG) \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_extended.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

# Config-selected single-industry physical/grid one-at-a-time screen. Outputs
# remain outside the v0.8.0 mainline and reuse the selected industry's no-AI
# baseline because registered factors
# do not alter the baseline load, tariff, PV, or battery assumptions.
sensitivity-smoke:
	conda run -n $(ENV) snakemake single_industry_sensitivity --cores $(CORES) \
		--configfile config/runs/all_industries_core.yaml \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_sensitivity.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

sensitivity-smoke-dry-run:
	conda run -n $(ENV) snakemake single_industry_sensitivity --cores $(CORES) \
		--configfile config/runs/all_industries_core.yaml \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_sensitivity_dryrun.XXXXXX)" \
		--dry-run $(SNAKEMAKE_ARGS)

sensitivity-grid-hybrid:
	conda run -n $(ENV) snakemake single_industry_grid_hybrid_sensitivity --cores $(CORES) \
		--configfile config/runs/all_industries_core.yaml \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_grid_hybrid.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

sensitivity-grid-hybrid-dry-run:
	conda run -n $(ENV) snakemake single_industry_grid_hybrid_sensitivity --cores $(CORES) \
		--configfile config/runs/all_industries_core.yaml \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_grid_hybrid_dryrun.XXXXXX)" \
		--dry-run $(SNAKEMAKE_ARGS)

sensitivity-group-multisite:
	conda run -n $(ENV) snakemake single_industry_group_multisite_sensitivity --cores $(CORES) \
		--configfile config/runs/all_industries_core.yaml \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_group_multisite.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

sensitivity-group-multisite-dry-run:
	conda run -n $(ENV) snakemake single_industry_group_multisite_sensitivity --cores $(CORES) \
		--configfile config/runs/all_industries_core.yaml \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_group_multisite_dryrun.XXXXXX)" \
		--dry-run $(SNAKEMAKE_ARGS)

# Descriptive 31-industry cost comparison using completed IF, IG and II core
# results only. It changes no sensitivity parameter and does not create new
# optimization scenarios.
industry-cost-differences:
	conda run -n $(ENV) snakemake core_industry_cost_differences --cores $(CORES) \
		--configfile $(CONFIG) \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_industry_cost.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

national-cloud-center:
	conda run -n $(ENV) snakemake national_cloud_center --cores $(CORES) \
		--configfile config/runs/all_industries_core.yaml \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_cloud_center.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

national-grid-comparison:
	conda run -n $(ENV) snakemake national_grid_capacity_comparison --cores $(CORES) \
		--configfile config/runs/all_industries_core.yaml \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_national_compare.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

national-no-shift-sensitivity:
	conda run -n $(ENV) snakemake national_no_shift_sensitivity --cores $(CORES) \
		--configfile config/runs/all_industries_core.yaml \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_no_shift.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

national-high-impact-sensitivity:
	conda run -n $(ENV) snakemake national_high_impact_sensitivity --cores $(CORES) \
		--configfile config/runs/all_industries_core.yaml \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_high_impact.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

national-extended-sensitivity:
	conda run -n $(ENV) snakemake national_extended_sensitivity --cores $(CORES) \
		--configfile config/runs/all_industries_core.yaml \
		--runtime-source-cache-path "$$(mktemp -d /private/tmp/dllm_extended.XXXXXX)" \
		--rerun-incomplete $(SNAKEMAKE_ARGS)

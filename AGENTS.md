# Agent Guidelines for This Fork

This document contains guidance **specific to this fork** that differs from the upstream `eugr/spark-vllm-docker`. For general usage, see `README.md`.

## Fork Identity

| | |
|---|---|
| **Upstream** | `eugr/spark-vllm-docker` |
| **This fork** | `saumen/spark-vllm-docker` |
| **Purpose** | Custom vLLM configurations with modified recipes |

## Git Sync Workflow

```bash
# Fetch upstream changes
git fetch upstream

# Merge into your branch
git pull upstream main

# Push to your fork
git push origin main
```

### All vLLM Flags Go in YAML

The `run-recipe.py` CLI does **not** support most vLLM-specific flags. These **must** be in the YAML `command` section:

| vLLM Flag | YAML Location |
|-----------|---------------|
| `--enable-chunked-prefill` | `command:` section |
| `--async-scheduling` | `command:` section |
| `--kv-cache-metrics` | `command:` section |
| `--language-model-only` | `command:` section |
| `--skip-mm-profiling` | `command:` section |
| `--speculative-config` | `command:` section |
| `solo_only` | Root level (internal flag) |
| `max_num_batched_tokens` | `defaults:` section |

### CLI Overrides (Rarely Needed)

Only use CLI flags for temporary overrides of YAML defaults:

| CLI Flag | YAML Equivalent |
|----------|-----------------|
| `--port`, `--host` | `defaults.port`, `defaults.host` |
| `--tensor-parallel`, `--tp` | `defaults.tensor_parallel` |
| `--gpu-memory-utilization` | `defaults.gpu_memory_utilization` |
| `--max-model-len` | `defaults.max_model_len` |
| `-e VAR=VALUE` | `env` section |

## Merge Conflict Resolution

When pulling upstream changes, conflicts may occur in:

### Recipe YAMLs
- **Keep**: Your custom defaults, vLLM flags, `solo_only`
- **Integrate**: Upstream's new flags, bug fixes, syntax improvements
- **Resolve**: Conflicting default values (choose based on your use case)

### Shell Scripts
- **Keep**: Your wrapper scripts (`run-*.sh`)
- **Integrate**: Upstream's new script features/flags

### Dockerfile / Build Scripts
- **Integrate**: Upstream's security fixes, dependency updates
- **Keep**: Your customizations if any

## Agent Decision Rules

1. **Adding vLLM flags?** → Edit YAML `command` section only
2. **Changing defaults?** → Edit YAML `defaults` section only
3. **Creating shell script?** → Keep it minimal: `./run-recipe.sh <recipe> --solo`
4. **Upstream adds new flag?** → Add to YAML `command` section
5. **Merge conflict in recipe?** → Keep your vLLM flags, integrate upstream structure

## Verification Before Suggesting Changes

- Check `run-recipe.py` for CLI argument definitions
- Verify recipe syntax: `python3 -c "import yaml; yaml.safe_load(open('recipes/<name>.yaml'))"`
- Test with `--dry-run` before actual execution

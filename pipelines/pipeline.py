from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from parseq_core.scenario import Scenario

try:
    from .langauges import LANG_NAME_MAP
except ImportError:
    from langauges import LANG_NAME_MAP

try:
    from .pipeline_vllm import (
        VLLMGenerationContext,
        VLLMRuntimeConfig,
        render_vllm_generation_lines,
        write_vllm_runtime_model_config,
    )
except ImportError:
    from pipeline_vllm import (  # type: ignore[no-redef]
        VLLMGenerationContext,
        VLLMRuntimeConfig,
        render_vllm_generation_lines,
        write_vllm_runtime_model_config,
    )


@dataclass
class EvalConfig:
    tool_dir: str = "evaluation/mt"
    tool_python: str = ".venv/bin/python"
    config_path: Optional[str] = "evaluation/mt/metrics.example.json"
    metrics: list[str] = field(default_factory=lambda: ["comet_kiwi_qe"])
    batch_size: int = 8


@dataclass
class SlurmConfig:
    partition: str = "gpu"
    time: str = "4:00:00"
    gres: str = "gpu:nvidia_rtx_a6000:1"
    cpus_per_task: int = 2
    mem: str = "64G"
    exclude: Optional[str] = None
    additional_args: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    scenario: str = "configs/mt/mt_multi_turn_please_translate_again.yaml"
    tgt: Optional[str] = "de"
    model_config: Optional[str] = None
    model: Optional[str] = None
    prompts_path: Optional[str] = "data/mt_demo_en_de.jsonl"
    prompt_key: Optional[str] = "source"
    extra_data_fields: list[str] = field(default_factory=lambda: ["source", "target"])
    num_generations: Optional[int] = None
    max_tokens: Optional[int] = None
    limit_prompts: Optional[int] = None
    output_root: str = "outputs/mt/pipeline_runs"
    run_name: Optional[str] = None
    scenario_overrides: list[str] = field(default_factory=list)
    core_python: str = ".venv/bin/python"
    slurm_port_from_jobid: bool = True
    submit_jobs: bool = True
    eval_dependency: str = "afterok"
    vllm_runtime: VLLMRuntimeConfig = field(default_factory=VLLMRuntimeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    slurm_gen: SlurmConfig = field(default_factory=SlurmConfig)
    slurm_eval: SlurmConfig = field(default_factory=SlurmConfig)


def _load_config(path: Path, overrides: list[str] | None = None) -> PipelineConfig:
    """Load pipeline config from YAML and merge optional dotlist overrides."""
    base = OmegaConf.structured(PipelineConfig)
    user = OmegaConf.load(path)
    if overrides:
        user = OmegaConf.merge(user, OmegaConf.from_dotlist(overrides))
    merged = OmegaConf.merge(base, user)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


class SingleScenarioPipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        """Initialize paths, logging, and a scenario resolved with generation overrides."""
        self.cfg = cfg
        self.root = Path(__file__).resolve().parents[1]
        self._apply_model_config()
        self.scenario_path = self._resolve_from_root(cfg.scenario)
        if not self.scenario_path.exists():
            raise FileNotFoundError(f"Scenario not found: {self.scenario_path}")

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dataset_name = self._dataset_name_for_run()
        language_name = self._language_name_for_run()
        self.run_name = cfg.run_name or f"{ts}_{self.scenario_path.stem}_{dataset_name}_{language_name}"

        output_root = self._resolve_from_root(cfg.output_root)
        self.run_dir = (output_root / self.run_name).resolve()
        self.generation_dir = self.run_dir / "generation"
        self.generation_scenario_dir = self.generation_dir / self.scenario_path.stem
        self.evaluation_dir = self.run_dir / "evaluation"
        self.log_file = self.run_dir / "pipeline.log"
        self.gen_job_script = self.run_dir / "gen_job.sh"
        self.eval_job_script = self.run_dir / "eval_job.sh"
        self.vllm_runtime_model_config = self.run_dir / "vllm_runtime_model_config.yaml"
        self.vllm_generated_command = self.run_dir / "vllm_generated_command.sh"
        self.vllm_log_file = self.run_dir / "vllm_server.log"
        self.vllm_port_meta_file = self.run_dir / "vllm_port.txt"
        self.log = self._setup_logging()
        self.resolved_prompts_path = self._resolve_prompts_path()

        self.base_generation_overrides = self._generation_overrides()
        self.resolved_scenario = Scenario.load(
            self.scenario_path, overrides=self.base_generation_overrides
        )
        self.samples_path = self.resolved_scenario.output_path()
        self.backend = self.resolved_scenario.backend
        self.server_host = self.resolved_scenario.vllm_host
        self.server_port = self.resolved_scenario.vllm_port
        self.resolved_model = self.resolved_scenario.model
        self.vllm_server_args = list(self.resolved_scenario.vllm_server_args or [])

    def _setup_logging(self) -> logging.Logger:
        """Configure per-run file and console loggers."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"single_scenario_pipeline_{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(self.log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def _resolve_from_root(self, path_str: str) -> Path:
        """Return an absolute path against project root without resolving symlinks."""
        path = Path(path_str)
        if not path.is_absolute():
            path = self.root / path
        return path.absolute()

    def _resolve_command_path_from_root(self, path_or_cmd: str) -> str:
        """Return an absolute command path if it exists under project root, else unchanged."""
        path = Path(path_or_cmd)
        if path.is_absolute():
            return str(path)
        candidate = (self.root / path).absolute()
        if candidate.exists():
            return str(candidate)
        return path_or_cmd

    def _apply_model_config(self) -> None:
        """Load optional model config YAML and apply its defaults to pipeline config."""
        if not self.cfg.model_config:
            return

        model_cfg_path = self._resolve_from_root(self.cfg.model_config)
        if not model_cfg_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_cfg_path}")

        model_cfg = OmegaConf.to_container(OmegaConf.load(model_cfg_path), resolve=True)
        if not isinstance(model_cfg, dict):
            raise ValueError(f"Model config must be a mapping: {model_cfg_path}")

        if not self.cfg.model and model_cfg.get("model"):
            self.cfg.model = str(model_cfg["model"])

        runtime_cfg = model_cfg.get("vllm_runtime_config")
        if runtime_cfg is not None:
            if not isinstance(runtime_cfg, str) or not runtime_cfg.strip():
                raise ValueError(
                    f"Model config key 'vllm_runtime_config' must be a non-empty string: {model_cfg_path}"
                )
            self.cfg.vllm_runtime.runtime_config = runtime_cfg.strip()

        vllm_args = model_cfg.get("vllm_server_args")
        if vllm_args is not None:
            if not isinstance(vllm_args, list):
                raise ValueError(
                    f"Model config key 'vllm_server_args' must be a list: {model_cfg_path}"
                )
            overrides = list(self.cfg.scenario_overrides or [])
            if not any(str(ov).startswith("vllm_server_args=") for ov in overrides):
                serialized_args = json.dumps([str(a) for a in vllm_args], separators=(",", ":"))
                overrides.append(f"vllm_server_args={serialized_args}")
            self.cfg.scenario_overrides = overrides

        model_scenario_overrides: list[str] = []
        raw_model_override = model_cfg.get("scenario_override")
        if raw_model_override is not None:
            model_scenario_overrides.extend(
                self._normalize_model_scenario_override_mapping(
                    raw_model_override, "scenario_override", model_cfg_path
                )
            )

        # Support explicit dotlist entries too, while keeping model-config entries
        # as the final authority.
        raw_model_overrides = model_cfg.get("scenario_overrides")
        if raw_model_overrides is not None:
            if not isinstance(raw_model_overrides, list):
                raise ValueError(
                    f"Model config key 'scenario_overrides' must be a list[str]: {model_cfg_path}"
                )
            for idx, item in enumerate(raw_model_overrides):
                if not isinstance(item, str) or "=" not in item:
                    raise ValueError(
                        f"Model config key 'scenario_overrides[{idx}]' must be a dotlist string like "
                        f"'key=value': {model_cfg_path}"
                    )
                model_scenario_overrides.append(item)

        if model_scenario_overrides:
            self.cfg.scenario_overrides = [
                *(self.cfg.scenario_overrides or []),
                *model_scenario_overrides,
            ]

        ngpus = model_cfg.get("ngpus")
        if ngpus is not None:
            if isinstance(ngpus, bool):
                raise ValueError(f"Model config key 'ngpus' must be an integer >= 1: {model_cfg_path}")
            try:
                ngpus_val = int(ngpus)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Model config key 'ngpus' must be an integer >= 1: {model_cfg_path}"
                ) from exc
            if ngpus_val < 1:
                raise ValueError(f"Model config key 'ngpus' must be >= 1: {model_cfg_path}")
            self.cfg.slurm_gen.gres = self._with_gpu_count(self.cfg.slurm_gen.gres, ngpus_val)

    @staticmethod
    def _with_gpu_count(gres: str | None, ngpus: int) -> str:
        """Return a SLURM `--gres` string with GPU count set to `ngpus`."""
        if not gres:
            return f"gpu:nvidia_rtx_a6000:{ngpus}"
        parts = gres.split(":")
        if not parts or not parts[0].startswith("gpu"):
            return gres
        if len(parts) >= 3:
            parts[-1] = str(ngpus)
        else:
            parts.append(str(ngpus))
        return ":".join(parts)

    @staticmethod
    def _override_value_to_dotlist(value: object, *, model_cfg_path: Path, key: str) -> str:
        """Serialize a value for an OmegaConf dotlist override."""
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (list, dict)):
            return json.dumps(value, separators=(",", ":"))
        raise ValueError(
            f"Unsupported value type for model config override '{key}': "
            f"{type(value).__name__} in {model_cfg_path}"
        )

    @classmethod
    def _normalize_model_scenario_override_mapping(
        cls,
        raw: object,
        field_name: str,
        model_cfg_path: Path,
    ) -> list[str]:
        """Convert model-config scenario override mapping into dotlist strings."""
        if not isinstance(raw, dict):
            raise ValueError(
                f"Model config key '{field_name}' must be a mapping: {model_cfg_path}"
            )
        overrides: list[str] = []
        for key, value in raw.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"Model config key '{field_name}' contains a non-string key: {model_cfg_path}"
                )
            serialized = cls._override_value_to_dotlist(
                value, model_cfg_path=model_cfg_path, key=key
            )
            overrides.append(f"{key}={serialized}")
        return overrides

    @staticmethod
    def _sanitize_name(value: str) -> str:
        """Normalize a path-derived identifier so it is safe for folder names."""
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
        return normalized or "dataset"

    @staticmethod
    def _normalize_language_code(value: str) -> str:
        """Normalize a language code and validate it against `LANG_NAME_MAP`."""
        code = value.strip().replace("-", "_").split("_", 1)[0].lower()
        if code not in LANG_NAME_MAP:
            supported = ", ".join(sorted(LANG_NAME_MAP))
            raise ValueError(f"Unsupported target language '{value}'. Supported: {supported}")
        return code

    def _requested_tgt_code(self) -> str | None:
        """Read a target language code from scenario overrides if provided."""
        for override in reversed(self.cfg.scenario_overrides or []):
            if override.startswith("tgt="):
                return self._normalize_language_code(override.split("=", 1)[1])
        return None

    def _effective_tgt_code(self) -> str | None:
        """Return effective target language code from overrides or config default."""
        override_tgt = self._requested_tgt_code()
        if override_tgt:
            return override_tgt
        if self.cfg.tgt:
            return self._normalize_language_code(self.cfg.tgt)
        return None

    def _resolve_prompts_path(self) -> Path | None:
        """Resolve prompt path; supports either a file or a directory keyed by target language."""
        if not self.cfg.prompts_path:
            return None

        prompts_path = self._resolve_from_root(self.cfg.prompts_path)
        if prompts_path.is_file():
            return prompts_path
        if not prompts_path.exists():
            raise FileNotFoundError(f"prompts_path does not exist: {prompts_path}")
        if not prompts_path.is_dir():
            raise ValueError(f"prompts_path must be a file or directory: {prompts_path}")

        tgt = self._effective_tgt_code()
        if not tgt:
            raise ValueError(
                f"prompts_path is a directory ({prompts_path}) but no target language was provided."
            )

        matches = sorted(prompts_path.glob(f"en-{tgt}*.jsonl"))
        if not matches:
            raise FileNotFoundError(
                f"No prompt file found for target '{tgt}' under {prompts_path} "
                f"(expected pattern: en-{tgt}*.jsonl)."
            )
        if len(matches) > 1:
            exact = prompts_path / f"en-{tgt}.jsonl"
            if exact.exists():
                return exact.resolve()
            raise ValueError(
                f"Multiple prompt files match target '{tgt}' under {prompts_path}: "
                + ", ".join(str(p.name) for p in matches)
                + ". Set prompts_path to an explicit file."
            )
        return matches[0].resolve()

    def _dataset_name_for_run(self) -> str:
        """Infer dataset slug from `prompts_path` for default run folder naming."""
        if not self.cfg.prompts_path:
            return "no_prompts"

        prompts_path = self._resolve_from_root(self.cfg.prompts_path)
        parts = prompts_path.parts

        if "processed" in parts:
            idx = parts.index("processed")
            if idx + 1 < len(parts):
                return self._sanitize_name(parts[idx + 1])

        if "data" in parts:
            idx = parts.index("data")
            if idx + 1 < len(parts):
                candidate = parts[idx + 1]
                if Path(candidate).suffix:
                    return self._sanitize_name(prompts_path.stem)
                return self._sanitize_name(candidate)

        if prompts_path.parent != prompts_path:
            return self._sanitize_name(prompts_path.parent.name)
        return self._sanitize_name(prompts_path.stem)

    def _language_name_for_run(self) -> str:
        """Infer language pair slug from `prompts_path` for default run folder naming."""
        requested_tgt = self._effective_tgt_code()
        if requested_tgt:
            return f"en-{requested_tgt}"

        if not self.cfg.prompts_path:
            return "unknown_lp"

        prompts_path = self._resolve_from_root(self.cfg.prompts_path)
        stem = prompts_path.stem

        explicit_lp = re.search(
            r"([A-Za-z]{2,3}(?:_[A-Za-z]{2,3})?)[-_]([A-Za-z]{2,3}(?:_[A-Za-z]{2,3})?)$",
            stem,
        )
        if explicit_lp:
            src = explicit_lp.group(1)
            tgt = explicit_lp.group(2)
            return self._sanitize_name(f"{src}-{tgt}")

        tokens = re.findall(r"[A-Za-z]{2,3}(?:_[A-Za-z]{2,3})?", stem)
        if len(tokens) >= 2:
            return self._sanitize_name(f"{tokens[-2]}-{tokens[-1]}")
        return "unknown_lp"

    def _run_command(self, cmd: list[str], stage: str) -> subprocess.CompletedProcess[str]:
        """Run a subprocess command, logging stdout/stderr and raising on failure."""
        cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
        self.log.info("(%s) %s", stage, cmd_str)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            self.log.debug("(%s) stdout:\n%s", stage, result.stdout.strip())
        if result.stderr:
            self.log.debug("(%s) stderr:\n%s", stage, result.stderr.strip())
        if result.returncode != 0:
            raise RuntimeError(
                f"{stage} failed with code {result.returncode}: {result.stderr.strip()}"
            )
        return result

    def _generation_overrides(
        self,
        *,
        vllm_host: str | None = None,
        vllm_port_expr: str | None = None,
    ) -> list[str]:
        """Build scenario overrides used for the generation stage."""
        overrides: list[str] = [
            f"name={self.scenario_path.stem}",
            f"output_dir={self.generation_scenario_dir}",
            f"dump_path={self.generation_scenario_dir / 'scenario_resolved.yaml'}",
        ]
        if self.cfg.model:
            overrides.append(f"model={self.cfg.model}")
        if self.resolved_prompts_path:
            overrides.append(f"prompts_path={self.resolved_prompts_path}")
        if self.cfg.prompt_key:
            overrides.append(f"prompt_key={self.cfg.prompt_key}")
        if self.cfg.extra_data_fields:
            overrides.append(
                f"extra_data_fields={json.dumps(self.cfg.extra_data_fields, separators=(',', ':'))}"
            )
        if self.cfg.num_generations is not None:
            overrides.append(f"num_generations={self.cfg.num_generations}")
        if self.cfg.max_tokens is not None:
            overrides.append(f"max_tokens={self.cfg.max_tokens}")
        if self.cfg.limit_prompts is not None:
            overrides.append(f"limit_prompts={self.cfg.limit_prompts}")
        if vllm_host is not None and vllm_port_expr is not None:
            overrides.extend(
                [
                    f"vllm_host={vllm_host}",
                    f"vllm_port={vllm_port_expr}",
                    f"vllm_base_url=http://{vllm_host}:{vllm_port_expr}/v1",
                ]
            )
        overrides.extend(self.cfg.scenario_overrides or [])
        requested_tgt = self._effective_tgt_code()
        if requested_tgt:
            if self._requested_tgt_code() is None:
                overrides.append(f"tgt={requested_tgt}")
            overrides.extend(
                [
                    "src=en",
                    f"src_lang_name={LANG_NAME_MAP['en']}",
                    f"tgt_lang_name={LANG_NAME_MAP[requested_tgt]}",
                ]
            )
        return overrides

    @staticmethod
    def _shell_quote_arg(arg: str) -> str:
        """Shell-quote an argument while preserving `${...}` variable expansion."""
        # Preserve environment variable expansion for args containing ${...}.
        if "${" in arg:
            return f"\"{arg}\""
        return shlex.quote(arg)

    def _render_gen_command(self, *, vllm_port_expr: str | None = None) -> str:
        """Render the parseq generation command as a single shell-safe string."""
        core_python = self._resolve_from_root(self.cfg.core_python)
        if not core_python.exists():
            raise FileNotFoundError(f"Core python not found: {core_python}")
        overrides = (
            self._generation_overrides(vllm_host=self.server_host, vllm_port_expr=vllm_port_expr)
            if vllm_port_expr is not None
            else self.base_generation_overrides
        )
        cmd = [
            str(core_python),
            "-m",
            "parseq_core.main",
            "--scenario",
            str(self.scenario_path),
            *overrides,
        ]
        return " ".join(self._shell_quote_arg(str(part)) for part in cmd)

    def _eval_command(self, samples_path: Path) -> list[str]:
        """Build the evaluation CLI command for a generated samples file."""
        eval_tool_dir = self._resolve_from_root(self.cfg.eval.tool_dir)
        eval_python = Path(self.cfg.eval.tool_python)
        if not eval_python.is_absolute():
            eval_python = (eval_tool_dir / eval_python).absolute()
        if not eval_python.exists():
            raise FileNotFoundError(
                f"Evaluation python not found: {eval_python}. "
                f"Create/install evaluator env under {eval_tool_dir} first."
            )

        cmd: list[str] = [
            str(eval_python),
            "-m",
            "mt_evaluation.cli",
            "--file",
            str(samples_path),
            "--output-dir",
            str(self.evaluation_dir),
        ]

        if self.cfg.eval.config_path:
            eval_config_path = self._resolve_from_root(self.cfg.eval.config_path)
            cmd.extend(["--config", str(eval_config_path)])
        else:
            metrics = [m for m in self.cfg.eval.metrics if str(m).strip()]
            if not metrics:
                metrics = ["comet_kiwi_qe"]
            for metric in metrics:
                cmd.extend(["--metric", metric])
            cmd.extend(["--batch-size", str(self.cfg.eval.batch_size)])
        return cmd

    def _render_slurm_header(self, stage: str, slurm: SlurmConfig, stdout: Path, stderr: Path) -> list[str]:
        """Render common SLURM directives for a stage job script."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=parseq-{stage}-{self.run_name}",
            f"#SBATCH --partition={slurm.partition}",
            f"#SBATCH --time={slurm.time}",
            f"#SBATCH --cpus-per-task={slurm.cpus_per_task}",
            f"#SBATCH --mem={slurm.mem}",
            f"#SBATCH --output={stdout}",
            f"#SBATCH --error={stderr}",
        ]
        if slurm.gres:
            lines.append(f"#SBATCH --gres={slurm.gres}")
        if slurm.exclude:
            lines.append(f"#SBATCH --exclude={slurm.exclude}")
        if slurm.additional_args:
            lines.extend(slurm.additional_args)
        return lines

    def _write_gen_job_script(self) -> Path:
        """Write the generation SLURM job script (with vLLM startup for vLLM backend)."""
        script = self.gen_job_script
        stdout = self.run_dir / "slurm_gen.out"
        stderr = self.run_dir / "slurm_gen.err"
        core_src = (self.root / "src").resolve()
        lines = self._render_slurm_header("gen", self.cfg.slurm_gen, stdout, stderr)
        lines.extend(
            [
                "",
                "set -euo pipefail",
                f"cd {shlex.quote(str(self.root))}",
                f"export PYTHONPATH={shlex.quote(str(core_src))}${{PYTHONPATH:+:${{PYTHONPATH}}}}",
                f"PIPELINE_LOG={shlex.quote(str(self.log_file))}",
                "GEN_START_EPOCH=$(date +%s)",
                "glog() {",
                "  local now_ts now_epoch elapsed",
                "  now_ts=$(date '+%Y-%m-%d %H:%M:%S')",
                "  now_epoch=$(date +%s)",
                "  elapsed=$((now_epoch - GEN_START_EPOCH))",
                "  echo \"[gen][time=${now_ts}][+${elapsed}s] $*\" | tee -a \"$PIPELINE_LOG\"",
                "}",
                "glog \"job started\"",
            ]
        )

        if self.backend != "vllm":
            lines.extend(
                [
                    "glog \"backend is not vllm; running generation directly\"",
                    self._render_gen_command(),
                    "glog \"generation done\"",
                ]
            )
            script.write_text("\n".join(lines) + "\n", encoding="utf-8")
            script.chmod(0o755)
            return script

        vllm_runtime_cfg_path = self._resolve_from_root(self.cfg.vllm_runtime.runtime_config)
        vllm_runner_script_path = self._resolve_from_root(self.cfg.vllm_runtime.runner_script)
        vllm_runner_python = self._resolve_command_path_from_root(self.cfg.vllm_runtime.runner_python)
        if not vllm_runtime_cfg_path.exists():
            raise FileNotFoundError(f"vLLM runtime config not found: {vllm_runtime_cfg_path}")
        if not vllm_runner_script_path.exists():
            raise FileNotFoundError(f"vLLM runner script not found: {vllm_runner_script_path}")
        vllm_model_cfg_path = write_vllm_runtime_model_config(
            path=self.vllm_runtime_model_config,
            model=self.resolved_model,
            vllm_server_args=self.vllm_server_args,
        )
        gen_cmd = self._render_gen_command(vllm_port_expr="${VLLM_PORT}")
        vllm_ctx = VLLMGenerationContext(
            root=self.root,
            run_dir=self.run_dir,
            pipeline_log=self.log_file,
            server_host=self.server_host,
            server_port=self.server_port,
            slurm_port_from_jobid=self.cfg.slurm_port_from_jobid,
            vllm_runtime_config_path=vllm_runtime_cfg_path,
            vllm_runner_script_path=vllm_runner_script_path,
            vllm_runner_python=vllm_runner_python,
            vllm_model_config_path=vllm_model_cfg_path,
            vllm_generated_command_path=self.vllm_generated_command,
            vllm_log_file=self.vllm_log_file,
            vllm_port_meta_file=self.vllm_port_meta_file,
            generation_cmd=gen_cmd,
        )
        lines.extend(render_vllm_generation_lines(vllm_ctx))
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o755)
        return script

    def _write_eval_job_script(self, cmd: list[str], samples_path: Path) -> Path:
        """Write the dependent evaluation SLURM job script to disk."""
        script = self.eval_job_script
        stdout = self.run_dir / "slurm_eval.out"
        stderr = self.run_dir / "slurm_eval.err"
        lines = self._render_slurm_header("eval", self.cfg.slurm_eval, stdout, stderr)
        lines.extend(
            [
                "",
                "set -euo pipefail",
                f"cd {shlex.quote(str(self.root))}",
                f"if [ ! -f {shlex.quote(str(samples_path))} ]; then",
                f"  echo \"[eval] missing samples file: {samples_path}\" >&2",
                "  exit 1",
                "fi",
                f"export PYTHONPATH={shlex.quote(str((self._resolve_from_root(self.cfg.eval.tool_dir) / 'src').resolve()))}"
                "${PYTHONPATH:+:${PYTHONPATH}}",
                "echo \"[eval] start $(date -Is)\"",
                " ".join(shlex.quote(str(p)) for p in cmd),
                "echo \"[eval] done $(date -Is)\"",
            ]
        )
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o755)
        return script

    def _parse_job_id(self, sbatch_stdout: str) -> str:
        """Extract a SLURM job id from `sbatch` stdout."""
        match = re.search(r"Submitted batch job (\d+)", sbatch_stdout)
        if not match:
            raise RuntimeError(f"Could not parse SLURM job id from sbatch output: {sbatch_stdout!r}")
        return match.group(1)

    def _submit_job(self, script_path: Path, stage: str, dependency: str | None = None) -> str:
        """Submit a SLURM script and return its parsed job id."""
        cmd = ["sbatch"]
        if dependency:
            cmd.extend(["--dependency", dependency])
        cmd.append(str(script_path))
        result = self._run_command(cmd, stage=stage)
        return self._parse_job_id(result.stdout)

    def run(self) -> None:
        """Materialize scripts, optionally submit jobs, and log run metadata."""
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

        self.log.info("Pipeline mode: single scenario (SLURM generation -> dependent SLURM evaluation)")
        self.log.info("Scenario: %s", self.scenario_path)
        self.log.info("Run dir: %s", self.run_dir)
        self.log.info("Backend: %s", self.backend)

        samples_path = self.samples_path
        eval_cmd = self._eval_command(samples_path)

        gen_script = self._write_gen_job_script()
        eval_script = self._write_eval_job_script(eval_cmd, samples_path)

        self.log.info("Generated job scripts:")
        self.log.info("  generation: %s", gen_script)
        self.log.info("  evaluation: %s", eval_script)

        if not self.cfg.submit_jobs:
            self.log.info("submit_jobs=false; scripts were written but not submitted.")
            self.log.info("Manual submit:")
            self.log.info("  sbatch %s", gen_script)
            self.log.info(
                "  sbatch --dependency=%s:<GEN_JOB_ID> %s",
                self.cfg.eval_dependency,
                eval_script,
            )
            return

        gen_job_id = self._submit_job(gen_script, stage="submit_gen")
        dependency = f"{self.cfg.eval_dependency}:{gen_job_id}"
        eval_job_id = self._submit_job(eval_script, stage="submit_eval", dependency=dependency)

        self.log.info("Submitted generation job: %s", gen_job_id)
        self.log.info("Submitted evaluation job: %s (dependency: %s)", eval_job_id, dependency)
        self.log.info("Job logs:")
        self.log.info("  %s", self.run_dir / "slurm_gen.out")
        self.log.info("  %s", self.run_dir / "slurm_gen.err")
        self.log.info("  %s", self.run_dir / "slurm_eval.out")
        self.log.info("  %s", self.run_dir / "slurm_eval.err")
        self.log.info("Pipeline controller log: %s", self.log_file)


def main() -> None:
    """CLI entry point for the single-scenario SLURM pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Single-scenario MT pipeline: submit generation SLURM job and dependent evaluation SLURM job."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipelines/configs/wmt24p.yaml"),
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--tgt",
        type=str,
        default=None,
        help="Target language code (source is fixed to English), e.g. ru or zh.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model YAML under configs/model_configs (uses `model:` value as default).",
    )
    args, dotlist = parser.parse_known_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parents[1] / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    cfg = _load_config(config_path, overrides=dotlist)
    if args.tgt:
        cfg.tgt = SingleScenarioPipeline._normalize_language_code(args.tgt)
    if args.model_config:
        cfg.model_config = args.model_config
    pipeline = SingleScenarioPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()

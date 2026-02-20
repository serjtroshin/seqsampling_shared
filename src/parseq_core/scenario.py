from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Prompt(BaseModel):
    id: str
    text: str
    extra_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata copied from prompt JSONL rows.",
    )


class Scenario(BaseModel):
    """Configuration for running a sampling scenario."""

    name: str = "default"
    backend: Literal["vllm", "openai"] = "vllm"
    sampling_mode: Literal["parallel", "iterative", "iterkpar", "enumeration", "multi_turn"] = "parallel"
    model: str
    prompts_path: Path
    output_dir: Path
    output_filename: str = "samples.jsonl"

    # Language settings
    src: str = "en"
    src_lang_name: str = "english"
    tgt: str = "de"
    tgt_lang_name: str = "german"

    # Prompt control
    prompt_template: str | None = None
    system_prompt: str | None = None
    prompt_field_name: str = "solution"
    prompt_example_json: str | None = None
    prompt_key: str | None = None  # optional explicit field for JSONL prompts
    extra_data_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Optional JSONL keys to copy into sampled output records "
            "(e.g. ['target', 'lp', 'domain'])."
        ),
    )
    limit_prompts: int | None = Field(
        default=None,
        ge=1,
        description="If set, only load the first N prompts from the source file.",
    )
    previous_solutions_text: str = (
        "Your solutions must be different from previous solutions: {previous}"
    )
    first_turn_instruction: str | None = None
    reiteration_instruction: str | None = None
    response_instruction: str | None = None
    dump_path: Path | None = None  # optional path to write resolved scenario YAML
    reuse_from: Path | None = None  # optional path to reuse parallel outputs
    constrained_gen: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional seqsampling ConstrainedGenConfig payload. "
            "Example: {'json_mode': true} or {'json_schema': {...}}."
        ),
    )
    enumeration_field_name: str = "solutions"
    enumeration_include_final_answer: bool = False
    enumeration_final_answer_field_name: str = "final_answer"

    # Sampling parameters
    top_k: int = Field(default=5, ge=1)
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    num_generations: int = Field(default=1, ge=1)
    n_parallel: int = Field(default=1, ge=1, description="Parallel generations per iteration (seqsampling GenerationConfig.n)")
    seed: int | None = None
    history_k: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Keep last k generations in history-style filenames "
            "(iterative/iterkpar/enumeration/multi_turn); None keeps all for iterative prompts."
        ),
    )

    # vLLM / OpenAI compatibility knobs
    trust_remote_code: bool = False
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str | None = None
    vllm_host: str = "127.0.0.1"
    vllm_port: int = Field(default=8000, ge=1, le=65535)
    vllm_server_args: list[str] = Field(default_factory=list)
    vllm_extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra JSON body fields passed to OpenAI-compatible vLLM requests.",
    )
    openai_api_key: str | None = None
    request_timeout: float | None = Field(
        default=3600.0,
        gt=0.0,
        description="HTTP request timeout (seconds) for OpenAI-compatible calls.",
    )
    max_concurrent: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional override for concurrent OpenAI-compatible requests. "
            "Defaults to 1 for parallel mode and 16 for iterative/iterkpar/enumeration/multi_turn."
        ),
    )
    # Logprob collection / likelihood scoring
    request_logprobs: bool = Field(
        default=False,
        description="When true, request and store token-level logprobs for generations.",
    )
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description="Number of alternate logprobs to return per token (chat/completions).",
    )
    likelihood_temperature: float | None = Field(
        default=0.0,
        ge=0.0,
        description="Temperature used when scoring likelihoods (defaults to 0 for deterministic scoring).",
    )
    likelihood_top_p: float | None = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Top-p used when scoring likelihoods.",
    )

    @model_validator(mode="after")
    def _expand_paths(self) -> "Scenario":
        # Resolve paths relative to the scenario file location if possible.
        if hasattr(self, "_config_path"):
            base: Path = getattr(self, "_config_path")
            if not self.prompts_path.is_absolute():
                self.prompts_path = (base / self.prompts_path).resolve()
            if not self.output_dir.is_absolute():
                self.output_dir = (base / self.output_dir).resolve()
            if self.dump_path and not self.dump_path.is_absolute():
                self.dump_path = (base / self.dump_path).resolve()
            if self.reuse_from and not self.reuse_from.is_absolute():
                self.reuse_from = (base / self.reuse_from).resolve()
        return self

    # -------- Derived output paths --------
    def resolved_output_dir(self) -> Path:
        """
        If the configured output_dir looks like a base folder (e.g., ../outputs/),
        append the scenario name so each scenario writes to its own subdir.
        Otherwise, honor the explicit path.
        """
        if self.output_dir.name in {"output", "outputs"} or self.output_dir == self.output_dir.parent:
            return (self.output_dir / self.name).resolve()
        if self.output_dir.name == self.name:
            return self.output_dir
        return self.output_dir

    def resolved_output_filename(self) -> str:
        """
        Add history_k to the filename for sequential-style runs so artifacts are easy to tell apart.
        """
        filename = self.output_filename
        if self.is_sequential_mode() and self.history_k:
            path = Path(filename)
            suffix = path.suffix
            stem = path.stem
            tag = f"history{self.history_k}"
            if tag not in stem:
                stem = f"{stem}_{tag}"
            filename = f"{stem}{suffix}"
        return filename

    def output_path(self) -> Path:
        return self.resolved_output_dir() / self.resolved_output_filename()

    @classmethod
    def load(cls, path: Path, overrides: list[str] | None = None) -> "Scenario":
        """
        Load a scenario using Hydra/OmegaConf so users can write YAML and pass CLI overrides.
        """
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "hydra-core (OmegaConf) is required to load scenario configs. "
                "Install with `pip install hydra-core`."
            ) from exc

        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

        data = OmegaConf.to_container(cfg, resolve=True)
        obj = cls(**data)
        obj._config_path = path.parent  # type: ignore[attr-defined]
        obj._resolved_cfg = cfg  # type: ignore[attr-defined]
        obj = obj._expand_paths()
        return obj

    def _maybe_limit_prompts(self, prompts: list[Prompt]) -> list[Prompt]:
        if self.limit_prompts is None:
            return prompts
        return prompts[: self.limit_prompts]

    def load_prompts(self) -> list[Prompt]:
        """Load prompts from a .txt (one per line) or .jsonl file."""
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_path}")

        if self.prompts_path.suffix == ".jsonl":
            prompts: list[Prompt] = []
            for idx, line in enumerate(self.prompts_path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if self.prompt_key:
                    text = payload.get(self.prompt_key)
                else:
                    text = (
                        payload.get("prompt")
                        or payload.get("text")
                        or payload.get("message")
                        or payload.get("source")
                    )
                if text is None:
                    raise ValueError(
                        "Each JSONL line must include a prompt field. "
                        "Checked keys: "
                        f"{[self.prompt_key] if self.prompt_key else ['prompt','text','message','source']}"
                    )
                pid = payload.get("id", str(idx))
                extra_data = {
                    key: payload.get(key)
                    for key in self.extra_data_fields
                } if self.extra_data_fields else {}
                prompts.append(Prompt(id=str(pid), text=str(text), extra_data=extra_data))
            return self._maybe_limit_prompts(prompts)

        # Treat everything else as plain text
        lines = [
            line.strip() for line in self.prompts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prompts = [Prompt(id=str(idx), text=line) for idx, line in enumerate(lines)]
        return self._maybe_limit_prompts(prompts)

    def format_prompt(self, prompt: Prompt) -> str:
        if self.prompt_template:
            return self.prompt_template.format(prompt=prompt.text)
        return prompt.text

    # Convenience flags
    def is_iterative(self) -> bool:
        return self.sampling_mode == "iterative"

    def is_iterkpar(self) -> bool:
        return self.sampling_mode == "iterkpar"

    def is_enumeration(self) -> bool:
        return self.sampling_mode == "enumeration"

    def is_multi_turn(self) -> bool:
        return self.sampling_mode == "multi_turn"

    def is_sequential_mode(self) -> bool:
        return self.sampling_mode in {"iterative", "iterkpar", "enumeration", "multi_turn"}

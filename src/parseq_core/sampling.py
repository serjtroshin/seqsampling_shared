from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Literal
import warnings

from seqsampling.models.base import ChatMessage
from seqsampling.models.base import Generation
from seqsampling.parsing.base import ExtractionParser
from seqsampling.models.openai_client import OpenAIClient
from seqsampling.prompts.schema import PromptContext
from seqsampling.sampling.config import ConstrainedGenConfig, GenerationConfig, SamplingConfig
from seqsampling.sampling.iteration import (
    IterationPromptContext,
    JsonPromptSchema,
    SequentialSamplingResult,
)

from .scenario import Prompt, Scenario

PromptContextLike = PromptContext | IterationPromptContext


class BasePromptSchema(ABC):
    """Shared prompt schema interface for typing."""

    @abstractmethod
    def build_messages(self, ctx: PromptContextLike, turn_id: int = 0) -> List[ChatMessage]:
        raise NotImplementedError


class ConfigurableJsonPromptSchema(JsonPromptSchema, BasePromptSchema):
    """JsonPromptSchema with configurable 'previous solutions' text."""

    previous_solutions_text: str

    def __init__(self, previous_solutions_text: str, **kwargs):
        super().__init__(**kwargs)
        self.previous_solutions_text = previous_solutions_text or (
            "Your solutions must be different from previous solutions: {previous}"
        )

    def build_messages(self, ctx: IterationPromptContext, turn_id: int = 0) -> List[ChatMessage]:
        previous = self.previous_solutions_text.format(previous=str(ctx.previous_solutions))
        return [
            ChatMessage(role="system", content=self.system_instruction),
            ChatMessage(
                role="user",
                content=(
                    "You must output a JSON object with a field "
                    f"'{self.field_name}', which is an array of exactly "
                    f"1 item. Example: {self.example_json}\n\n"
                    f"{previous}\n\n"
                    f"Task input:\n{ctx.input_text}"
                ),
            ),
        ]


@dataclass
class PreviousSolutionsPlainPromptSchema(BasePromptSchema):
    """Prompt schema that asks for a single plain-text output (no JSON)."""

    system_instruction: str
    previous_solutions_text: str | None = None
    response_instruction: str | None = None

    def build_messages(self, ctx: IterationPromptContext, turn_id: int = 0) -> List[ChatMessage]:
        previous = (
            self.previous_solutions_text.format(previous=str(ctx.previous_solutions))
            if self.previous_solutions_text
            else f"Previously sampled outputs: {ctx.previous_solutions}"
        )
        instruction = (
            self.response_instruction
            or "Provide one new response as plain text (no JSON, no lists)."
        )
        return [
            ChatMessage(role="system", content=self.system_instruction),
            ChatMessage(
                role="user",
                content=(
                    f"{instruction}\n"
                    f"{previous}\n\n"
                    f"{ctx.input_text}"
                ),
            ),
        ]


@dataclass
class FirstTurnPlainPromptSchema(BasePromptSchema):
    """Prompt schema that asks for a single plain-text output (no JSON)."""

    system_instruction: str
    first_turn_text: str | None = None
    response_instruction: str | None = None

    def build_messages(self, ctx: PromptContextLike, turn_id: int = 0) -> List[ChatMessage]:
        instruction = (
            self.response_instruction
            or "Provide one new response as plain text (no JSON, no lists)."
        )
        return [
            ChatMessage(role="system", content=self.system_instruction),
            ChatMessage(
                role="user",
                content=(
                    f"{instruction}\n"
                    f"{self.first_turn_text or ''}\n\n"
                    f"{ctx.input_text}"
                ),
            ),
        ]


@dataclass
class ConditionalPromptSchema(BasePromptSchema):
    """Prompt schema that switches between first turn and reiteration turn."""

    first_turn_prompt_schema: FirstTurnPlainPromptSchema
    reiteration_turn_prompt_schema: PreviousSolutionsPlainPromptSchema

    def build_messages(self, ctx: IterationPromptContext, turn_id: int = 0) -> List[ChatMessage]:
        if turn_id == 0:
            return self.first_turn_prompt_schema.build_messages(ctx, turn_id=turn_id)
        return self.reiteration_turn_prompt_schema.build_messages(ctx, turn_id=turn_id)


@dataclass
class PlainParallelPromptSchema(BasePromptSchema):
    """Prompt schema for parallel sampling (no history)."""

    system_instruction: str
    response_instruction: str | None = None
    first_turn_instruction: str | None = None

    def build_messages(self, ctx: PromptContext, turn_id: int = 0) -> List[ChatMessage]:
        return FirstTurnPlainPromptSchema(
            system_instruction=self.system_instruction,
            response_instruction=self.response_instruction,
            first_turn_text=self.first_turn_instruction,
        ).build_messages(ctx, turn_id=turn_id)


class PlainTextParser:
    """Return the raw generation text (trimmed)."""

    def parse(self, text: str) -> str:
        return text.strip()


class RemoveThinkingParser(ExtractionParser):
    """Extract the part of the text after </think> tag, if present."""

    def extract(self, text: str, end_del: str = "</think>") -> str:
        end_tag = end_del.lower()
        lower_text = text.lower()
        end_idx = lower_text.find(end_tag)
        if end_idx != -1:
            return text[end_idx + len(end_tag):].strip()
        return text.strip()


def _reasoning_to_text(value: Any) -> str | None:
    """Best-effort conversion for reasoning payloads returned by some vLLM models."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    chunks.append(item)
            elif isinstance(item, dict):
                candidate = item.get("text")
                if isinstance(candidate, str) and candidate.strip():
                    chunks.append(candidate)
        text = "\n".join(chunks).strip()
        return text if text else None
    if isinstance(value, dict):
        candidate = value.get("text")
        if isinstance(candidate, str):
            text = candidate.strip()
            return text if text else None
    return str(value)


class ReasoningAwareOpenAIClient(OpenAIClient):
    """OpenAI-compatible client with fallback for responses using reasoning fields."""

    @staticmethod
    def _extract_generations(response) -> List[Generation]:
        gens: List[Generation] = []
        for choice in response.choices:
            text = choice.message.content
            if text is None:
                reasoning_content = _reasoning_to_text(
                    getattr(choice.message, "reasoning_content", None)
                )
                reasoning = _reasoning_to_text(getattr(choice.message, "reasoning", None))
                if reasoning_content is not None:
                    warnings.warn(
                        "OpenAI returned a null message content; using message.reasoning_content.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    text = reasoning_content
                elif reasoning is not None:
                    warnings.warn(
                        "OpenAI returned a null message content; using message.reasoning.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    text = reasoning
                else:
                    warnings.warn(
                        "OpenAI returned a null message content with no reasoning fallback; substituting an empty string.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    text = ""
            gens.append(Generation(text=text, raw=choice.model_dump()))
        return gens


def build_seqsampling_client(scenario: Scenario) -> OpenAIClient:
    """
    Use OpenAI-compatible API for both vLLM (via base_url) and OpenAI.
    """
    api_key = scenario.vllm_api_key or scenario.openai_api_key or "token"
    timeout = scenario.request_timeout or 3600.0
    base_url = scenario.vllm_base_url if scenario.backend == "vllm" else None
    # Adapt request concurrency in parallel mode to keep total in-flight
    # generations bounded while avoiding the n=1 throughput collapse.
    target_inflight = max(1, scenario.target_inflight_generations)
    if scenario.sampling_mode == "parallel":
        default_max_concurrent = max(1, target_inflight // max(1, scenario.num_generations))
    else:
        default_max_concurrent = target_inflight
    max_concurrent = (
        scenario.max_concurrent
        if scenario.max_concurrent is not None
        else default_max_concurrent
    )

    return ReasoningAwareOpenAIClient(
        model=scenario.model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_concurrent=max_concurrent,
    )


def build_generation_config(
    scenario: Scenario,
    mode: Literal["parallel", "iterative", "enumeration"],
) -> GenerationConfig:
    n = scenario.num_generations if mode == "parallel" else scenario.n_parallel
    return GenerationConfig(
        n=n,
        max_tokens=scenario.max_tokens,
        temperature=scenario.temperature,
        top_p=scenario.top_p,
    )


def build_sampling_config(
    scenario: Scenario,
    mode: Literal["parallel", "iterative", "enumeration"],
) -> SamplingConfig:
    extra_params = None
    if scenario.backend == "vllm":
        extra_body: dict[str, Any] = {}
        if scenario.top_k is not None:
            extra_body["top_k"] = scenario.top_k
        if scenario.vllm_extra_body:
            extra_body.update(scenario.vllm_extra_body)
        if extra_body:
            extra_params = {"extra_body": extra_body}
    k = 1 if mode == "parallel" else scenario.num_generations
    return SamplingConfig(
        k=k,
        generation_config=build_generation_config(scenario, mode),
        extra_params=extra_params,
    )


def build_constrained_config(
    scenario: Scenario,
    default_json_schema: dict[str, Any] | None = None,
) -> ConstrainedGenConfig | None:
    """
    Build a seqsampling ConstrainedGenConfig from scenario.constrained_gen.
    If default_json_schema is provided, use it unless config already sets a schema/mode.
    """
    config_data = dict(scenario.constrained_gen or {})
    if default_json_schema is not None and "json_schema" not in config_data and not config_data.get("json_mode"):
        config_data["json_schema"] = default_json_schema
        config_data.setdefault("json_schema_strict", True)
    if not config_data:
        return None
    allowed_keys = {f.name for f in fields(ConstrainedGenConfig)}
    unknown = sorted(set(config_data) - allowed_keys)
    if unknown:
        print(
            "[warn] ignoring unknown constrained_gen keys: "
            + ", ".join(unknown),
            flush=True,
        )
    filtered = {k: v for k, v in config_data.items() if k in allowed_keys}
    if not filtered:
        return None
    return ConstrainedGenConfig(**filtered)


def write_samples_jsonl(
    output_path: Path,
    scenario: Scenario,
    prompts: List[Prompt],
    outputs: List[SequentialSamplingResult],
    mode: Literal["parallel", "iterative", "enumeration"],
    created_at: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    with output_path.open("w", encoding="utf-8") as fout:
        for prompt_obj, result in zip(prompts, outputs):
            record = {
                "scenario": scenario.name,
                "backend": scenario.backend,
                "model": scenario.model,
                "prompt_id": prompt_obj.id,
                "prompt": scenario.format_prompt(prompt_obj),
                "created_at": created_at,
                "solutions": result.solutions_per_prompt,
                "raw_generations": result.raw_generations,
                "parallel_ids": result.parallel_ids,
                "sequential_ids": result.sequential_ids,
            }
            if prompt_obj.extra_data:
                for key, value in prompt_obj.extra_data.items():
                    if key not in record:
                        record[key] = value
            if mode == "parallel":
                record["generations"] = result.solutions_per_prompt
            final_answers = getattr(result, "final_answers", None)
            if final_answers is not None:
                record["final_answers"] = final_answers
            parsed_generations = getattr(result, "parsed_generations", None)
            if parsed_generations is not None:
                record["parsed_generations"] = parsed_generations
            fout.write(json_dumps(record) + "\n")


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

from seqsampling.models.base import ChatMessage
from seqsampling.parsing.base import PARSE_FAILED_PLACEHOLDER, ParseError
from seqsampling.prompts.schema import PromptContext
from seqsampling.sampling.enumeration import EnumerationSampler
from seqsampling.sampling.sampler import SequentialSamplingResult

from .scenario import Scenario
from .sampling import (
    BasePromptSchema,
    RemoveThinkingParser,
    build_constrained_config,
    build_sampling_config,
    build_seqsampling_client,
    write_samples_jsonl,
)


@dataclass
class EnumerationPromptSchema(BasePromptSchema):
    system_instruction: str
    first_turn_instruction: str | None = None
    response_instruction: str | None = None
    field_name: str = "solutions"
    include_final_answer: bool = False
    final_answer_field_name: str = "final_answer"

    def build_messages(self, ctx: PromptContext, turn_id: int = 0) -> List[ChatMessage]:
        instruction = (
            self.response_instruction
            or "Provide diverse outputs while preserving meaning."
        )
        task_instruction = self.first_turn_instruction or ""
        if self.include_final_answer:
            example_json = (
                f'{{"{self.field_name}": ["candidate 1", "candidate 2"], '
                f'"{self.final_answer_field_name}": "best final answer"}}'
            )
            output_format = (
                "Return exactly one JSON object and nothing else. "
                f"'{self.field_name}' must contain no more than {ctx.k} strings. "
                f"'{self.final_answer_field_name}' must be a single final string answer."
            )
        else:
            example_json = f'{{"{self.field_name}": ["candidate 1", "candidate 2"]}}'
            output_format = (
                "Return exactly one JSON object and nothing else. "
                f"'{self.field_name}' must contain no more than {ctx.k} strings."
            )

        return [
            ChatMessage(role="system", content=self.system_instruction),
            ChatMessage(
                role="user",
                content=(
                    f"{instruction}\n"
                    f"{task_instruction}\n\n"
                    f"{output_format}\n"
                    f"Example JSON: {example_json}\n\n"
                    f"Task input:\n{ctx.input_text}"
                ),
            ),
        ]


@dataclass
class ParsedEnumerationGeneration:
    solutions: list[str]
    final_answer: str | None = None


class EnumerationJsonParser:
    def __init__(
        self,
        *,
        field_name: str = "solutions",
        expected_k: int | None = None,
        include_final_answer: bool = False,
        final_answer_field_name: str = "final_answer",
    ) -> None:
        self.field_name = field_name
        self.expected_k = expected_k
        self.include_final_answer = include_final_answer
        self.final_answer_field_name = final_answer_field_name

    def parse(self, text: str) -> ParsedEnumerationGeneration:
        import json

        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid JSON: {exc}") from exc

        if not isinstance(obj, dict):
            raise ParseError("Expected JSON object.")
        if self.field_name not in obj:
            raise ParseError(f"Missing key '{self.field_name}'.")

        raw_solutions = obj[self.field_name]
        if not isinstance(raw_solutions, list):
            raise ParseError(f"'{self.field_name}' must be a list.")

        solutions = [str(item).strip() for item in raw_solutions]

        final_answer: str | None = None
        if self.include_final_answer:
            raw_final = obj.get(self.final_answer_field_name)
            if raw_final is None:
                raise ParseError(f"Missing key '{self.final_answer_field_name}'.")
            final_answer = str(raw_final).strip()
            if not final_answer:
                raise ParseError(f"'{self.final_answer_field_name}' must be non-empty.")

        return ParsedEnumerationGeneration(solutions=solutions, final_answer=final_answer)


@dataclass
class EnumerationSamplingResult(SequentialSamplingResult):
    final_answers: list[str] | None = None
    parsed_generations: list[dict[str, Any]] | None = None


@dataclass
class StructuredEnumerationSampler(EnumerationSampler):
    parser: EnumerationJsonParser
    prompt_schema: EnumerationPromptSchema

    def run(self, inputs: List[str]) -> List[SequentialSamplingResult]:  # type: ignore[override]
        print(
            f"Running EnumerationSampler on {len(inputs)} inputs with "
            f"n={self.sampling_config.generation_config.n}, k={self.sampling_config.k}"
        )

        message_batches = [
            self.prompt_schema.build_messages(PromptContext(input_text=inp, k=self.sampling_config.k))
            for inp in inputs
        ]

        extra = self.sampling_config.extra_params or {}
        extra = {**extra, **self._constrained_extra_params()}
        gen_cfg = self.sampling_config.generation_config

        batched_gens = self.client.generate_batch(
            batch_messages=message_batches,
            **asdict(gen_cfg),
            extra_params=extra,
        )

        outputs: list[SequentialSamplingResult] = []
        for gens in batched_gens:
            raw_generations: list[str] = []
            all_solutions: list[Any] = []
            parallel_ids: list[int] = []
            sequential_ids: list[int] = []
            final_answers: list[str] | None = [] if self.parser.include_final_answer else None
            parsed_generations: list[dict[str, Any]] = []

            for par_id, gen in enumerate(gens):
                text = gen.text
                raw_generations.append(text)
                parsed_text = self._extract_answer(text)
                try:
                    parsed = self.parser.parse(parsed_text)
                except ParseError as exc:
                    if self._should_raise_on_parse_failure():
                        raise
                    warnings.warn(f"Parsing failed for generation {par_id}: {exc}")
                    parsed = ParsedEnumerationGeneration(
                        solutions=[PARSE_FAILED_PLACEHOLDER] * self.sampling_config.k,
                        final_answer=PARSE_FAILED_PLACEHOLDER if self.parser.include_final_answer else None,
                    )

                all_solutions.extend(parsed.solutions)
                parallel_ids.extend([par_id] * len(parsed.solutions))
                sequential_ids.extend(range(len(parsed.solutions)))

                parsed_payload: dict[str, Any] = {self.parser.field_name: parsed.solutions}
                if self.parser.include_final_answer:
                    assert final_answers is not None
                    parsed_payload[self.parser.final_answer_field_name] = parsed.final_answer
                    final_answers.append(parsed.final_answer or PARSE_FAILED_PLACEHOLDER)
                parsed_generations.append(parsed_payload)

            outputs.append(
                EnumerationSamplingResult(
                    solutions_per_prompt=all_solutions,
                    raw_generations=raw_generations,
                    parallel_ids=parallel_ids,
                    sequential_ids=sequential_ids,
                    final_answers=final_answers,
                    parsed_generations=parsed_generations,
                )
            )

        return outputs


def build_enumeration_json_schema(scenario: Scenario) -> dict[str, Any]:
    field_name = scenario.enumeration_field_name
    final_key = scenario.enumeration_final_answer_field_name
    properties: dict[str, Any] = {
        field_name: {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": scenario.num_generations,
        }
    }
    required = [field_name]
    if scenario.enumeration_include_final_answer:
        properties[final_key] = {"type": "string"}
        required.append(final_key)
    return {
        "title": f"{scenario.name}_enumeration_response",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_enumeration_prompt_schema(scenario: Scenario) -> EnumerationPromptSchema:
    return EnumerationPromptSchema(
        system_instruction=scenario.system_prompt or "You are a helpful assistant.",
        first_turn_instruction=scenario.first_turn_instruction,
        response_instruction=scenario.response_instruction
        or "Produce multiple diverse outputs in one response.",
        field_name=scenario.enumeration_field_name,
        include_final_answer=scenario.enumeration_include_final_answer,
        final_answer_field_name=scenario.enumeration_final_answer_field_name,
    )


def build_enumeration_sampler(scenario: Scenario) -> StructuredEnumerationSampler:
    return StructuredEnumerationSampler(
        client=build_seqsampling_client(scenario),
        sampling_config=build_sampling_config(scenario, "enumeration"),
        constrained_config=build_constrained_config(
            scenario,
            default_json_schema=build_enumeration_json_schema(scenario),
        ),
        prompt_schema=build_enumeration_prompt_schema(scenario),
        parser=EnumerationJsonParser(
            field_name=scenario.enumeration_field_name,
            expected_k=None,
            include_final_answer=scenario.enumeration_include_final_answer,
            final_answer_field_name=scenario.enumeration_final_answer_field_name,
        ),
        string_extractor=RemoveThinkingParser(),
    )


def run_enumeration_scenario(
    scenario_path: Path,
    overrides: list[str] | None = None,
    draft_prompt: bool = False,
) -> Path:
    scenario = Scenario.load(scenario_path, overrides=overrides)
    _maybe_dump_resolved_config(scenario)
    prompts = scenario.load_prompts()
    formatted_prompts = [scenario.format_prompt(p) for p in prompts]

    if draft_prompt:
        if not formatted_prompts:
            print("[draft] no prompts to render.")
            raise SystemExit(0)
        schema = build_enumeration_prompt_schema(scenario)
        msg_objs = schema.build_messages(
            PromptContext(input_text=formatted_prompts[0], k=scenario.num_generations),
            turn_id=0,
        )
        print("[draft] enumeration:")
        for msg in msg_objs:
            print(f"--- {msg.role} ---\n{msg.content}\n")
        raise SystemExit(0)

    sampler = build_enumeration_sampler(scenario)
    outputs = sampler.run(formatted_prompts)
    output_path = scenario.output_path()
    write_samples_jsonl(output_path, scenario, prompts, outputs, mode="enumeration")
    return output_path


def _maybe_dump_resolved_config(scenario: Scenario) -> None:
    if not scenario.dump_path:
        return
    cfg = getattr(scenario, "_resolved_cfg", None)
    scenario.dump_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        from omegaconf import OmegaConf

        scenario.dump_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    else:
        import json

        scenario.dump_path.write_text(
            json.dumps(scenario.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )


__all__ = [
    "EnumerationPromptSchema",
    "EnumerationJsonParser",
    "StructuredEnumerationSampler",
    "build_enumeration_sampler",
    "run_enumeration_scenario",
]

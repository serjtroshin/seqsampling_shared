from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

from seqsampling.sampling.iteration import (
    IterationPromptContext,
    IterationSampler,
    SequentialSamplingResult,
)

from .scenario import Scenario
from .sampling import (
    BasePromptSchema,
    ConditionalPromptSchema,
    FirstTurnPlainPromptSchema,
    PreviousSolutionsPlainPromptSchema,
    PlainTextParser,
    RemoveThinkingParser,
    build_sampling_config,
    build_seqsampling_client,
    write_samples_jsonl,
)


@dataclass
class History:
    """Keeps the last k generations (or all if k is None)."""

    k: int | None = None
    items: list[str] = None

    def __post_init__(self) -> None:
        if self.items is None:
            self.items = []

    def add(self, text: str) -> None:
        self.items.append(text)
        if self.k is not None and len(self.items) > self.k:
            self.items = self.items[-self.k :]

    def last(self) -> list[str]:
        return list(self.items)


@dataclass
class HistoryIterationSampler(IterationSampler):
    """Iteration sampler that respects a bounded history window."""

    history_k: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.prompt_schema is None:
            raise ValueError("prompt_schema must be provided to HistoryIterationSampler")

    def run(self, inputs: List[str], draft_prompt: bool = False) -> List[SequentialSamplingResult]:  # type: ignore[override]
        histories = [History(self.history_k) for _ in inputs]
        raw_generations: List[List[str]] = [[] for _ in inputs]
        all_solutions: List[List[Any]] = [[] for _ in inputs]
        parallel_ids: List[List[int]] = [[] for _ in inputs]
        sequential_ids: List[List[int]] = [[] for _ in inputs]

        for turn_id in range(self.sampling_config.k):
            message_batches = [
                self.prompt_schema.build_messages(
                    IterationPromptContext(
                        input_text=inp,
                        k=self.sampling_config.k,
                        previous_solutions=histories[i].last(),
                    ),
                    turn_id=turn_id,
                )
                for i, inp in enumerate(inputs)
            ]
            extra = self.sampling_config.extra_params or {}
            extra = {**extra, **self._constrained_extra_params()}

            if draft_prompt:
                for input_id, messages in enumerate(message_batches):
                    print(f"[draft] turn {turn_id}, prompt {input_id} messages:")
                    for msg in messages:
                        role = msg.role
                        content = msg.content
                        print(f"--- {role} ---\n{content}\n")
                if turn_id > 1:
                    raise SystemExit(0)
                continue

            try:
                batched_gens = self.client.generate_batch(
                    batch_messages=message_batches,
                    **asdict(self.sampling_config.generation_config),
                    extra_params=extra,
                )
            except Exception as exc:  # noqa: BLE001
                import os

                endpoint = getattr(self, "endpoint", "unknown")
                model_name = getattr(self, "model_name", "unknown")
                port = os.getenv("VLLM_PORT") or "unset"
                host = os.getenv("VLLM_HOST") or "127.0.0.1"
                raise RuntimeError(
                    f"Generation failed (turn={turn_id}, prompts={len(message_batches)}) "
                    f"endpoint={endpoint}, host={host}, port={port}, model={model_name}"
                ) from exc

            for input_id, gens in enumerate(batched_gens):
                for i, gen in enumerate(gens):
                    text = gen.text
                    raw_generations[input_id].append(text)
                    parsed = self._extract_answer(text)
                    try:
                        new_solution = self.parser.parse(parsed)
                    except Exception:
                        new_solution = parsed  # keep raw parsed content if parser fails
                    histories[input_id].add(parsed)
                    all_solutions[input_id].append(new_solution)
                    parallel_ids[input_id].append(i)
                    sequential_ids[input_id].append(turn_id)

        if draft_prompt:
            raise SystemExit(0)

        outputs: List[SequentialSamplingResult] = []
        for input_id in range(len(inputs)):
            outputs.append(
                SequentialSamplingResult(
                    solutions_per_prompt=all_solutions[input_id],
                    raw_generations=raw_generations[input_id],
                    parallel_ids=parallel_ids[input_id],
                    sequential_ids=sequential_ids[input_id],
                )
            )
        return outputs


@dataclass
class HistoryIterationKParallelSampler(HistoryIterationSampler):
    """Iterative sampler that keeps one random trajectory across k-parallel turns."""

    seed: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self._rng = random.Random(self.seed)

    def run(self, inputs: List[str], draft_prompt: bool = False) -> List[SequentialSamplingResult]:  # type: ignore[override]
        histories = [History(self.history_k) for _ in inputs]
        raw_generations: List[List[str]] = [[] for _ in inputs]
        all_solutions: List[List[Any]] = [[] for _ in inputs]
        parallel_ids: List[List[int]] = [[] for _ in inputs]
        sequential_ids: List[List[int]] = [[] for _ in inputs]

        for turn_id in range(self.sampling_config.k):
            message_batches = [
                self.prompt_schema.build_messages(
                    IterationPromptContext(
                        input_text=inp,
                        k=self.sampling_config.k,
                        previous_solutions=histories[i].last(),
                    ),
                    turn_id=turn_id,
                )
                for i, inp in enumerate(inputs)
            ]
            extra = self.sampling_config.extra_params or {}
            extra = {**extra, **self._constrained_extra_params()}

            if draft_prompt:
                for input_id, messages in enumerate(message_batches):
                    print(f"[draft] turn {turn_id}, prompt {input_id} messages:")
                    for msg in messages:
                        role = msg.role
                        content = msg.content
                        print(f"--- {role} ---\n{content}\n")
                if turn_id > 1:
                    raise SystemExit(0)
                continue

            for input_id, messages in enumerate(message_batches):
                # log messages for debugging
                print(f"Turn {turn_id}, prompt {input_id} messages:")
                for msg in messages:
                    role = msg.role
                    content = msg.content
                    print(f"--- {role} -- {content}")

            try:
                batched_gens = self.client.generate_batch(
                    batch_messages=message_batches,
                    **asdict(self.sampling_config.generation_config),
                    extra_params=extra,
                )
            except Exception as exc:  # noqa: BLE001
                import os

                endpoint = getattr(self, "endpoint", "unknown")
                model_name = getattr(self, "model_name", "unknown")
                port = os.getenv("VLLM_PORT") or "unset"
                host = os.getenv("VLLM_HOST") or "127.0.0.1"
                raise RuntimeError(
                    f"Generation failed (turn={turn_id}, prompts={len(message_batches)}) "
                    f"endpoint={endpoint}, host={host}, port={port}, model={model_name}"
                ) from exc

            for input_id, gens in enumerate(batched_gens):
                parsed_turn_generations: list[str] = []
                for i, gen in enumerate(gens):
                    text = gen.text
                    raw_generations[input_id].append(text)
                    parsed = self._extract_answer(text)
                    parsed_turn_generations.append(parsed)
                    try:
                        new_solution = self.parser.parse(parsed)
                    except Exception:
                        new_solution = parsed  # keep raw parsed content if parser fails
                    all_solutions[input_id].append(new_solution)
                    parallel_ids[input_id].append(i)
                    sequential_ids[input_id].append(turn_id)

                if parsed_turn_generations:
                    histories[input_id].add(self._rng.choice(parsed_turn_generations))

        if draft_prompt:
            raise SystemExit(0)

        outputs: List[SequentialSamplingResult] = []
        for input_id in range(len(inputs)):
            outputs.append(
                SequentialSamplingResult(
                    solutions_per_prompt=all_solutions[input_id],
                    raw_generations=raw_generations[input_id],
                    parallel_ids=parallel_ids[input_id],
                    sequential_ids=sequential_ids[input_id],
                )
            )
        return outputs


def build_iterative_prompt_schema(scenario: Scenario) -> BasePromptSchema:
    first_turn_prompt_schema = FirstTurnPlainPromptSchema(
        system_instruction=scenario.system_prompt or "You are a helpful assistant.",
        first_turn_text=scenario.first_turn_instruction,
        response_instruction=scenario.response_instruction
        or "Provide exactly one concise output as plain text",
    )
    reiteration_turn_prompt_schema = PreviousSolutionsPlainPromptSchema(
        system_instruction=scenario.system_prompt or "You are a helpful assistant.",
        previous_solutions_text=scenario.previous_solutions_text,
        response_instruction=scenario.response_instruction
        or "Provide exactly one concise output as plain text",
    )
    return ConditionalPromptSchema(
        first_turn_prompt_schema=first_turn_prompt_schema,
        reiteration_turn_prompt_schema=reiteration_turn_prompt_schema,
    )


def build_iterative_sampler(scenario: Scenario) -> IterationSampler:
    if scenario.is_iterkpar():
        sampler = HistoryIterationKParallelSampler(
            client=build_seqsampling_client(scenario),
            sampling_config=build_sampling_config(scenario, "iterative"),
            parser=PlainTextParser(),
            prompt_schema=build_iterative_prompt_schema(scenario),
            string_extractor=RemoveThinkingParser(),
            history_k=scenario.history_k,
            seed=scenario.seed,
        )
    else:
        sampler = HistoryIterationSampler(
            client=build_seqsampling_client(scenario),
            sampling_config=build_sampling_config(scenario, "iterative"),
            parser=PlainTextParser(),
            prompt_schema=build_iterative_prompt_schema(scenario),
            string_extractor=RemoveThinkingParser(),
            history_k=scenario.history_k,
        )
    sampler.endpoint = scenario.vllm_base_url if scenario.backend == "vllm" else "openai"
    sampler.model_name = scenario.model
    return sampler


def run_iterative_scenario(
    scenario_path: Path,
    overrides: list[str] | None = None,
    draft_prompt: bool = False,
) -> Path:
    scenario = Scenario.load(scenario_path, overrides=overrides)
    _maybe_dump_resolved_config(scenario)
    prompts = scenario.load_prompts()

    formatted_prompts = [scenario.format_prompt(p) for p in prompts]
    sampler = build_iterative_sampler(scenario)
    outputs = sampler.run(formatted_prompts, draft_prompt=draft_prompt)

    output_path = scenario.output_path()
    write_samples_jsonl(output_path, scenario, prompts, outputs, mode="iterative")
    return output_path


def _maybe_dump_resolved_config(scenario: Scenario) -> None:
    """Write the resolved scenario config to dump_path if provided."""
    if not scenario.dump_path:
        return
    cfg = getattr(scenario, "_resolved_cfg", None)
    scenario.dump_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        from omegaconf import OmegaConf

        scenario.dump_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    else:
        import json

        scenario.dump_path.write_text(json.dumps(scenario.model_dump(), indent=2, default=str), encoding="utf-8")

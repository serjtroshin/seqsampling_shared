from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

from seqsampling.models.base import ChatMessage
from seqsampling.sampling.iteration import IterationPromptContext, IterationSampler, SequentialSamplingResult

from .scenario import Scenario
from .sampling import (
    BasePromptSchema,
    FirstTurnPlainPromptSchema,
    PlainTextParser,
    RemoveThinkingParser,
    build_sampling_config,
    build_seqsampling_client,
    write_samples_jsonl,
)


@dataclass
class ConversationTurn:
    user: str
    assistant: str


@dataclass
class ConversationHistory:
    """Keeps the last k (user, assistant) turns, or all if k is None."""

    k: int | None = None
    items: list[ConversationTurn] | None = None

    def __post_init__(self) -> None:
        if self.items is None:
            self.items = []

    def add(self, user: str, assistant: str) -> None:
        self.items.append(ConversationTurn(user=user, assistant=assistant))
        if self.k is not None and len(self.items) > self.k:
            self.items = self.items[-self.k :]

    def as_messages(self) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        for turn in self.items:
            messages.append(ChatMessage(role="user", content=turn.user))
            messages.append(ChatMessage(role="assistant", content=turn.assistant))
        return messages


@dataclass
class MultiTurnPromptContext(IterationPromptContext):
    history_messages: list[ChatMessage] | None = None


@dataclass
class FirstTurnMultiTurnPromptSchema(FirstTurnPlainPromptSchema):
    """First-turn prompt schema for multi-turn conversation."""


@dataclass
class ReiterationMultiTurnPromptSchema(FirstTurnPlainPromptSchema):
    """
    Reiteration schema that relies on conversation history for previous outputs.
    No explicit previous-solutions listing is added to the prompt text.
    """

    reiteration_instruction: str | None = None

    def build_messages(self, ctx: MultiTurnPromptContext, turn_id: int = 0) -> List[ChatMessage]:
        history_messages = list(ctx.history_messages or [])
        instruction = (
            self.response_instruction
            or "Provide one new response as plain text (no JSON, no lists)."
        )
        reiteration_text = self.reiteration_instruction or self.first_turn_text or ""
        return [
            ChatMessage(role="system", content=self.system_instruction),
            *history_messages,
            ChatMessage(
                role="user",
                content=(
                    f"{instruction}\n"
                    f"{reiteration_text}\n\n"
                    f"{ctx.input_text}"
                ),
            ),
        ]


@dataclass
class MultiTurnConditionalPromptSchema(BasePromptSchema):
    first_turn_prompt_schema: FirstTurnMultiTurnPromptSchema
    reiteration_turn_prompt_schema: ReiterationMultiTurnPromptSchema

    def build_messages(self, ctx: MultiTurnPromptContext, turn_id: int = 0) -> List[ChatMessage]:
        if turn_id == 0:
            return self.first_turn_prompt_schema.build_messages(ctx, turn_id=turn_id)
        return self.reiteration_turn_prompt_schema.build_messages(ctx, turn_id=turn_id)


@dataclass
class HistoryMultiTurnSampler(IterationSampler):
    """Sequential sampler that keeps explicit (user, assistant) history across turns."""

    history_k: int | None = None
    prompt_schema: MultiTurnConditionalPromptSchema | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.prompt_schema is None:
            raise ValueError("prompt_schema must be provided to HistoryMultiTurnSampler")

    def run(self, inputs: List[str], draft_prompt: bool = False) -> List[SequentialSamplingResult]:  # type: ignore[override]
        histories = [ConversationHistory(self.history_k) for _ in inputs]
        raw_generations: List[List[str]] = [[] for _ in inputs]
        all_solutions: List[List[Any]] = [[] for _ in inputs]
        parallel_ids: List[List[int]] = [[] for _ in inputs]
        sequential_ids: List[List[int]] = [[] for _ in inputs]

        for turn_id in range(self.sampling_config.k):
            message_batches = [
                self.prompt_schema.build_messages(
                    MultiTurnPromptContext(
                        input_text=inp,
                        k=self.sampling_config.k,
                        previous_solutions=[],
                        history_messages=histories[i].as_messages(),
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
                        print(f"--- {msg.role} ---\n{msg.content}\n")
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
                user_message = next(
                    (m.content for m in reversed(message_batches[input_id]) if m.role == "user"),
                    "",
                )
                for i, gen in enumerate(gens):
                    text = gen.text
                    raw_generations[input_id].append(text)
                    parsed = self._extract_answer(text)
                    try:
                        new_solution = self.parser.parse(parsed)
                    except Exception:
                        new_solution = parsed

                    histories[input_id].add(user=user_message, assistant=parsed)
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


def build_multi_turn_prompt_schema(scenario: Scenario) -> MultiTurnConditionalPromptSchema:
    first_turn_prompt_schema = FirstTurnMultiTurnPromptSchema(
        system_instruction=scenario.system_prompt or "You are a helpful assistant.",
        first_turn_text=scenario.first_turn_instruction,
        response_instruction=scenario.response_instruction
        or "Provide exactly one concise output as plain text",
    )
    reiteration_turn_prompt_schema = ReiterationMultiTurnPromptSchema(
        system_instruction=scenario.system_prompt or "You are a helpful assistant.",
        first_turn_text=scenario.first_turn_instruction,
        reiteration_instruction=scenario.reiteration_instruction,
        response_instruction=scenario.response_instruction
        or "Provide exactly one concise output as plain text",
    )
    return MultiTurnConditionalPromptSchema(
        first_turn_prompt_schema=first_turn_prompt_schema,
        reiteration_turn_prompt_schema=reiteration_turn_prompt_schema,
    )


def build_multi_turn_sampler(scenario: Scenario) -> HistoryMultiTurnSampler:
    sampler = HistoryMultiTurnSampler(
        client=build_seqsampling_client(scenario),
        sampling_config=build_sampling_config(scenario, "iterative"),
        parser=PlainTextParser(),
        prompt_schema=build_multi_turn_prompt_schema(scenario),
        string_extractor=RemoveThinkingParser(),
        history_k=scenario.history_k,
    )
    sampler.endpoint = scenario.vllm_base_url if scenario.backend == "vllm" else "openai"
    sampler.model_name = scenario.model
    return sampler


def run_multi_turn_scenario(
    scenario_path: Path,
    overrides: list[str] | None = None,
    draft_prompt: bool = False,
) -> Path:
    scenario = Scenario.load(scenario_path, overrides=overrides)
    _maybe_dump_resolved_config(scenario)
    prompts = scenario.load_prompts()

    formatted_prompts = [scenario.format_prompt(p) for p in prompts]
    sampler = build_multi_turn_sampler(scenario)
    outputs = sampler.run(formatted_prompts, draft_prompt=draft_prompt)

    output_path = scenario.output_path()
    write_samples_jsonl(output_path, scenario, prompts, outputs, mode="iterative")
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
    "ConversationHistory",
    "HistoryMultiTurnSampler",
    "build_multi_turn_prompt_schema",
    "build_multi_turn_sampler",
    "run_multi_turn_scenario",
]

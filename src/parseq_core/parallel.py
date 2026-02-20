from __future__ import annotations

from pathlib import Path
from typing import Dict

from seqsampling.prompts.schema import PromptContext
from seqsampling.sampling.parallel import ParallelSampler

from .scenario import Scenario
from .sampling import (
    BasePromptSchema,
    PlainParallelPromptSchema,
    PlainTextParser,
    RemoveThinkingParser,
    build_sampling_config,
    build_seqsampling_client,
    write_samples_jsonl,
)


def load_parallel_prompts(scenario: Scenario) -> Dict[str, str]:
    """
    Build a prompt lookup for a parallel sampling scenario.
    Keys: prompt_id (stringified), Values: formatted prompt text.
    """
    prompts = scenario.load_prompts()
    return {str(p.id): scenario.format_prompt(p) for p in prompts}


def build_parallel_prompt_schema(scenario: Scenario) -> BasePromptSchema:
    return PlainParallelPromptSchema(
        system_instruction=scenario.system_prompt or "You are a helpful assistant.",
        response_instruction=scenario.response_instruction
        or "Provide exactly one concise output as plain text",
        first_turn_instruction=scenario.first_turn_instruction,
    )


def build_parallel_sampler(scenario: Scenario) -> ParallelSampler:
    """
    Create a ParallelSampler configured from the scenario (no history).
    """
    return ParallelSampler(
        client=build_seqsampling_client(scenario),
        sampling_config=build_sampling_config(scenario, "parallel"),
        prompt_schema=build_parallel_prompt_schema(scenario),
        parser=PlainTextParser(),
        string_extractor=RemoveThinkingParser(),
    )


def run_parallel_scenario(scenario: Scenario, draft_prompt: bool = False) -> Path:
    prompts = scenario.load_prompts()
    formatted_prompts = [scenario.format_prompt(p) for p in prompts]

    if draft_prompt:
        if not formatted_prompts:
            print("[draft] no prompts to render.")
            raise SystemExit(0)
        schema = build_parallel_prompt_schema(scenario)
        msg_objs = schema.build_messages(
            PromptContext(input_text=formatted_prompts[0], k=0),
            turn_id=0,
        )
        print("[draft] parallel:")
        for msg in msg_objs:
            print(f"--- {msg.role} ---\n{msg.content}\n")
        raise SystemExit(0)

    sampler = build_parallel_sampler(scenario)
    outputs = sampler.run(formatted_prompts)
    output_path = scenario.output_path()
    write_samples_jsonl(output_path, scenario, prompts, outputs, mode="parallel")
    return output_path


__all__ = [
    "PlainParallelPromptSchema",
    "load_parallel_prompts",
    "build_parallel_prompt_schema",
    "build_parallel_sampler",
    "run_parallel_scenario",
]

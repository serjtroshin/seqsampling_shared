from __future__ import annotations

from pathlib import Path

from seqsampling.prompts.schema import PromptContext
from seqsampling.sampling.iteration import IterationPromptContext

from .enumeration import build_enumeration_prompt_schema
from .iterative import build_iterative_prompt_schema
from .multi_turn import MultiTurnPromptContext, build_multi_turn_prompt_schema
from .parallel import build_parallel_prompt_schema
from .scenario import Prompt, Scenario


def prompt_trace_path_for_scenario(scenario: Scenario) -> Path | None:
    if not scenario.dump_path:
        return None
    return scenario.dump_path.parent / "prompt_trace.txt"


def maybe_dump_prompt_trace(scenario: Scenario) -> Path | None:
    trace_path = prompt_trace_path_for_scenario(scenario)
    if trace_path is None:
        return None

    prompts = scenario.load_prompts()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if not prompts:
        trace_path.write_text("[trace] no prompts loaded\n", encoding="utf-8")
        return trace_path

    trace_path.write_text(render_prompt_trace(scenario, prompts[0]), encoding="utf-8")
    return trace_path


def render_prompt_trace(scenario: Scenario, prompt: Prompt) -> str:
    lines = [
        f"scenario={scenario.name}",
        f"prompt_id={prompt.id}",
        f"prompt_text={prompt.text}",
    ]

    if scenario.is_multi_turn():
        lines.append(f"reiteration_include_input_text={scenario.reiteration_include_input_text}")
    lines.append(f"num_generations={scenario.num_generations}")
    lines.append("")

    for turn_id, messages in _trace_messages_by_turn(scenario, prompt):
        lines.append(f"=== turn {turn_id + 1} ===")
        for message in messages:
            lines.append(f"[{message.role}]")
            lines.append(str(message.content))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _trace_messages_by_turn(scenario: Scenario, prompt: Prompt):
    formatted_prompt = scenario.build_sampler_input(prompt)

    if scenario.is_multi_turn():
        schema = build_multi_turn_prompt_schema(scenario)
        for turn_id in range(scenario.num_generations):
            yield turn_id, schema.build_messages(
                MultiTurnPromptContext(
                    input_text=formatted_prompt,
                    k=scenario.num_generations,
                    previous_solutions=[],
                    history_messages=[],
                ),
                turn_id=turn_id,
            )
        return

    if scenario.is_iterative() or scenario.is_iterkpar():
        schema = build_iterative_prompt_schema(scenario)
        for turn_id in range(scenario.num_generations):
            yield turn_id, schema.build_messages(
                IterationPromptContext(
                    input_text=formatted_prompt,
                    k=scenario.num_generations,
                    previous_solutions=[],
                ),
                turn_id=turn_id,
            )
        return

    if scenario.is_enumeration():
        schema = build_enumeration_prompt_schema(scenario)
        yield 0, schema.build_messages(
            PromptContext(input_text=formatted_prompt, k=scenario.num_generations),
            turn_id=0,
        )
        return

    schema = build_parallel_prompt_schema(scenario)
    yield 0, schema.build_messages(
        PromptContext(input_text=formatted_prompt, k=0),
        turn_id=0,
    )

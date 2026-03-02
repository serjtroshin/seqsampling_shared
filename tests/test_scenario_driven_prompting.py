from __future__ import annotations

import json
from pathlib import Path

from seqsampling.models.base import ChatMessage
from seqsampling.prompts.schema import PromptContext

from active_mt_distill.sft.dataset_export import export_for_llamafactory
from parseq_core.iterative import History
from parseq_core.multi_turn import (
    ConversationHistory,
    MultiTurnPromptContext,
    build_multi_turn_prompt_schema,
)
from parseq_core.parallel import build_parallel_prompt_schema
from parseq_core.prompt_trace import maybe_dump_prompt_trace
from parseq_core.scenario import Scenario


def test_generation_prompt_uses_row_level_language_markers(tmp_path: Path) -> None:
    prompts_path = tmp_path / "candidate_pool.jsonl"
    prompts_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "src_lang": "eng_Latn",
                "tgt_lang": "fra_Latn",
                "src_text": "Hello world",
                "tgt_text": "Bonjour le monde",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: mt_parallel",
                "backend: vllm",
                "sampling_mode: parallel",
                "model: test-model",
                f"prompts_path: {prompts_path}",
                "prompt_key: src_text",
                "extra_data_fields: [src_lang, tgt_lang, tgt_text]",
                "output_dir: outputs",
                'prompt_template: "{src_lang_name}: {prompt}"',
                'first_turn_instruction: "Translate the following text from {src_lang_name} to {tgt_lang_name}."',
                'response_instruction: "Provide only one translation."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scenario = Scenario.load(scenario_path)
    prompt = scenario.load_prompts()[0]

    assert prompt.extra_data["src_lang"] == "eng_Latn"
    assert prompt.extra_data["tgt_lang"] == "fra_Latn"
    assert prompt.extra_data["target"] == "Bonjour le monde"
    assert scenario.format_prompt(prompt) == "English: Hello world"

    schema = build_parallel_prompt_schema(scenario)
    messages = schema.build_messages(PromptContext(input_text=scenario.build_sampler_input(prompt), k=1))

    assert messages[1].content is not None
    assert "Translate the following text from English to French." in messages[1].content
    assert "English: Hello world" in messages[1].content


def test_sft_export_uses_scenario_formatting(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "opus_train_pool.jsonl"
    train_jsonl.write_text(
        json.dumps(
            {
                "id": "opus-row-1",
                "src_lang": "eng_Latn",
                "tgt_lang": "deu_Latn",
                "src_text": "Fingerprints.",
                "tgt_text": "Fingerabdruecke.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    scenario_path = tmp_path / "base_SFT_scenario.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: base_sft_scenario",
                "backend: vllm",
                "sampling_mode: parallel",
                "model: test-model",
                f"prompts_path: {train_jsonl}",
                "prompt_key: src_text",
                "extra_data_fields: [src_lang, tgt_lang, tgt_text]",
                "output_dir: outputs",
                'prompt_template: "{src_lang_name}: {prompt}"',
                'first_turn_instruction: "Translate the following text from {src_lang_name} to {tgt_lang_name}."',
                'response_instruction: "Provide only the translation."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = export_for_llamafactory(
        train_jsonl=train_jsonl,
        output_dir=tmp_path / "lf_dataset",
        dataset_name="active_mt_train",
        scenario_path=scenario_path,
        scenario_overrides=None,
    )

    train_payload = json.loads(Path(result["train_file"]).read_text(encoding="utf-8"))
    assert len(train_payload) == 1
    assert train_payload[0]["instruction"] == "Translate the following text from English to German."
    assert train_payload[0]["input"] == "English: Fingerprints."
    assert train_payload[0]["output"] == "Fingerabdruecke."


def test_multi_turn_reiteration_uses_row_level_language_markers(tmp_path: Path) -> None:
    prompts_path = tmp_path / "candidate_pool.jsonl"
    prompts_path.write_text(
        json.dumps(
            {
                "id": "row-2",
                "src_lang": "eng_Latn",
                "tgt_lang": "rus_Cyrl",
                "src_text": "Good morning",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    scenario_path = tmp_path / "multi_turn.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: mt_multi_turn",
                "backend: vllm",
                "sampling_mode: multi_turn",
                "model: test-model",
                f"prompts_path: {prompts_path}",
                "prompt_key: src_text",
                "extra_data_fields: [src_lang, tgt_lang]",
                "output_dir: outputs",
                'prompt_template: "{src_lang_name}: {prompt}"',
                'first_turn_instruction: "Translate the following text from {src_lang_name} to {tgt_lang_name}."',
                'reiteration_instruction: "Please again translate the following text from {src_lang_name} to {tgt_lang_name}."',
                'response_instruction: "Provide only one translation."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scenario = Scenario.load(scenario_path)
    prompt = scenario.load_prompts()[0]
    schema = build_multi_turn_prompt_schema(scenario)
    messages = schema.build_messages(
        MultiTurnPromptContext(
            input_text=scenario.build_sampler_input(prompt),
            k=2,
            previous_solutions=[],
            history_messages=[],
        ),
        turn_id=1,
    )

    assert [message.role for message in messages] == ["user"]
    assert messages[0].content is not None
    assert "Please again translate the following text from English to Russian." in messages[0].content
    assert "English: Good morning" in messages[0].content


def test_multi_turn_reiteration_can_hide_input_text(tmp_path: Path) -> None:
    prompts_path = tmp_path / "candidate_pool.jsonl"
    prompts_path.write_text(
        json.dumps(
            {
                "id": "row-3",
                "src_lang": "eng_Latn",
                "tgt_lang": "deu_Latn",
                "src_text": "Good night",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    scenario_path = tmp_path / "multi_turn_hide_input.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: mt_multi_turn",
                "backend: vllm",
                "sampling_mode: multi_turn",
                "model: test-model",
                f"prompts_path: {prompts_path}",
                "prompt_key: src_text",
                "extra_data_fields: [src_lang, tgt_lang]",
                "output_dir: outputs",
                'prompt_template: "{src_lang_name}: {prompt}"',
                'reiteration_instruction: "Please translate again for a better version."',
                "reiteration_include_input_text: false",
                'response_instruction: "Provide only one translation."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scenario = Scenario.load(scenario_path)
    prompt = scenario.load_prompts()[0]
    schema = build_multi_turn_prompt_schema(scenario)
    messages = schema.build_messages(
        MultiTurnPromptContext(
            input_text=scenario.build_sampler_input(prompt),
            k=2,
            previous_solutions=[],
            history_messages=[],
        ),
        turn_id=1,
    )

    assert [message.role for message in messages] == ["user"]
    assert messages[0].content is not None
    assert "Please translate again for a better version." in messages[0].content
    assert "English: Good night" not in messages[0].content


def test_multi_turn_can_split_first_and_reiteration_response_instructions(tmp_path: Path) -> None:
    prompts_path = tmp_path / "candidate_pool.jsonl"
    prompts_path.write_text(
        json.dumps(
            {
                "id": "row-4",
                "src_lang": "eng_Latn",
                "tgt_lang": "deu_Latn",
                "src_text": "See you later",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    scenario_path = tmp_path / "multi_turn_split_response.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: mt_multi_turn",
                "backend: vllm",
                "sampling_mode: multi_turn",
                "model: test-model",
                f"prompts_path: {prompts_path}",
                "prompt_key: src_text",
                "extra_data_fields: [src_lang, tgt_lang]",
                "output_dir: outputs",
                'prompt_template: "{src_lang_name}: {prompt}"',
                'first_turn_instruction: "Translate the following text from {src_lang_name} to {tgt_lang_name}."',
                'reiteration_instruction: "Please translate again for a better version."',
                'first_turn_response_instruction: "FIRST TURN RESPONSE."',
                'reiteration_response_instruction: "REITERATION RESPONSE."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scenario = Scenario.load(scenario_path)
    prompt = scenario.load_prompts()[0]
    schema = build_multi_turn_prompt_schema(scenario)

    first_turn_messages = schema.build_messages(
        MultiTurnPromptContext(
            input_text=scenario.build_sampler_input(prompt),
            k=2,
            previous_solutions=[],
            history_messages=[],
        ),
        turn_id=0,
    )
    reiteration_messages = schema.build_messages(
        MultiTurnPromptContext(
            input_text=scenario.build_sampler_input(prompt),
            k=2,
            previous_solutions=[],
            history_messages=[],
        ),
        turn_id=1,
    )

    assert first_turn_messages[1].content is not None
    assert "FIRST TURN RESPONSE." in first_turn_messages[1].content
    assert "REITERATION RESPONSE." not in first_turn_messages[1].content

    assert reiteration_messages[0].content is not None
    assert "REITERATION RESPONSE." in reiteration_messages[0].content
    assert "FIRST TURN RESPONSE." not in reiteration_messages[0].content


def test_multi_turn_reiteration_reuses_only_user_assistant_history(tmp_path: Path) -> None:
    scenario_path = tmp_path / "multi_turn_history.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: mt_multi_turn",
                "backend: vllm",
                "sampling_mode: multi_turn",
                "model: test-model",
                f"prompts_path: {tmp_path / 'prompts.txt'}",
                "output_dir: outputs",
                'reiteration_instruction: "Please translate again for a better version."',
                'response_instruction: "Provide only one translation."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "prompts.txt").write_text("Test prompt\n", encoding="utf-8")

    scenario = Scenario.load(scenario_path)
    prompt = scenario.load_prompts()[0]
    schema = build_multi_turn_prompt_schema(scenario)
    messages = schema.build_messages(
        MultiTurnPromptContext(
            input_text=scenario.build_sampler_input(prompt),
            k=3,
            previous_solutions=[],
            system_message="You are a helpful assistant.",
            history_messages=[
                ChatMessage(role="user", content="previous user"),
                ChatMessage(role="assistant", content="previous assistant"),
            ],
        ),
        turn_id=1,
    )

    assert [message.role for message in messages] == ["user", "assistant", "user"]


def test_prompt_trace_dump_uses_first_prompt_and_all_turns(tmp_path: Path) -> None:
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("Alpha line\nBeta line\n", encoding="utf-8")

    scenario_path = tmp_path / "multi_turn_trace.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "name: mt_multi_turn",
                "backend: vllm",
                "sampling_mode: multi_turn",
                "model: test-model",
                f"prompts_path: {prompts_path}",
                "output_dir: outputs",
                f"dump_path: {tmp_path / 'generation' / 'mt_multi_turn' / 'scenario_resolved.yaml'}",
                'prompt_template: "{src_lang_name}: {prompt}"',
                'first_turn_instruction: "Translate the following text from {src_lang_name} to {tgt_lang_name}."',
                'reiteration_instruction: "Please translate again for a better version."',
                "reiteration_include_input_text: false",
                'response_instruction: "Provide only one translation."',
                "num_generations: 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scenario = Scenario.load(scenario_path)
    trace_path = maybe_dump_prompt_trace(scenario)

    assert trace_path is not None
    assert trace_path.exists()
    payload = trace_path.read_text(encoding="utf-8")
    assert "prompt_id=0" in payload
    assert "prompt_text=Alpha line" in payload
    assert "=== turn 1 ===" in payload
    assert "=== turn 3 ===" in payload
    assert payload.count("english: Alpha line") == 1
    assert payload.count("[system]") == 1
    assert payload.count("Please translate again for a better version.") == 2


def test_history_none_keeps_full_available_history() -> None:
    history = History(k=None)
    history.add("first")
    history.add("second")
    history.add("third")
    assert history.last() == ["first", "second", "third"]

    conversation = ConversationHistory(k=None)
    conversation.add("u1", "a1")
    conversation.add("u2", "a2")
    conversation.add("u3", "a3")
    assert [(item.user, item.assistant) for item in conversation.items or []] == [
        ("u1", "a1"),
        ("u2", "a2"),
        ("u3", "a3"),
    ]

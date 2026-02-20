from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI

try:  # required for chat-template rendering used by parseq-likelihood
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore

from seqsampling.prompts.schema import PromptContext

from .parallel import build_parallel_prompt_schema, load_parallel_prompts
from .scenario import Scenario

# Cache tokenizers per model to avoid reloading.
_TOKENIZER_CACHE: dict[str, Any] = {}


def _load_prompts_from_samples(path: Path) -> dict[str, str]:
    """Load prompt_id->prompt mapping from a parallel samples JSONL file."""
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt_id = str(row.get("prompt_id", ""))
        prompt_text = row.get("prompt")
        if prompt_id and isinstance(prompt_text, str):
            mapping[prompt_id] = prompt_text
    return mapping


def _load_parallel_prompts_with_fallback(scenario: Scenario) -> dict[str, str]:
    """
    Prefer scenario prompt loading; if the resolved prompts file is unavailable,
    fall back to run-local parallel samples.
    """
    try:
        return load_parallel_prompts(scenario)
    except FileNotFoundError:
        sample_candidates = [
            scenario.output_path(),
            scenario.resolved_output_dir() / "samples.jsonl",
        ]
        for sample_path in sample_candidates:
            if sample_path.exists():
                mapping = _load_prompts_from_samples(sample_path)
                if mapping:
                    return mapping
        raise


def _build_client(scenario: Scenario) -> OpenAI:
    """Construct an OpenAI-compatible client for vLLM/OpenAI backends."""
    api_key = scenario.vllm_api_key or scenario.openai_api_key or "token"
    base_url = scenario.vllm_base_url if scenario.backend == "vllm" else None
    timeout = scenario.request_timeout or 3600.0
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _get_tokenizer(model: str, trust_remote_code: bool):
    if AutoTokenizer is None:
        raise RuntimeError(
            "transformers is required for parseq-likelihood chat-template scoring."
        )
    if model not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model] = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )
    return _TOKENIZER_CACHE[model]


def _render_for_scoring(
    scenario: Scenario,
    prompt_text: str,
    response_text: str,
) -> tuple[str, str]:
    """
    Return (rendered_prefix, rendered_full).
    """
    assert scenario.system_prompt and scenario.system_prompt.strip(), (
        "parseq-likelihood requires a non-empty system_prompt in the scoring scenario."
    )

    tokenizer = _get_tokenizer(scenario.model, scenario.trust_remote_code)

    # Mirror parallel sampling prompt construction (same path as draft prompt rendering),
    # then apply tokenizer chat template for scoring.
    schema = build_parallel_prompt_schema(scenario)
    msg_objs = schema.build_messages(PromptContext(input_text=prompt_text, k=0), turn_id=0)
    base_msgs = [
        {"role": str(msg.role), "content": str(msg.content)}
        for msg in msg_objs
    ]

    prefix_msgs = base_msgs + [{"role": "assistant", "content": ""}]
    full_msgs = base_msgs + [{"role": "assistant", "content": response_text}]

    rendered_prefix = tokenizer.apply_chat_template(
        prefix_msgs, tokenize=False, add_generation_prompt=False
    )
    rendered_full = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False
    )
    return rendered_prefix, rendered_full


def _as_float(value: Any) -> float | None:
    """Best-effort conversion for mixed backend payload types."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_top_logprobs(entries: Any) -> list[dict[str, Any]]:
    """Normalize backend-specific top_logprobs entries into a stable dict schema."""
    if not entries:
        return []

    if isinstance(entries, dict):
        return [
            {
                "token": token,
                "token_id": None,
                "logprob": _as_float(logprob),
            }
            for token, logprob in entries.items()
        ]

    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict):
            normalized.append(
                {
                    "token": entry.get("token"),
                    "token_id": entry.get("token_id"),
                    "logprob": _as_float(entry.get("logprob")),
                }
            )
            continue
        token = getattr(entry, "token", None)
        token_id = getattr(entry, "token_id", None)
        logprob = _as_float(getattr(entry, "logprob", None))
        if token is None and token_id is None and logprob is None:
            token = entry
        normalized.append(
            {
                "token": token,
                "token_id": token_id,
                "logprob": logprob,
            }
        )
    return normalized


def _extract_completion_tokens(
    choice: Any,
    response_start_token_idx: int,
    response_text: str,
) -> dict[str, Any]:
    """
    Extract response-only token stats from a completion choice.

    `response_start_token_idx` is computed from a separate prefix-only request,
    avoiding brittle byte/char offsets for multilingual text.
    """
    logprobs = getattr(choice, "logprobs", None)
    assert logprobs is not None, "Expected completion choice.logprobs to be present."
    tokens_all = list(getattr(logprobs, "tokens", []) or [])
    logprobs_all = list(getattr(logprobs, "token_logprobs", []) or [])
    tops_all = list(getattr(logprobs, "top_logprobs", []) or [])
    offsets_all = list(getattr(logprobs, "text_offset", []) or [])

    prompt_token_ids = list(getattr(choice, "prompt_token_ids", []) or [])
    token_ids_all = prompt_token_ids if len(prompt_token_ids) >= len(tokens_all) else list(
        getattr(logprobs, "token_ids", []) or []
    )

    start = max(0, min(response_start_token_idx, len(tokens_all)))
    response_tokens: list[dict[str, Any]] = []
    response_logprobs: list[float | None] = []

    for idx in range(start, len(tokens_all)):
        lp = _as_float(logprobs_all[idx]) if idx < len(logprobs_all) else None
        entry = {
            "token": tokens_all[idx],
            "token_id": token_ids_all[idx] if idx < len(token_ids_all) else None,
            "logprob": lp,
            "offset": offsets_all[idx] if idx < len(offsets_all) else None,
            "top_logprobs": _normalize_top_logprobs(tops_all[idx] if idx < len(tops_all) else None),
        }
        response_tokens.append(entry)
        response_logprobs.append(lp)

    finite = [lp for lp in response_logprobs if lp is not None]
    logprob_sum = float(sum(finite)) if finite else None
    logprob_mean_token = float(logprob_sum / len(finite)) if finite else None
    response_bytes = len(response_text.encode("utf-8"))
    logprob_mean_byte = (
        float(logprob_sum / response_bytes)
        if logprob_sum is not None and response_bytes > 0
        else None
    )

    return {
        "response_tokens": response_tokens,
        "response_token_ids": [t["token_id"] for t in response_tokens],
        "response_token_logprobs": response_logprobs if response_logprobs else None,
        "response_logprob_sum": logprob_sum,
        "response_logprob_mean": logprob_mean_token,  # backward compatibility
        "response_logprob_mean_token": logprob_mean_token,
        "response_logprob_mean_byte": logprob_mean_byte,
        "response_token_count": len(finite),
        "response_byte_count": response_bytes,
    }


def _build_completion_extra_body(
    config: dict[str, Any],
    *,
    return_token_ids: bool,
) -> dict[str, Any]:
    """Build vLLM-specific completion request extras used for scoring."""
    extra_body: dict[str, Any] = {}

    top_k = config.get("top_k")
    if top_k is not None:
        extra_body["top_k"] = int(top_k)

    min_p = config.get("min_p")
    if min_p is not None:
        extra_body["min_p"] = float(min_p)

    repetition_penalty = config.get("repetition_penalty")
    if repetition_penalty is not None:
        extra_body["repetition_penalty"] = float(repetition_penalty)

    allowed_token_ids = config.get("allowed_token_ids")
    if allowed_token_ids:
        extra_body["allowed_token_ids"] = allowed_token_ids

    bad_words = config.get("bad_words")
    if bad_words:
        extra_body["bad_words"] = bad_words

    if return_token_ids:
        extra_body["return_token_ids"] = True

    return extra_body


def _completion_choice(
    client: OpenAI,
    scenario: Scenario,
    prompt: str,
    config: dict[str, Any],
    top_logprobs: int,
) -> Any:
    """Issue one completion request and return the first choice."""
    completion = client.completions.create(
        model=scenario.model,
        prompt=prompt,
        max_tokens=0,
        logprobs=max(1, top_logprobs),
        echo=True,
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        frequency_penalty=float(config["frequency_penalty"]),
        presence_penalty=float(config["presence_penalty"]),
        extra_body=_build_completion_extra_body(config, return_token_ids=True),
    )
    if not completion.choices:
        raise RuntimeError("Completion response has no choices.")
    return completion.choices[0]


def _count_prompt_tokens(choice: Any) -> int:
    """Read prompt-token count from completion metadata (vLLM-compatible payload)."""
    prompt_token_ids = getattr(choice, "prompt_token_ids", None)
    if isinstance(prompt_token_ids, list) and prompt_token_ids:
        return len(prompt_token_ids)

    logprobs = getattr(choice, "logprobs", None)
    tokens = getattr(logprobs, "tokens", None)
    if isinstance(tokens, list):
        return len(tokens)

    raise RuntimeError(
        "Could not determine prompt token count from completion response. "
        "Ensure the backend supports return_token_ids/logprobs in completion API."
    )


def _score_response(
    client: OpenAI,
    scenario: Scenario,
    prompt_text: str,
    response: str,
    *,
    config: dict[str, Any],
    top_logprobs: int,
    prefix_token_count_cache: dict[str, int],
) -> dict[str, Any]:
    rendered_prefix, rendered_full = _render_for_scoring(
        scenario, prompt_text, response
    )

    prefix_token_count = prefix_token_count_cache.get(rendered_prefix)
    if prefix_token_count is None:
        prefix_choice = _completion_choice(
            client,
            scenario,
            prompt=rendered_prefix,
            config=config,
            top_logprobs=1,
        )
        prefix_token_count = _count_prompt_tokens(prefix_choice)
        prefix_token_count_cache[rendered_prefix] = prefix_token_count

    full_choice = _completion_choice(
        client,
        scenario,
        prompt=rendered_full,
        config=config,
        top_logprobs=top_logprobs,
    )
    full_prompt_token_count = _count_prompt_tokens(full_choice)
    assert prefix_token_count <= full_prompt_token_count, (
        "Prefix token count cannot exceed full prompt token count."
    )
    token_data = _extract_completion_tokens(
        full_choice,
        response_start_token_idx=prefix_token_count,
        response_text=response,
    )

    return {
        "prompt": prompt_text,
        "response": response,
        "rendered_prefix": rendered_prefix,
        "rendered_full": rendered_full,
        "response_tokens": token_data["response_tokens"],
        "response_token_ids": token_data["response_token_ids"],
        "response_token_logprobs": token_data["response_token_logprobs"],
        "response_logprob_sum": token_data["response_logprob_sum"],
        "response_logprob_mean": token_data["response_logprob_mean"],
        "response_logprob_mean_token": token_data["response_logprob_mean_token"],
        "response_logprob_mean_byte": token_data["response_logprob_mean_byte"],
        "response_token_count": token_data["response_token_count"],
        "response_byte_count": token_data["response_byte_count"],
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "top_k": int(config["top_k"]),
        "min_p": float(config["min_p"]),
        "frequency_penalty": float(config["frequency_penalty"]),
        "presence_penalty": float(config["presence_penalty"]),
        "repetition_penalty": float(config["repetition_penalty"]),
        "top_logprobs": max(1, top_logprobs),
        "rendered_prefix_token_count": prefix_token_count,
        "rendered_full_token_count": full_prompt_token_count,
        "rendered_length_bytes": len(rendered_full.encode("utf-8")),
        "used_chat_template": True,
        "scoring_preset": config["preset"],
    }


def _iter_solution_outputs(path: Path) -> Iterator[dict[str, Any]]:
    """
    Yield outputs from the 'solutions' field only.
    """
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)

        responses = payload.get("solutions")
        if not isinstance(responses, list):
            raise ValueError(
                f"Expected list field 'solutions' on line {line_no} in {path}; "
                "parseq-likelihood now scores solutions only."
            )

        parallel_ids = payload.get("parallel_ids") if isinstance(payload.get("parallel_ids"), list) else []
        sequential_ids = payload.get("sequential_ids") if isinstance(payload.get("sequential_ids"), list) else []
        prompt_id = str(payload.get("prompt_id", ""))

        for idx, resp in enumerate(responses):
            if resp is None:
                continue
            yield {
                "prompt_id": prompt_id,
                "response": str(resp),
                "response_idx": idx,
                "parallel_id": parallel_ids[idx] if idx < len(parallel_ids) else None,
                "sequential_id": sequential_ids[idx] if idx < len(sequential_ids) else None,
            }


def _resolve_scoring_config(args: argparse.Namespace, scenario: Scenario) -> dict[str, Any]:
    """Resolve effective scoring parameters from preset + CLI overrides."""
    extra = dict(scenario.vllm_extra_body or {})

    if args.scoring_preset == "matched":
        temperature = scenario.temperature
        top_p = scenario.top_p
        top_k = scenario.top_k
        min_p = extra.get("min_p", 0.0)
        frequency_penalty = extra.get("frequency_penalty", 0.0)
        presence_penalty = extra.get("presence_penalty", 0.0)
        repetition_penalty = extra.get("repetition_penalty", 1.0)
        allowed_token_ids = extra.get("allowed_token_ids")
        bad_words = extra.get("bad_words")
    else:
        # Approximate the model distribution with ad-hoc sampling filters disabled.
        temperature = 1.0
        top_p = 1.0
        top_k = 0
        min_p = 0.0
        frequency_penalty = 0.0
        presence_penalty = 0.0
        repetition_penalty = 1.0
        allowed_token_ids = None
        bad_words = None

    if args.temperature is not None:
        temperature = args.temperature
    if args.top_p is not None:
        top_p = args.top_p
    if args.top_k is not None:
        top_k = args.top_k
    if args.min_p is not None:
        min_p = args.min_p
    if args.frequency_penalty is not None:
        frequency_penalty = args.frequency_penalty
    if args.presence_penalty is not None:
        presence_penalty = args.presence_penalty
    if args.repetition_penalty is not None:
        repetition_penalty = args.repetition_penalty

    return {
        "preset": args.scoring_preset,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": float(min_p),
        "frequency_penalty": float(frequency_penalty),
        "presence_penalty": float(presence_penalty),
        "repetition_penalty": float(repetition_penalty),
        "allowed_token_ids": allowed_token_ids,
        "bad_words": bad_words,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score likelihoods for sequential outputs under a parallel scenario prompt."
        )
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Parallel sampling scenario to define prompts/model.",
    )
    parser.add_argument(
        "--sequential-samples",
        type=Path,
        required=True,
        help="JSONL produced by an iterative (sequential) run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSONL (default: <output_dir>/likelihoods_from_sequential.jsonl).",
    )
    parser.add_argument("--limit", type=int, help="Optional limit on number of responses to score.")
    parser.add_argument(
        "--scoring-preset",
        choices=["true", "matched"],
        default="true",
        help=(
            "Scoring distribution preset: 'true' disables ad-hoc filters "
            "(temperature=1, top_p=1, top_k=0, min_p=0, neutral penalties), "
            "'matched' copies scenario sampling settings."
        ),
    )
    parser.add_argument("--temperature", type=float, help="Temperature override for scoring.")
    parser.add_argument("--top-p", dest="top_p", type=float, help="Top-p override for scoring.")
    parser.add_argument("--top-k", type=int, help="Top-k override for scoring.")
    parser.add_argument("--min-p", type=float, help="Min-p override for scoring.")
    parser.add_argument("--presence-penalty", type=float, help="Presence penalty override for scoring.")
    parser.add_argument("--frequency-penalty", type=float, help="Frequency penalty override for scoring.")
    parser.add_argument("--repetition-penalty", type=float, help="Repetition penalty override for scoring.")
    parser.add_argument(
        "--top-logprobs",
        type=int,
        help="Number of alternative logprobs to request when scoring (min 1).",
    )

    args, overrides = parser.parse_known_args()

    scenario = Scenario.load(args.scenario, overrides=overrides)
    if scenario.backend != "vllm":
        raise ValueError(
            "parseq-likelihood currently supports backend=vllm only "
            "(requires vLLM-compatible completion logprob/token-id fields)."
        )
    if not scenario.system_prompt or not scenario.system_prompt.strip():
        raise ValueError(
            "parseq-likelihood requires a non-empty system_prompt in the scoring scenario."
        )
    prompt_lookup = _load_parallel_prompts_with_fallback(scenario)
    client = _build_client(scenario)
    config = _resolve_scoring_config(args, scenario)

    top_logprobs = max(
        1,
        args.top_logprobs if args.top_logprobs is not None else (scenario.top_logprobs or 1),
    )

    output_path = args.output or (scenario.resolved_output_dir() / "likelihoods_from_sequential.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_prompt = 0
    scored = 0
    prefix_token_count_cache: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as fout:
        for row in _iter_solution_outputs(args.sequential_samples):
            if args.limit is not None and scored >= args.limit:
                break

            prompt_text = prompt_lookup.get(row["prompt_id"])
            if prompt_text is None:
                missing_prompt += 1
                continue

            record = _score_response(
                client,
                scenario,
                prompt_text,
                row["response"],
                config=config,
                top_logprobs=top_logprobs,
                prefix_token_count_cache=prefix_token_count_cache,
            )
            record.update(
                {
                    "scenario": scenario.name,
                    "backend": scenario.backend,
                    "model": scenario.model,
                    "prompt_id": row["prompt_id"],
                    "response_idx": row["response_idx"],
                    "parallel_id": row["parallel_id"],
                    "sequential_id": row["sequential_id"],
                }
            )
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            scored += 1

    msg = f"Scored {scored} responses -> {output_path}"
    if missing_prompt:
        msg += f" (skipped {missing_prompt} with unknown prompt_id)"
    print(msg)


if __name__ == "__main__":
    main()

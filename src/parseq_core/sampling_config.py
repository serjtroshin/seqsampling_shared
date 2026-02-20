from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SamplingConfig(BaseModel):
    backend: Literal["vllm", "openai"]
    model: str
    prompts_path: Path
    output_path: Path
    top_k: int = Field(default=5, ge=1)
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    num_generations: int = Field(default=1, ge=1)
    seed: int | None = None
    trust_remote_code: bool = False
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str | None = None
    vllm_host: str = "127.0.0.1"
    vllm_port: int = Field(default=8000, ge=1, le=65535)
    vllm_server_args: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> SamplingConfig:
        data = path.read_text(encoding="utf-8")
        return cls.model_validate_json(data)

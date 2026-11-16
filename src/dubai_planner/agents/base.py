"""Shared agent plumbing: prompt files, the JSON context block, LLM-output base model.

Every LLM-backed agent follows the same contract:

``run(input_model, llm, tools, ...) -> output_model``

- ``input_model`` / ``output_model`` are Pydantic (defined in the agent module).
- ``llm`` is anything :func:`~dubai_planner.llm.base.as_chat_llm` accepts: a LangChain
  chat model (LangGraph), a CrewAI LLM (CrewAI) or a ``ChatLLM`` such as ``MockLLM``.
- ``tools`` is a :class:`~dubai_planner.tools.toolbox.Toolbox`.

Prompts are Markdown files in ``agents/prompts/`` with ``$placeholders``
(``string.Template``, so JSON braces need no escaping). User-provided text is always
wrapped in ``<user_request>`` tags and the system prompts tell the model to treat it as
data, never as instructions (prompt-injection guard).
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from string import Template
from typing import Any

from pydantic import BaseModel, ConfigDict

MAX_USER_TEXT = 4000


class LLMOutput(BaseModel):
    """Base for schemas the LLM fills: unknown keys are ignored rather than fatal.

    Small local models often add harmless extra keys; rejecting them would waste the
    single repair attempt. Required fields and types are still validated strictly.
    """

    model_config = ConfigDict(extra="ignore")


@cache
def _prompt_text(name: str) -> str:
    return (resources.files("dubai_planner.agents") / "prompts" / f"{name}.md").read_text(
        encoding="utf-8")


def load_prompt(name: str, **values: Any) -> str:
    """Load ``prompts/<name>.md`` and substitute ``$placeholders`` (missing → KeyError)."""
    return Template(_prompt_text(name)).substitute({k: str(v) for k, v in values.items()})


def context_block(data: BaseModel | dict[str, Any]) -> str:
    """Machine-readable context the model (and the mock provider) reads."""
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    return f"<context>\n{json.dumps(payload, ensure_ascii=False, default=str)}\n</context>"


def user_text_block(text: str) -> str:
    """Wrap untrusted user text; strip tag look-alikes so it cannot close the block."""
    clean = text.replace("<", "‹").replace(">", "›")[:MAX_USER_TEXT]
    return f"<user_request>\n{clean}\n</user_request>"

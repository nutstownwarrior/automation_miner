"""Turn a mined, backtested candidate into validated automation YAML.

The LLM is used at exactly one point: translating an *already proven* candidate
into HA's YAML dialect with a readable alias.  It never sees raw history - only
the neutral schema plus the resolved entity ids it is permitted to use.

Whatever it returns goes through the same gate as the deterministic renderer,
and if it fails the gate we fall back to that renderer rather than showing the
user nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..miners.base import Candidate
from .blueprint import automation_id, candidate_to_automation, render_yaml
from .provider import BaseProvider, LLMError, NullProvider
from .validate import ValidationReport, validate_automation

_LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You convert a pre-validated home-automation rule into a Home Assistant automation.

Rules you MUST follow:
- Output ONE JSON object and nothing else. No prose, no markdown fences.
- Use ONLY the entity_ids listed in "allowed_entities". Never invent an entity_id,
  device_id, area_id or service. Inventing one makes the output unusable.
- Use ONLY the services listed in "allowed_services".
- Keep the semantics of the supplied rule EXACTLY: same triggers, same conditions,
  same actions. You may improve the alias and description wording only.
- "alias" must be a short human sentence describing what the automation does.
- "mode" must be one of: single, restart, queued, parallel.

The JSON object must have this shape:
{
  "alias": "...",
  "description": "...",
  "mode": "single",
  "trigger": [ {"platform": "...", ...} ],
  "condition": [ {"condition": "...", ...} ],
  "action": [ {"service": "domain.service", "target": {"entity_id": "..."}, "data": {}} ]
}
"""


@dataclass
class GenerationResult:
    """What came out of the generation step."""

    config: dict[str, Any] | None = None
    yaml_text: str = ""
    source: str = "blueprint"  # "llm" | "blueprint" | "llm-rejected"
    report: ValidationReport | None = None
    llm_error: str | None = None
    llm_report: ValidationReport | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.config) and bool(self.report and self.report.ok)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source": self.source,
            "yaml": self.yaml_text,
            "config": self.config,
            "validation": self.report.as_dict() if self.report else None,
            "llm_validation": self.llm_report.as_dict() if self.llm_report else None,
            "llm_error": self.llm_error,
            "notes": self.notes,
        }


def build_prompt(candidate: Candidate, resolver=None, services: list[str] | None = None) -> str:
    """The user prompt: the neutral schema plus the entities it may reference."""
    allowed_entities = []
    for entity_id in candidate.entities:
        info = resolver.resolve(entity_id) if resolver else None
        allowed_entities.append(
            {
                "entity_id": entity_id,
                "name": info.name if info else entity_id,
                "area": info.area_name if info else None,
                "device": info.device_name if info else None,
                "labels": info.label_names if info else [],
            }
        )
    allowed_services = sorted({a.service for a in candidate.actions})
    if services:
        allowed_services = sorted(set(allowed_services) & set(services)) or allowed_services

    payload = {
        "rule": {
            "summary": candidate.describe(resolver),
            "trigger": [t.as_dict() for t in candidate.triggers],
            "condition": [c.as_dict() for c in candidate.conditions],
            "action": [a.as_dict() for a in candidate.actions],
        },
        "evidence": candidate.evidence.summary(),
        "backtest": (candidate.backtest or {}).get("summary"),
        "allowed_entities": allowed_entities,
        "allowed_services": allowed_services,
    }
    return json.dumps(payload, indent=2, default=str)


def _normalise_llm_config(raw: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    """Accept the modern plural keys and force our stable id onto the result."""
    config = dict(raw)
    if "triggers" in config and "trigger" not in config:
        config["trigger"] = config.pop("triggers")
    if "conditions" in config and "condition" not in config:
        config["condition"] = config.pop("conditions")
    if "actions" in config and "action" not in config:
        config["action"] = config.pop("actions")
    for key in ("trigger", "condition", "action"):
        value = config.get(key)
        if isinstance(value, dict):
            config[key] = [value]
        elif value is None and key == "condition":
            config[key] = []
    config["id"] = automation_id(candidate)
    return config


def generate(
    candidate: Candidate,
    resolver=None,
    provider: BaseProvider | None = None,
    client=None,
    services: list[str] | None = None,
    run_check_config: bool = True,
) -> GenerationResult:
    """Produce validated YAML for *candidate*, LLM-assisted where available."""
    result = GenerationResult()
    provider = provider or NullProvider()

    # The deterministic rendering is always computed: it is both the fallback
    # and the reference we validate the LLM's output against.
    try:
        blueprint_config = candidate_to_automation(candidate, resolver)
    except ValueError as err:
        result.notes.append(f"cannot render candidate: {err}")
        return result

    if provider.enabled:
        try:
            raw = provider.complete_json(
                SYSTEM_PROMPT, build_prompt(candidate, resolver, services)
            )
            llm_config = _normalise_llm_config(raw, candidate)
            llm_report = validate_automation(
                llm_config,
                resolver=resolver,
                client=client,
                known_services=services,
                run_check_config=run_check_config,
            )
            result.llm_report = llm_report
            if llm_report.ok:
                result.config = llm_config
                result.yaml_text = render_yaml(llm_config)
                result.source = "llm"
                result.report = llm_report
                result.notes.append(f"Generated by {provider.name}, passed the validation gate.")
                return result
            result.source = "llm-rejected"
            result.notes.append(
                "LLM output was REJECTED by the validation gate: "
                + "; ".join(llm_report.errors[:5])
            )
            if llm_report.unknown_entities:
                result.notes.append(
                    "Hallucinated entity ids: " + ", ".join(llm_report.unknown_entities[:5])
                )
        except LLMError as err:
            result.llm_error = str(err)
            result.notes.append(f"LLM unavailable ({err}); used deterministic rendering.")

    if resolver is not None and not (
        resolver.sources.get("registry") or resolver.sources.get("states")
    ):
        result.notes.append(
            "Neither the entity registry nor /api/states could be read, so entity references "
            "cannot be verified and this rule cannot be applied automatically. Copy the YAML "
            "manually, or fix Home Assistant API access for the add-on."
        )

    result.config = blueprint_config
    result.yaml_text = render_yaml(blueprint_config)
    if result.source != "llm-rejected":
        result.source = "blueprint"
    result.report = validate_automation(
        blueprint_config,
        resolver=resolver,
        client=client,
        known_services=services,
        run_check_config=run_check_config,
    )
    if result.source == "blueprint" and not provider.enabled:
        result.notes.append(
            "Rendered deterministically from the mined schema (no LLM configured)."
        )
    return result

"""Mode 2: Downstream LLM Persona Decorator with Zero Simulation Impact (§5.2)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.core.types import DecisionTrace
from autogen.intelligence.village.persona.symbolic_dialogue import (
    SymbolicDialogueEngine,
    VillageDialogueEvent,
)


class PersonaProfile(BaseModel):
    """Narrative persona metadata for a villager (strictly downstream of decision engine)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    name: str
    role: str
    trait_summary: str = "pragmatic villager"
    risk_tolerance: float = Field(default=0.25, ge=0.0, le=1.0)


def _read_dotenv_configs(dotenv_path: str | Path = ".env") -> dict[str, str]:
    """Reads API keys and endpoints from environment variables or local .env file."""
    keys = [
        "NEMOTRON_API_KEY",
        "NVIDIA_API_KEY",
        "NEMOTRON_BASE_URL",
        "NEMOTRON_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
    ]
    configs: dict[str, str] = {}
    for k in keys:
        val = os.environ.get(k)
        if val:
            configs[k] = val

    path = Path(dotenv_path)
    if path.is_file():
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k in keys and v and k not in configs:
                    configs[k] = v
        except OSError:
            pass
    return configs


class LLMPersonaBridge:
    """Dual-mode narrative bridge translating DecisionTraces into villager dialogue (§5.2).

    Strictly downstream of the intelligence engine:
    - Never mutates world state, coordinates, balances, or chosen actions.
    - Seamlessly queries NVIDIA Nemotron (or other OpenAI-compatible endpoints) using urllib.
    - Falls back safely to SymbolicDialogueEngine when no API key is configured or on timeout/error.
    """

    def __init__(
        self,
        symbolic_engine: SymbolicDialogueEngine | None = None,
        enable_external_llm: bool = False,
        custom_completion_fn: Callable[[str], str] | None = None,
        dotenv_path: str | Path = ".env",
    ) -> None:
        self.symbolic_engine = symbolic_engine or SymbolicDialogueEngine()
        self.enable_external_llm = enable_external_llm
        self.custom_completion_fn = custom_completion_fn
        self.configs = _read_dotenv_configs(dotenv_path)

        self.nemotron_key = self.configs.get("NEMOTRON_API_KEY") or self.configs.get("NVIDIA_API_KEY")
        self.nemotron_base_url = self.configs.get(
            "NEMOTRON_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.nemotron_model = self.configs.get(
            "NEMOTRON_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
        )

        self.openai_key = self.configs.get("OPENAI_API_KEY")
        self.gemini_key = self.configs.get("GEMINI_API_KEY")

        # Auto-activate if a valid key was found and enable_external_llm wasn't explicitly disabled
        if self.has_valid_api_key and not enable_external_llm:
            self.enable_external_llm = True

    @property
    def has_valid_api_key(self) -> bool:
        """Returns True if any supported LLM API key is present."""
        return bool(self.nemotron_key or self.openai_key or self.gemini_key or self.custom_completion_fn)

    @property
    def active_provider_name(self) -> str:
        """Human-readable description of the active completion backend."""
        if self.custom_completion_fn:
            return "Custom Callback"
        if self.nemotron_key:
            return f"NVIDIA Nemotron ({self.nemotron_model.split('/')[-1]})"
        if self.openai_key:
            return "OpenAI"
        if self.gemini_key:
            return "Gemini"
        return "Symbolic Rule Engine (Offline)"

    @property
    def is_llm_active(self) -> bool:
        """Returns True if an LLM provider or completion hook is active."""
        if not self.enable_external_llm:
            return False
        return self.has_valid_api_key

    def _call_nemotron_api(self, prompt: str) -> str:
        """Dispatches an OpenAI-compatible completion request to the NVIDIA NIM endpoint."""
        return self._call_llm_with_fallback(prompt)

    def _call_llm_with_fallback(self, prompt: str) -> str:
        """Dispatches an OpenAI-compatible completion request to NVIDIA NIM.

        Falls back gracefully from reasoning model (e.g. lightning) to fast instruction
        model (e.g. bolt-e4b) if the primary model returns empty content or triggers 429.
        """
        if self.custom_completion_fn is not None:
            return self.custom_completion_fn(prompt).strip()

        key = self.nemotron_key or self.openai_key
        if not key:
            return ""

        base_url = self.nemotron_base_url.rstrip("/")
        url = f"{base_url}/chat/completions"

        # Cascade through candidate models
        models_to_try = [self.nemotron_model]
        if "bolt" not in self.nemotron_model.lower():
            models_to_try.append("nvidia/nemotron-3.5-bolt-e4b")
        if "nano" not in self.nemotron_model.lower():
            models_to_try.append("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

        for model in models_to_try:
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.65,
                    "max_tokens": 512,
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key}",
                        "User-Agent": "AUTOGEN-AI-Village/1.0",
                    },
                )
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    if resp.status == 200:
                        body = json.loads(resp.read().decode("utf-8"))
                        choices = body.get("choices", [])
                        if choices:
                            msg = choices[0].get("message", {})
                            content = msg.get("content")
                            if content and str(content).strip():
                                return str(content).strip()
            except Exception:
                continue

        return ""

    def _make_symbolic_villager_dialogue(
        self, name: str, role: str, action: str, recipient_name: str | None
    ) -> str:
        """Deterministic in-character natural speech fallback (never dry telemetry strings)."""
        recip = f"{recipient_name}, " if recipient_name and recipient_name != "Self" else ""
        act_upper = action.upper()
        if "STEAL" in act_upper:
            return "Quiet now... hunger leaves me no choice but to take what the granary hoards."
        if act_upper == "TITHE_GRANARY":
            return f"{recip}tithing surplus wheat to the commons so none go hungry."
        if "TRADE" in act_upper or "MARKET" in act_upper:
            return f"{recip}fair weights and honest coin at my stall today."
        if act_upper == "DRAW_WATER":
            return f"{recip}the town well runs sweet and cool this morning."
        if "HARVEST" in act_upper:
            return f"{recip}the golden stalks are heavy and ready for harvest."
        if "FORAGE" in act_upper:
            return f"{recip}gathering timber and mountain herbs along the trail."
        if "GUARD" in act_upper or "PATROL" in act_upper:
            return f"{recip}granary chests and village roads stand watched and secure."
        if act_upper == "REST_HUT":
            return f"{recip}locking my chest and resting after an honest day's toil."
        if recip:
            return f"{recip}good day! May fortune favor our village harvest."
        return f"Tending to my duties as the village {role.lower()}."

    def generate_village_dialogues_batch(
        self, agents_context: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Generates natural in-character spoken dialogue for all active villagers in a single batch call.

        Returns a dictionary mapping agent_id -> spoken line for all 10 villagers.
        Respects NVIDIA NIM concurrency constraints by issuing ONE prompt per simulation step.
        """
        fallback_map: dict[str, str] = {}
        for a in agents_context:
            aid = a["agent_id"]
            name = a["name"]
            role = a["role"]
            recip = a.get("recipient_name")
            act = a.get("action", "STAY")
            fallback_map[aid] = self._make_symbolic_villager_dialogue(name, role, act, recip)

        if not self.is_llm_active or not agents_context:
            return fallback_map

        scene_lines: list[str] = []
        for a in agents_context:
            aid = a["agent_id"]
            name = a["name"]
            role = a["role"]
            act_desc = a.get("action_desc", "attending to village tasks")
            recip = a.get("recipient_name")
            peer_clause = f" (conversing with {recip})" if recip else ""
            scene_lines.append(f"- {aid}: {name} the {role}, {act_desc}{peer_clause}")

        prompt = (
            "You are the master bard and narrative director of a living medieval village simulation.\n"
            "Here are the 10 villagers currently active in the village:\n"
            + "\n".join(scene_lines)
            + "\n\nFor EACH villager, write exactly ONE short, flavorful, natural in-character spoken dialogue line "
            "(under 12 words) conversing with their peer or neighbor.\n"
            "Return strictly a JSON object mapping each agent_id to their spoken line:\n"
            '{\n  "villager_arun": "...",\n  "villager_meera": "...",\n  ...\n}\n'
            "Respond ONLY with valid JSON. No markdown backticks, no explanations."
        )

        try:
            raw_response = self._call_llm_with_fallback(prompt)
            if raw_response:
                # If custom completion function or single-line mock returned plain text
                if not raw_response.startswith("{") and "{" not in raw_response:
                    return {a["agent_id"]: raw_response for a in agents_context}
                cleaned = re.sub(r"^```(?:json)?\s*", "", raw_response.strip(), flags=re.MULTILINE)
                cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    result = dict(fallback_map)
                    for aid, line in parsed.items():
                        if aid in fallback_map and isinstance(line, str) and line.strip():
                            result[aid] = line.strip().strip('"').strip("'")
                    return result
        except Exception:
            pass

        return fallback_map

    def build_prompt(
        self,
        persona: PersonaProfile,
        trace: DecisionTrace,
        recipient_name: str | None,
        trust_score: float,
        recent_memories: list[str],
    ) -> str:
        """Constructs an evocative in-character prompt for NVIDIA Nemotron."""
        v_act = trace.rationale.get("village_action", "STAY")
        action_desc = v_act.replace("_", " ").lower()
        mem_text = recent_memories[0] if recent_memories else f"Decided to {action_desc}"
        target = (
            f"to {recipient_name}"
            if recipient_name and recipient_name != "Self"
            else "aloud to themselves or a passerby"
        )
        return (
            f"You are {persona.name}, a {persona.role} in a medieval village society.\n"
            f"Action: You just chose to {action_desc}. Context: {mem_text}.\n"
            f"Speaking {target}, say one brief in-character line (under 15 words). No quotes, no explanations."
        )

    def decorate_decision(
        self,
        persona: PersonaProfile,
        trace: DecisionTrace,
        position: tuple[int, int],
        hunger: float,
        hydration: float,
        coins: float,
        wheat_qty: float,
        recipient_id: str | None = None,
        recipient_name: str | None = None,
        trust_toward_recipient: float = 0.5,
        witnessed_offender_id: str | None = None,
        witnessed_offender_name: str | None = None,
        recent_memories: list[str] | None = None,
        force_symbolic: bool = False,
    ) -> VillageDialogueEvent:
        """Translates a completed DecisionTrace into a VillageDialogueEvent.

        Always produces a valid event; falls back to Mode 1 SymbolicDialogueEngine
        if LLM mode is disabled, force_symbolic is requested, or if an exception occurs.
        """
        base_event = self.symbolic_engine.generate_utterance(
            trace=trace,
            speaker_name=persona.name,
            speaker_role=persona.role,
            position=position,
            hunger=hunger,
            hydration=hydration,
            coins=coins,
            wheat_qty=wheat_qty,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            trust_toward_recipient=trust_toward_recipient,
            witnessed_offender_id=witnessed_offender_id,
            witnessed_offender_name=witnessed_offender_name,
        )

        if not self.is_llm_active or force_symbolic:
            return base_event

        try:
            prompt = self.build_prompt(
                persona=persona,
                trace=trace,
                recipient_name=recipient_name,
                trust_score=trust_toward_recipient,
                recent_memories=recent_memories or [],
            )
            llm_text = ""
            if self.custom_completion_fn is not None:
                llm_text = self.custom_completion_fn(prompt).strip()
            elif self.nemotron_key or self.openai_key:
                llm_text = self._call_nemotron_api(prompt).strip()

            if not llm_text:
                return base_event
            mode_name = "nemotron_llm" if self.nemotron_key else "llm_decorator"
            return base_event.model_copy(update={"utterance": llm_text, "mode": mode_name})
        except Exception:
            # Strict resilience invariant: never break simulation loop on LLM failure
            return base_event

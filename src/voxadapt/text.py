from __future__ import annotations

import re
from typing import Protocol

from .models import Channel, RewriteMode, UserProfile


class TextPolisher(Protocol):
    def polish(
        self,
        text: str,
        *,
        mode: RewriteMode,
        channel: Channel,
        profile: UserProfile,
    ) -> str: ...


_FILLERS = re.compile(r"\b(?:um+|uh+|erm+|you know|like)\b[ ,]*", re.IGNORECASE)
_REPEATED_WORD = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;!?])")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def disfluency_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    fillers = _FILLERS.findall(text)
    repeats = _REPEATED_WORD.findall(text)
    return min(1.0, (len(fillers) + len(repeats)) / len(words))


class RuleBasedPolisher:
    """Deterministic baseline used in tests and benchmark comparisons.

    It deliberately performs conservative edits. A learned model can replace this
    component without changing the API, personalization store, or evaluation harness.
    """

    def polish(
        self,
        text: str,
        *,
        mode: RewriteMode,
        channel: Channel,
        profile: UserProfile,
    ) -> str:
        output = text.strip()
        output = _REPEATED_WORD.sub(r"\1", output)
        if mode is not RewriteMode.VERBATIM:
            output = _FILLERS.sub("", output)
        output = _SPACE_BEFORE_PUNCT.sub(r"\1", output)
        output = _MULTISPACE.sub(" ", output).strip()

        for source, target in sorted(
            profile.replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            output = re.sub(rf"\b{re.escape(source)}\b", target, output, flags=re.IGNORECASE)

        if output:
            output = output[0].upper() + output[1:]

        preferences = profile.channel_preferences.get(channel.value, {})
        wants_terminal_punctuation = preferences.get(
            "terminal_punctuation", channel is not Channel.CHAT
        )
        if wants_terminal_punctuation and output and output[-1] not in ".!?":
            output += "."
        if channel is Channel.CHAT and preferences.get("lowercase", False):
            output = output.lower()
        return output


class TransformersPolisher:
    """Lazy Hugging Face adapter for a base or LoRA-merged seq2seq model."""

    def __init__(self, model_name_or_path: str, max_new_tokens: int = 192) -> None:
        try:
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install VoxAdapt with the 'ml' extra") from exc
        model_path = str(model_name_or_path)
        try:
            adapter_config = PeftConfig.from_pretrained(model_path)
        except (OSError, ValueError):
            self._pipeline = pipeline("text2text-generation", model=model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(adapter_config.base_model_name_or_path)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                adapter_config.base_model_name_or_path
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            self._pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        self._max_new_tokens = max_new_tokens
        self._profile_formatter = RuleBasedPolisher()

    def polish(
        self,
        text: str,
        *,
        mode: RewriteMode,
        channel: Channel,
        profile: UserProfile,
    ) -> str:
        prompt = f"Polish spoken text: {text}"
        result = self._pipeline(prompt, max_new_tokens=self._max_new_tokens, do_sample=False)
        generated = str(result[0]["generated_text"]).strip()
        return self._profile_formatter.polish(
            generated,
            mode=RewriteMode.VERBATIM,
            channel=channel,
            profile=profile,
        )

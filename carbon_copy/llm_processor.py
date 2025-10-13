from __future__ import annotations

import datetime as dt
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from .models import (
    CarbonCopyResult,
    LabelReviewRequest,
    LabelSuggestion,
)
from .review import collect_label_review_requests

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

@dataclass(slots=True)
class UsageEvent:
    kind: str
    model: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    estimated_cost_usd: Optional[float]
    note: str = ""


class LabelLLMProcessor:
    """Utility to normalize suspect labels via an OpenAI model."""

    def __init__(self, client: Optional["OpenAI"] = None, *, allow_mock: bool = False) -> None:
        if client is not None:
            self._client = client
        else:
            if OpenAI is None:
                if allow_mock:
                    self._client = None
                else:
                    raise RuntimeError(
                        "OpenAI client not available. Install the 'openai' package and set OPENAI_API_KEY."
                    )
            else:
                api_key_present = bool(os.getenv("OPENAI_API_KEY"))
                if allow_mock and not api_key_present:
                    self._client = None
                elif api_key_present:
                    self._client = OpenAI()
                else:
                    raise RuntimeError(
                        "OPENAI_API_KEY is not set; rerun with --llm-dry-run or configure the API key."
                    )
        self._usage_events: List[UsageEvent] = []
        self._allow_mock = allow_mock

    # ------------------------------------------------------------------ public --
    def review_result(
        self,
        result: CarbonCopyResult,
        *,
        model: str = os.getenv("OPENAI_LABEL_MODEL", "gpt-4.1-mini"),
        temperature: float = 0.0,
        max_attempts: int = 3,
        retry_base_delay: float = 4.0,
        dry_run: bool = False,
    ) -> List[LabelSuggestion]:
        """Run the LLM over all suspect labels in a `CarbonCopyResult`.

        Returns a list of label suggestions; optional caller processing can then
        accept/reject and apply the results.
        """
        requests = collect_label_review_requests(result)
        suggestions: List[LabelSuggestion] = []
        for request in requests:
            suggestion = (
                self._mock_suggestion(request)
                if dry_run
                else self._process_request(
                    request,
                    model=model,
                    temperature=temperature,
                    max_attempts=max_attempts,
                    retry_base_delay=retry_base_delay,
                )
            )
            if suggestion is not None:
                suggestions.append(suggestion)
        return suggestions

    def summarize_usage(
        self,
        *,
        pdf_name: str,
        output_dir: str = "out",
    ) -> Dict[str, object]:
        """Persist a usage summary (JSON + CSV) similar to the original processor."""
        os.makedirs(output_dir, exist_ok=True)
        total_in = self._sum_usage(lambda event: event.input_tokens or 0)
        total_out = self._sum_usage(lambda event: event.output_tokens or 0)
        total_tokens = self._sum_usage(lambda event: event.total_tokens or 0)
        total_cost: Optional[float]
        costs = [event.estimated_cost_usd for event in self._usage_events if event.estimated_cost_usd is not None]
        total_cost = sum(costs) if len(costs) == len(self._usage_events) and costs else None

        bundle = {
            "file": pdf_name,
            "events": [event.__dict__ for event in self._usage_events],
            "totals": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_tokens,
                "estimated_cost_usd": total_cost,
            },
        }

        stem = os.path.splitext(os.path.basename(pdf_name))[0]
        with open(os.path.join(output_dir, f"{stem}.usage.json"), "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=2)

        csv_path = os.path.join(output_dir, "usage_summary.csv")
        write_header = not os.path.exists(csv_path)
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        models = ",".join(sorted(set(event.model for event in self._usage_events)))
        with open(csv_path, "a", encoding="utf-8") as handle:
            if write_header:
                handle.write("timestamp,file,events,models,input_tokens,output_tokens,total_tokens,estimated_cost_usd\n")
            handle.write(
                f"{timestamp},{pdf_name},{len(self._usage_events)},{models},"
                f"{total_in},{total_out},{total_tokens},{'' if total_cost is None else total_cost}\n"
            )
        return bundle

    # -------------------------------------------------------------- internals --
    def _process_request(
        self,
        request: LabelReviewRequest,
        *,
        model: str,
        temperature: float,
        max_attempts: int,
        retry_base_delay: float,
    ) -> Optional[LabelSuggestion]:
        if self._client is None:
            raise RuntimeError(
                "OpenAI client not available. Install the 'openai' package or run in --llm-dry-run mode."
            )
        attempts = max(1, max_attempts)
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                    model=model,
                    messages=self._build_messages(request),
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                self._record_usage("label_normalization", model, response.usage)
                payload = self._parse_json_content(response.choices[0].message.content or "")
                return self._payload_to_suggestion(request, payload)
            except Exception as exc:  # pragma: no cover - depends on network
                last_error = exc
                if attempt == attempts - 1:
                    note = f"failure: {exc}"
                    self._usage_events.append(
                        UsageEvent(
                            kind="label_normalization",
                            model=model,
                            input_tokens=None,
                            output_tokens=None,
                            total_tokens=None,
                            estimated_cost_usd=None,
                            note=note,
                        )
                    )
                    return None
                delay = self._compute_retry_delay(exc, retry_base_delay, attempt)
                time.sleep(delay)
        if last_error:
            raise last_error
        return None

    def _mock_suggestion(self, request: LabelReviewRequest) -> LabelSuggestion:
        justification = "dry_run"
        return LabelSuggestion(
            statement=request.statement,
            row_index=request.row_index,
            normalized_label=request.cleaned_label,
            justification=justification,
            confidence=None,
        )

    def _build_messages(self, request: LabelReviewRequest) -> List[Dict[str, str]]:
        system = (
            "You are assisting with financial statement data normalization. "
            "Given a line item label and its context, produce a canonical accounting label. "
            "Return a JSON object with keys: normalized_label (string), "
            "justification (string explanation), and confidence (0-1 number). "
            "If the current label already looks correct, return it unchanged."
        )
        user_payload = {
            "statement_type": request.statement.value,
            "current_label": request.cleaned_label,
            "raw_label": request.raw_label,
            "extraction_flags": request.reasons,
            "values": {
                column: value
                for column, value in request.values.items()
                if value is not None
            },
        }
        user_text = (
            "Normalize the following financial statement label:\n"
            f"{json.dumps(user_payload, indent=2)}\n"
            "Respond ONLY with a JSON object: "
            '{"normalized_label": "...", "justification": "...", "confidence": 0.0}'
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

    def _parse_json_content(self, text: str) -> Dict[str, object]:
        try:
            content = text.strip()
            if not content:
                raise ValueError("Empty response content.")
            return json.loads(content)
        except json.JSONDecodeError:
            return self._parse_first_json_fragment(text)

    def _payload_to_suggestion(
        self,
        request: LabelReviewRequest,
        payload: Dict[str, object],
    ) -> Optional[LabelSuggestion]:
        label = str(payload.get("normalized_label") or "").strip()
        if not label:
            return None
        justification = str(payload.get("justification") or "").strip()
        confidence = payload.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None
        return LabelSuggestion(
            statement=request.statement,
            row_index=request.row_index,
            normalized_label=label,
            justification=justification or "model",
            confidence=confidence if isinstance(confidence, float) else None,
        )

    def _record_usage(
        self,
        kind: str,
        model: str,
        usage: Optional[object],
    ) -> None:
        if usage is None:
            self._usage_events.append(
                UsageEvent(kind=kind, model=model, input_tokens=None, output_tokens=None, total_tokens=None, estimated_cost_usd=None)
            )
            return
        input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        cost = self._estimate_cost(model, input_tokens, output_tokens)
        event = UsageEvent(
            kind=kind,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
        )
        self._usage_events.append(event)

    def _estimate_cost(
        self,
        model: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> Optional[float]:
        if input_tokens is None or output_tokens is None:
            return None
        safe_key = "".join(char if char.isalnum() else "_" for char in model).upper()
        price_in = os.getenv(f"OPENAI_PRICE_INPUT_PER_1M_{safe_key}")
        price_out = os.getenv(f"OPENAI_PRICE_OUTPUT_PER_1M_{safe_key}")
        if not price_in or not price_out:
            return None
        try:
            input_price = float(price_in)
            output_price = float(price_out)
        except ValueError:
            return None
        return (input_tokens / 1_000_000.0) * input_price + (output_tokens / 1_000_000.0) * output_price

    def _compute_retry_delay(self, error: Exception, base_delay: float, attempt: int) -> float:
        delay = base_delay * (2 ** attempt)
        delay += random.uniform(0, min(0.5, base_delay))
        message = str(error).lower()
        parsed = self._parse_wait_seconds(message)
        if parsed is not None:
            delay = parsed + random.uniform(0, 0.25)
        return delay

    @staticmethod
    def _parse_wait_seconds(message: str) -> Optional[float]:
        import re

        match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", message)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_first_json_fragment(text: str) -> Dict[str, object]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("No JSON object found in model output.")
        fragment = text[start : end + 1]
        return json.loads(fragment)

    def _sum_usage(self, extractor: Callable[[UsageEvent], int]) -> int:
        return sum(extractor(event) for event in self._usage_events)


def apply_label_suggestions(
    result: CarbonCopyResult,
    suggestions: Iterable[LabelSuggestion],
    *,
    min_confidence: float = 0.0,
) -> None:
    """Apply accepted label suggestions in-place to the provided result."""
    for suggestion in suggestions:
        if suggestion.confidence is not None and suggestion.confidence < min_confidence:
            continue
        block = result.blocks.get(suggestion.statement)
        if block is None:
            continue
        if suggestion.row_index < 0 or suggestion.row_index >= len(block.wide_table):
            continue
        block.wide_table.at[suggestion.row_index, "Line Item (as printed)"] = suggestion.normalized_label
        suspect = block.suspect_labels.get(suggestion.row_index)
        if suspect is not None:
            suspect.reasons.append("llm_normalized")
            block.suspect_labels[suggestion.row_index] = suspect

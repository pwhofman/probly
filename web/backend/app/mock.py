"""Scripted chat backend used for demo/mock mode.

Mirrors the duck-typed interface of :class:`app.gemma.GemmaChat` so the
route handlers in :mod:`app.routes.chat` can use either class without
changes. Enabled by setting ``MOCK_MODE=1`` before starting the server.

The scripted content — text, per-word confidences, concept spans,
per-reply "full" confidence, and alternative-word suggestions — lives
in the sidecar ``mock_script.json`` file next to this module rather
than as Python literals, so hand-edits don't require touching code.
The file is loaded and validated once at import time; any mismatch
between a reply's word count and its parallel confidence arrays raises
:class:`_ConfidenceShapeError` immediately so typos fail loudly
instead of bleeding into the demo.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# Simulated "first token" latency before streaming starts. The frontend
# buffers incoming deltas for up to ~0.6s while showing its "Thinking..."
# state (see web/frontend/src/components/ChatWindow.tsx). If deltas arrive
# during that window they all get flushed at once when the timer fires,
# killing the typing animation. Sleeping longer than the max thinking
# delay here guarantees the first chunk lands after the UI has flipped
# to the streaming state.
_INITIAL_DELAY_SECONDS = 0.8

# Per-chunk sleep between streamed pieces of a reply. Slow enough that
# the UI's typing animation is visible word-by-word, matching the feel
# of Gemma's TextIteratorStreamer in real mode.
_CHUNK_DELAY_SECONDS = 0.02
_MODEL_UNAVAILABLE_MESSAGE = "Model not available."


def _chunk_reply(text: str) -> list[str]:
    """Split a reply into word-sized chunks with trailing whitespace preserved."""
    # Keep whitespace attached to the preceding word so reassembling the
    # chunks on the client reproduces the original string exactly.
    return re.findall(r"\S+\s*", text) or [text]


@dataclass(frozen=True)
class ConceptSpan:
    """A multi-word concept span with its own hand-authored confidence.

    ``first_word`` and ``last_word`` are inclusive indices into the word
    list produced by :func:`_chunk_reply`. The frontend's "concept" display
    mode tints every word in ``[first_word, last_word]`` — including the
    whitespace between them — with ``confidence``.
    """

    first_word: int
    last_word: int
    confidence: float


@dataclass(frozen=True)
class _Reply:
    """A single scripted assistant reply plus its confidence/alternatives data.

    Populated by :func:`_load_demos` from the JSON sidecar. All parallel
    lists are validated to be length-consistent with
    ``_chunk_reply(text)`` before this dataclass is constructed, so
    downstream code can zip them with ``strict=True`` safely.
    """

    text: str
    word_confidences: list[float]
    concepts: list[ConceptSpan]
    full_confidence: float
    word_alternatives: list[list[str] | None]
    low_confidence: bool
    # Mock-only alternative reply text. When set, the frontend's action
    # row surfaces a small toggle button next to the Uncertainty button
    # that swaps the rendered message body between ``text`` and this
    # string. Real Gemma never produces this field.
    regenerate: str | None


class _ConfidenceShapeError(ValueError):
    """Raised when the scripted data in the JSON sidecar is malformed."""


_SCRIPT_PATH = Path(__file__).parent / "mock_script.json"


def _check_word_confidences(where: str, word_count: int, values: list[float]) -> None:
    if len(values) != word_count:
        msg = f"{where}: word_confidences has {len(values)} entries but text chunks into {word_count} words"
        raise _ConfidenceShapeError(msg)
    for j, value in enumerate(values):
        if not 0.0 <= value <= 1.0:
            msg = f"{where}: word_confidences[{j}] = {value} is outside [0, 1]"
            raise _ConfidenceShapeError(msg)


def _check_word_alternatives(
    where: str,
    word_count: int,
    alternatives: list[list[str] | None],
) -> None:
    if len(alternatives) != word_count:
        msg = f"{where}: word_alternatives has {len(alternatives)} entries but text chunks into {word_count} words"
        raise _ConfidenceShapeError(msg)
    for j, slot in enumerate(alternatives):
        if slot is None:
            continue
        if not slot:
            msg = f"{where}: word_alternatives[{j}] is an empty list; use null to mean 'no alternatives'"
            raise _ConfidenceShapeError(msg)
        for k, alt in enumerate(slot):
            if not isinstance(alt, str) or not alt:
                msg = f"{where}: word_alternatives[{j}][{k}] = {alt!r} is not a non-empty string"
                raise _ConfidenceShapeError(msg)


def _check_concepts(where: str, word_count: int, concepts: list[dict[str, Any]]) -> None:
    for j, span in enumerate(concepts):
        first_word = span["first_word"]
        last_word = span["last_word"]
        confidence = span["confidence"]
        if not 0 <= first_word <= last_word < word_count:
            msg = f"{where}: concepts[{j}] span [{first_word}, {last_word}] is out of range for {word_count} words"
            raise _ConfidenceShapeError(msg)
        if not 0.0 <= confidence <= 1.0:
            msg = f"{where}: concepts[{j}].confidence = {confidence} is outside [0, 1]"
            raise _ConfidenceShapeError(msg)


def _check_full_confidence(where: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        msg = f"{where}: full_confidence = {value} is outside [0, 1]"
        raise _ConfidenceShapeError(msg)


def _check_low_confidence(where: str, value: object) -> None:
    # ``isinstance(..., bool)`` rejects the JSON numeric 0/1 that would
    # otherwise sneak through because ``bool`` is a subclass of ``int``.
    if not isinstance(value, bool):
        msg = f"{where}: low_confidence = {value!r} must be a bool"
        raise _ConfidenceShapeError(msg)


def _validate_reply(
    demo_idx: int,
    reply_idx: int,
    chunks: list[str],
    word_confidences: list[float],
    word_alternatives: list[list[str] | None],
    concepts_raw: list[dict[str, Any]],
    full_confidence: float,
    low_confidence: object,
) -> None:
    """Fail loudly if any array in the parsed reply is out of sync."""
    where = f"demo {demo_idx} reply {reply_idx}"
    word_count = len(chunks)
    _check_word_confidences(where, word_count, word_confidences)
    _check_word_alternatives(where, word_count, word_alternatives)
    _check_concepts(where, word_count, concepts_raw)
    _check_full_confidence(where, full_confidence)
    _check_low_confidence(where, low_confidence)


def _load_demos() -> list[list[_Reply]]:
    """Parse the JSON sidecar into a list of demos, each a list of replies."""
    raw = json.loads(_SCRIPT_PATH.read_text())
    demos: list[list[_Reply]] = []
    for d, demo in enumerate(raw):
        replies: list[_Reply] = []
        for r, reply in enumerate(demo["replies"]):
            text = reply["text"]
            chunks = _chunk_reply(text)
            word_confidences = reply["word_confidences"]
            word_alternatives = reply["word_alternatives"]
            concepts_raw = reply["concepts"]
            full_confidence = reply["full_confidence"]
            low_confidence = reply["low_confidence"]
            regenerate = reply.get("regenerate")
            if regenerate is not None and not (isinstance(regenerate, str) and regenerate):
                msg = f"demo {d} reply {r}: regenerate must be a non-empty string or omitted"
                raise _ConfidenceShapeError(msg)

            _validate_reply(
                d,
                r,
                chunks,
                word_confidences,
                word_alternatives,
                concepts_raw,
                full_confidence,
                low_confidence,
            )

            concepts = [
                ConceptSpan(
                    first_word=c["first_word"],
                    last_word=c["last_word"],
                    confidence=c["confidence"],
                )
                for c in concepts_raw
            ]
            replies.append(
                _Reply(
                    text=text,
                    word_confidences=list(word_confidences),
                    concepts=concepts,
                    full_confidence=full_confidence,
                    word_alternatives=list(word_alternatives),
                    low_confidence=low_confidence,
                    regenerate=regenerate,
                )
            )
        demos.append(replies)
    return demos


# Load and validate at import time so server startup fails loudly on
# any malformed hand edit rather than surfacing the problem mid-demo.
_DEMOS: list[list[_Reply]] = _load_demos()


class MockChat:
    """Scripted replacement for :class:`app.gemma.GemmaChat`.

    Tracks a demo cursor across chat resets so a live walkthrough can
    flip through multiple scripted conversations by clicking the
    sidebar's "New chat" button. Each fresh chat (``user_turns == 1``)
    that arrives after a reply has already been served in the current
    demo advances the cursor to the next demo, clamped to the last
    entry so the demo never crashes once the script is exhausted.

    A single :class:`MockChat` instance lives on ``app.state.gemma``
    and is shared across request threads, so the demo cursor is
    guarded by a ``threading.Lock``.
    """

    def __init__(self, *, model_unavailable: bool = False) -> None:
        """Start the cursor at demo 0 with no reply served yet."""
        self._model_unavailable = model_unavailable
        self._demo_index = 0
        self._demo_started = False
        self._lock = threading.Lock()
        # Memoize the last ``_select`` result by the ``id`` of the messages
        # list so repeated calls within a single route handler (see
        # ``reply_stream`` + ``reply_confidence`` in
        # ``web/backend/app/routes/chat.py``) return the same reply without
        # advancing the demo cursor a second time.
        self._cached_messages_id: int | None = None
        self._cached_reply: _Reply | None = None

    def _select(self, messages: list[dict[str, str]]) -> _Reply:
        """Return the scripted reply for the current chat history.

        Advances the demo cursor exactly once when a fresh chat
        (``user_turns <= 1``) arrives after the current demo has
        already served at least one reply. Multi-turn follow-ups
        (``user_turns > 1``) stay within the current demo and just
        pick the next reply in it.

        The route handler calls ``reply_stream`` and ``reply_confidence``
        on the same ``messages`` list within a single request, so
        ``_select`` memoizes on ``id(messages)`` to avoid advancing the
        cursor twice for what is logically one turn.
        """
        messages_id = id(messages)
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        with self._lock:
            if messages_id == self._cached_messages_id and self._cached_reply is not None:
                return self._cached_reply
            if user_turns <= 1 and self._demo_started:
                self._demo_index = min(self._demo_index + 1, len(_DEMOS) - 1)
            self._demo_started = True
            replies = _DEMOS[self._demo_index]
            reply_idx = max(0, min(user_turns - 1, len(replies) - 1))
            chosen = replies[reply_idx]
            self._cached_messages_id = messages_id
            self._cached_reply = chosen
            return chosen

    def reply(self, messages: list[dict[str, str]]) -> str:
        """Return the full scripted reply for the current chat history."""
        if self._model_unavailable:
            return _MODEL_UNAVAILABLE_MESSAGE
        return self._select(messages).text

    def reply_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield the scripted reply one word-sized chunk at a time.

        Waits :data:`_INITIAL_DELAY_SECONDS` before the first chunk so the
        frontend's "Thinking..." buffering window (which dumps any
        accumulated deltas at once when it expires) is already over by
        the time deltas start arriving. Subsequent chunks are emitted
        with :data:`_CHUNK_DELAY_SECONDS` between them so the UI shows a
        visible typing animation, matching the feel of Gemma's
        per-token loop in :meth:`app.gemma.GemmaChat.reply_stream`.

        Only the text is streamed here. The per-word, per-concept, and
        full-response confidences for this turn are returned separately
        by :meth:`reply_confidence` once generation has finished.
        """
        if self._model_unavailable:
            yield _MODEL_UNAVAILABLE_MESSAGE
            return
        chosen = self._select(messages)
        chunks = _chunk_reply(chosen.text)
        for i, chunk in enumerate(chunks):
            time.sleep(_INITIAL_DELAY_SECONDS if i == 0 else _CHUNK_DELAY_SECONDS)
            yield chunk

    def reply_confidence(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        """Return the final confidence payload for the selected turn.

        Called by the route handler after :meth:`reply_stream` has been
        fully consumed. The returned dict is serialized as a single
        ``{"confidence": ...}`` NDJSON frame just before ``{"done": true}``,
        and is what drives all three display modes on the frontend.
        """
        if self._model_unavailable:
            return None
        chosen = self._select(messages)
        chunks = _chunk_reply(chosen.text)
        words: list[dict[str, Any]] = []
        for chunk, conf, alts in zip(
            chunks,
            chosen.word_confidences,
            chosen.word_alternatives,
            strict=True,
        ):
            word: dict[str, Any] = {"text": chunk, "confidence": conf}
            # Only attach the field when we actually have suggestions, so the
            # common-case wire payload stays minimal and consumers can treat
            # a missing key as "no alternatives".
            if alts is not None:
                word["alternatives"] = alts
            words.append(word)
        payload: dict[str, Any] = {
            "words": words,
            "concepts": [
                {
                    "first_word": span.first_word,
                    "last_word": span.last_word,
                    "confidence": span.confidence,
                }
                for span in chosen.concepts
            ],
            "full": chosen.full_confidence,
            "low_confidence": chosen.low_confidence,
        }
        # Only include the field when this reply actually carries an
        # alternative, so the common-case wire payload stays minimal and
        # the frontend can treat a missing key as "no regenerate".
        if chosen.regenerate is not None:
            payload["regenerate"] = chosen.regenerate
        return payload

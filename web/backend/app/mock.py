"""Scripted chat backend used for demo/mock mode.

Mirrors the duck-typed interface of :class:`app.gemma.GemmaChat` so the
route handlers in :mod:`app.routes.chat` can use either class without
changes. Enabled by setting ``MOCK_MODE=1`` before starting the server.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# Scripted assistant replies, played back in order for each fresh chat.
# The Nth user message in the incoming request selects ``SCRIPT[N - 1]``;
# once the script is exhausted, the final entry is returned for every
# subsequent turn. Because the turn is derived from the request payload
# rather than from server state, starting a new chat on the frontend
# (empty history, single user message) automatically replays the script
# from the top — no reset call or server restart needed.
SCRIPT: list[str] = [
    # Turn 1 — self-introduction. Uniformly high confidence, so that a
    # viewer flipping on the confidence panel immediately sees a "clean"
    # response. This sets up the biography turn below as a visible
    # contrast once the hallucinations start lighting up.
    "I am Gemma 4, wrapped in the probly harness. How can I help?",
    # Turn 2 — Friedrich Merz biography. Deliberately seeded with seven
    # subtle factual errors so the confidence highlighting has something
    # meaningful to light up:
    #   [3]   "5 "          wrong day       (should be 11 November)
    #   [7]   "Arnsberg, "  wrong birthplace (should be Brilon)
    #   [27]  "Heidelberg " wrong university (Merz studied at Bonn/Marburg)
    #   [90]  "CEO "        wrong title      (Chairman of the supervisory board)
    #   [95]  "2014 "       wrong start year (BlackRock chair from 2016)
    #   [137] "first "      wrong round      (elected on the second round --
    #                                         first chancellor-elect in German
    #                                         history to fail round one)
    #   [155] "four "       wrong count      (Merz has three children)
    # See SCRIPT_CONFIDENCES[1] below for the matching per-token floats.
    (
        "Friedrich Merz (born 5 November 1955 in Arnsberg, North "
        "Rhine-Westphalia) is a German politician and the current Chancellor "
        "of Germany. He studied law at the University of Heidelberg and began "
        "his career as a judge before moving into corporate law. In 1989 he "
        "was elected to the European Parliament, and from 1994 to 2009 he "
        "represented the Hochsauerland constituency in the Bundestag, where "
        "he served as chairman of the CDU/CSU parliamentary group from 2000 "
        "to 2002. After stepping back from politics in 2009, Merz pursued a "
        "business career, serving as CEO of BlackRock Germany from 2014 to "
        "2020. He returned to frontline politics and was elected leader of "
        "the CDU in January 2022. Following the CDU/CSU's victory in the "
        "February 2025 federal election, Merz was elected Chancellor of "
        "Germany on 6 May 2025, taking office in the first round of voting "
        "in the Bundestag. He is married to Charlotte Merz, a judge, and "
        "they have four children."
    ),
    "Last scripted reply — after this, I'll just keep repeating this line so the demo never crashes.",
]

# Simulated "first token" latency before streaming starts. The frontend
# buffers incoming deltas for up to ~2s while showing its "Thinking..."
# state (see web/frontend/src/components/ChatWindow.tsx). If deltas arrive
# during that window they all get flushed at once when the timer fires,
# killing the typing animation. Sleeping longer than the max thinking
# delay here guarantees the first chunk lands after the UI has flipped
# to the streaming state.
_INITIAL_DELAY_SECONDS = 2.2

# Per-chunk sleep between streamed pieces of a reply. Slow enough that
# the UI's typing animation is visible word-by-word, matching the feel
# of Gemma's TextIteratorStreamer in real mode.
_CHUNK_DELAY_SECONDS = 0.08


def _chunk_reply(text: str) -> list[str]:
    """Split a reply into word-sized chunks with trailing whitespace preserved."""
    # Keep whitespace attached to the preceding word so reassembling the
    # chunks on the client reproduces the original string exactly.
    return re.findall(r"\S+\s*", text) or [text]


# Per-token confidence floats parallel to SCRIPT. Each entry is a list of
# floats, one per chunk returned by ``_chunk_reply(SCRIPT[i])``. 0.0 means
# low confidence (rendered as opaque red on the frontend) and 1.0 means
# high confidence (transparent). Length is validated against ``_chunk_reply``
# at import time so a bad hand edit fails loudly instead of silently
# truncating.
SCRIPT_CONFIDENCES: list[list[float]] = [
    # SCRIPT[0] — 13 tokens, uniformly high confidence. The intro is the
    # "clean baseline" that sets up the biography turn below.
    #   0:"I "   1:"am "   2:"Gemma "   3:"4, "   4:"wrapped "   5:"in "
    #   6:"the " 7:"probly " 8:"harness. " 9:"How " 10:"can " 11:"I "
    #   12:"help?"
    [0.95] * 13,
    # SCRIPT[1] — 157 tokens for the Friedrich Merz biography. Three-tier
    # scheme:
    #   - function/connector words stay at ~0.90 (the model never fumbles
    #     on "the", "in", "of", ...);
    #   - fact-heavy prose sits at ~0.50 (realistic hedging on a 2B-class
    #     biographical recall);
    #   - the seven error tokens documented in SCRIPT[1] drop to 0.05-0.15
    #     and their immediate neighbours get partially lowered.
    # The resulting length-weighted mean is ~0.59, which the frontend's
    # "Full Response" mode renders as a visibly pink bubble; switching to
    # "Word Level" makes each individual error pop as a bright red span.
    [
        0.50,
        0.50,
        0.90,
        0.05,
        0.30,
        0.50,
        0.90,
        0.05,
        0.50,
        0.50,  # 0-9
        0.90,
        0.90,
        0.50,
        0.50,
        0.90,
        0.90,
        0.50,
        0.50,
        0.90,
        0.50,  # 10-19
        0.90,
        0.50,
        0.50,
        0.90,
        0.90,
        0.50,
        0.90,
        0.05,
        0.90,
        0.50,  # 20-29
        0.90,
        0.50,
        0.90,
        0.90,
        0.50,
        0.90,
        0.50,
        0.90,
        0.50,
        0.50,  # 30-39
        0.90,
        0.50,
        0.90,
        0.90,
        0.50,
        0.90,
        0.90,
        0.50,
        0.50,
        0.90,  # 40-49
        0.90,
        0.50,
        0.90,
        0.50,
        0.90,
        0.50,
        0.90,
        0.50,
        0.50,
        0.90,  # 50-59
        0.90,
        0.50,
        0.90,
        0.90,
        0.50,
        0.90,
        0.50,
        0.90,
        0.90,
        0.50,  # 60-69
        0.50,
        0.50,
        0.90,
        0.50,
        0.90,
        0.50,
        0.90,
        0.50,
        0.50,
        0.90,  # 70-79
        0.50,
        0.90,
        0.50,
        0.50,
        0.50,
        0.90,
        0.50,
        0.50,
        0.50,
        0.90,  # 80-89
        0.05,
        0.90,
        0.50,
        0.50,
        0.90,
        0.15,
        0.90,
        0.50,
        0.90,
        0.50,  # 90-99
        0.90,
        0.50,
        0.50,
        0.90,
        0.90,
        0.50,
        0.50,
        0.90,
        0.90,
        0.50,  # 100-109
        0.90,
        0.50,
        0.50,
        0.90,
        0.90,
        0.50,
        0.50,
        0.90,
        0.90,
        0.50,  # 110-119
        0.50,
        0.50,
        0.50,
        0.50,
        0.90,
        0.50,
        0.50,
        0.90,
        0.50,
        0.90,  # 120-129
        0.50,
        0.50,
        0.50,
        0.90,
        0.50,
        0.90,
        0.90,
        0.05,
        0.18,
        0.90,  # 130-139
        0.50,
        0.90,
        0.90,
        0.50,
        0.90,
        0.90,
        0.50,
        0.90,
        0.50,
        0.50,  # 140-149
        0.90,
        0.50,
        0.90,
        0.90,
        0.90,
        0.15,
        0.35,  # 150-156
    ],
    # SCRIPT[2] chunks into 17 tokens.
    [
        0.1,
        0.15,
        0.2,
        0.3,
        0.95,
        0.95,
        0.95,
        0.95,
        0.95,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.95,
    ],
]


class _ConfidenceShapeError(ValueError):
    """Raised when SCRIPT_CONFIDENCES is out of sync with SCRIPT at import."""

    def __init__(self, index: int, expected: int, actual: int) -> None:
        super().__init__(
            f"SCRIPT_CONFIDENCES[{index}] has {actual} entries but SCRIPT[{index}] chunks into {expected} tokens",
        )


# Fail fast if a hand edit desyncs the parallel arrays.
for _i, (_reply, _confs) in enumerate(zip(SCRIPT, SCRIPT_CONFIDENCES, strict=True)):
    _expected_len = len(_chunk_reply(_reply))
    _actual_len = len(_confs)
    if _expected_len != _actual_len:
        raise _ConfidenceShapeError(_i, _expected_len, _actual_len)


def _turn_index(messages: list[dict[str, str]]) -> int:
    """Return the script index for the current turn.

    Derived from the number of user messages in the incoming history: the
    1st user turn gets ``0``, the 2nd ``1``, and so on. Clamped to the
    last entry so long demos never crash once the script is exhausted.
    """
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    # Should never happen given _require_user_message in the route, but
    # guard so this helper is safe to call in isolation.
    return max(0, min(user_turns - 1, len(SCRIPT) - 1))


class MockChat:
    """Stateless scripted replacement for :class:`app.gemma.GemmaChat`.

    No per-instance turn counter: replies are selected purely from the
    incoming message history, so a fresh chat on the frontend (empty
    history -> single user message) always replays the script from the
    top, regardless of prior activity on this server process.
    """

    def reply(self, messages: list[dict[str, str]]) -> str:
        """Return the scripted reply for the current chat history."""
        return SCRIPT[_turn_index(messages)]

    def reply_stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[tuple[str, float]]:
        """Yield the scripted reply one ``(chunk, confidence)`` pair at a time.

        Waits :data:`_INITIAL_DELAY_SECONDS` before the first chunk so the
        frontend's "Thinking..." buffering window (which dumps any
        accumulated deltas at once when it expires) is already over by
        the time deltas start arriving. Subsequent chunks are emitted
        with :data:`_CHUNK_DELAY_SECONDS` between them so the UI shows a
        visible typing animation, matching the feel of Gemma's
        per-token loop in :meth:`app.gemma.GemmaChat.reply_stream`.
        """
        index = _turn_index(messages)
        chunks = _chunk_reply(SCRIPT[index])
        confidences = SCRIPT_CONFIDENCES[index]
        for i, (chunk, conf) in enumerate(zip(chunks, confidences, strict=True)):
            time.sleep(_INITIAL_DELAY_SECONDS if i == 0 else _CHUNK_DELAY_SECONDS)
            yield chunk, conf

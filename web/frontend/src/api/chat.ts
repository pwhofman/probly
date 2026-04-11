import type { ConceptSpan, ConfidencePayload, Message, Role, Word } from '../types';

interface ApiChatMessage {
  role: Role;
  content: string;
}

interface ApiChatResponse {
  message: ApiChatMessage;
}

export async function sendChat(messages: Message[]): Promise<ApiChatMessage> {
  const body = {
    messages: messages.map(({ role, content }) => ({ role, content })),
  };
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Chat request failed: ${res.status} ${res.statusText}`);
  }
  const data = (await res.json()) as ApiChatResponse;
  return data.message;
}

// Backend-side concept span shape (snake_case on the wire).
interface WireConceptSpan {
  first_word: number;
  last_word: number;
  confidence: number;
}

interface WireConfidencePayload {
  words: Word[];
  concepts: WireConceptSpan[];
  full: number;
  low_confidence?: boolean;
}

function decodeConfidence(payload: WireConfidencePayload): ConfidencePayload {
  const concepts: ConceptSpan[] = payload.concepts.map((c) => ({
    firstWord: c.first_word,
    lastWord: c.last_word,
    confidence: c.confidence,
  }));
  return {
    words: payload.words,
    concepts,
    full: payload.full,
    lowConfidence: payload.low_confidence,
  };
}

/**
 * Stream an assistant reply from the backend.
 *
 * During generation the server sends a sequence of ``{delta: string}``
 * frames with raw text only; the frontend just appends them to the
 * message body so the typing animation feels the same as before.
 *
 * When generation finishes, the server emits a single
 * ``{confidence: {...}}`` frame with word-, concept-, and line-level
 * confidences (all backend-authored, frontend does no computation),
 * followed by ``{done: true}``. If the backing chat object has no
 * confidence data to report (real Gemma case), the ``confidence`` frame
 * is omitted entirely and ``onConfidence`` is never called.
 *
 * Resolves when the stream ends; rejects on HTTP error or when the
 * backend emits an ``{error: ...}`` line mid-stream.
 */
export async function sendChatStream(
  messages: Message[],
  onDelta: (chunk: string) => void,
  onConfidence: (payload: ConfidencePayload) => void,
): Promise<void> {
  const body = {
    messages: messages.map(({ role, content }) => ({ role, content })),
  };
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    throw new Error(`Chat stream failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newlineIdx = buffer.indexOf('\n');
    while (newlineIdx >= 0) {
      const line = buffer.slice(0, newlineIdx).trim();
      buffer = buffer.slice(newlineIdx + 1);
      if (line) {
        const parsed = JSON.parse(line) as {
          delta?: string;
          confidence?: WireConfidencePayload;
          done?: boolean;
          error?: string;
        };
        if (parsed.error) throw new Error(parsed.error);
        if (parsed.delta !== undefined) {
          onDelta(parsed.delta);
        }
        if (parsed.confidence !== undefined) {
          onConfidence(decodeConfidence(parsed.confidence));
        }
      }
      newlineIdx = buffer.indexOf('\n');
    }
  }
}

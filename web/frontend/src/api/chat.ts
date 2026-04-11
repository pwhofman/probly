import type { Message, Role } from '../types';

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

/**
 * Stream an assistant reply from the backend. ``onDelta`` is called once per
 * chunk as the model produces text, receiving both the text and the model's
 * per-token confidence (top-1 softmax probability for real Gemma, hand-authored
 * floats for MockChat). Resolves when the stream ends; rejects on HTTP error
 * or when the backend emits an ``{"error": ...}`` line mid-stream.
 */
export async function sendChatStream(
  messages: Message[],
  onDelta: (chunk: string, confidence: number) => void,
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
          confidence?: number;
          done?: boolean;
          error?: string;
        };
        if (parsed.error) throw new Error(parsed.error);
        if (parsed.delta !== undefined) {
          // Missing confidence is tolerated so older servers (or partial
          // replay fixtures) don't crash the demo; fall back to 1.0 which
          // renders as "no tint" in the highlight modes.
          onDelta(parsed.delta, parsed.confidence ?? 1.0);
        }
      }
      newlineIdx = buffer.indexOf('\n');
    }
  }
}

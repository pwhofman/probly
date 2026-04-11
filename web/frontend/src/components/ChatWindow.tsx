import { useState } from 'react';
import type { Message, TokenConfidence } from '../types';
import { sendChatStream } from '../api/chat';
import MessageList from './MessageList';
import ChatInput from './ChatInput';

function makeId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

const QUICK_PROMPTS = [
  { label: 'Brainstorm', icon: '\u270E' },
  { label: 'Explain', icon: '\u{1F4D6}' },
  { label: 'Summarize', icon: '\u{1F4DD}' },
  { label: 'Code', icon: '\u003C\u002F\u003E' },
];

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isSending, setIsSending] = useState(false);

  const handleSend = async (text: string) => {
    const userMessage: Message = { id: makeId(), role: 'user', content: text };
    const assistantId = makeId();
    const withUser = [...messages, userMessage];
    // Skip the thinking simulation on the very first user message — it's
    // already blocked on model startup, so adding a fake delay just feels
    // slow. Only follow-up turns get a simulated "Thinking..." pause.
    const isFirstTurn = messages.length === 0;
    const thinkingMs = isFirstTurn ? 0 : Math.floor(Math.random() * 2000);
    const thoughtLabel = thinkingMs < 1000 ? '<1s' : '<2s';

    // Seed an empty assistant bubble up front so deltas can stream into it.
    setMessages([
      ...withUser,
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        thinking: isFirstTurn ? undefined : 'active',
      },
    ]);
    setIsSending(true);

    // Hold back streamed deltas for `thinkingMs` so the bubble shows a
    // "Thinking..." state, then flips to "Thought for <Xs>" once the pause
    // is over. On the first turn we skip the simulation entirely. Deltas
    // (text + confidence pairs) accumulate in ``bufferedTokens`` during the
    // thinking window and flush into ``message.tokens`` when the timer fires.
    const thinkingState: {
      active: boolean;
      bufferedText: string;
      bufferedTokens: TokenConfidence[];
    } = { active: !isFirstTurn, bufferedText: '', bufferedTokens: [] };

    const thinkingTimer = isFirstTurn
      ? null
      : window.setTimeout(() => {
          thinkingState.active = false;
          const flushedText = thinkingState.bufferedText;
          const flushedTokens = thinkingState.bufferedTokens;
          thinkingState.bufferedText = '';
          thinkingState.bufferedTokens = [];
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    thinking: 'done',
                    thoughtLabel,
                    content: m.content + flushedText,
                    tokens: [...(m.tokens ?? []), ...flushedTokens],
                  }
                : m,
            ),
          );
        }, thinkingMs);

    try {
      await sendChatStream(withUser, (delta, confidence) => {
        if (thinkingState.active) {
          thinkingState.bufferedText += delta;
          thinkingState.bufferedTokens.push({ text: delta, confidence });
          return;
        }
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: m.content + delta,
                  tokens: [...(m.tokens ?? []), { text: delta, confidence }],
                }
              : m,
          ),
        );
      });
    } catch (err) {
      if (thinkingTimer !== null) window.clearTimeout(thinkingTimer);
      const detail = err instanceof Error ? err.message : String(err);
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== assistantId),
        { id: makeId(), role: 'system', content: `Error: ${detail}` },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const isEmpty = messages.length === 0;

  if (isEmpty) {
    return (
      <div className="flex h-full flex-col">
        <div className="flex flex-1 items-center justify-center px-6">
          <div className="flex w-full max-w-2xl flex-col items-center">
            <div className="mb-8 flex items-center gap-4">
              <h1 className="font-serif text-4xl font-normal text-ink">
                Chat with Gemma
              </h1>
            </div>

            <div className="w-full">
              <ChatInput onSend={handleSend} disabled={isSending} />
            </div>

            <div className="mt-6 flex flex-wrap justify-center gap-2">
              {QUICK_PROMPTS.map((q) => (
                <button
                  key={q.label}
                  type="button"
                  onClick={() => handleSend(q.label + ' something for me.')}
                  className="flex items-center gap-2 rounded-full border border-rule bg-white px-4 py-2 text-sm text-ink/80 shadow-sm transition-colors hover:bg-canvas"
                >
                  <span className="text-muted">{q.icon}</span>
                  <span>{q.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-4xl">
          <MessageList messages={messages} />
        </div>
      </div>
      <div className="border-t border-rule/60 bg-canvas px-6 py-4">
        <div className="mx-auto w-full max-w-4xl">
          <ChatInput
            onSend={handleSend}
            disabled={isSending}
            placeholder="Reply..."
          />
          <p className="mt-2 text-center text-xs text-muted">
            AI models can make mistakes. Check low-confidence outputs.
          </p>
        </div>
      </div>
    </div>
  );
}

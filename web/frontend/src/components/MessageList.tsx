import { useEffect, useRef } from 'react';
import type { Message } from '../types';
import MessageBubble from './MessageBubble';

interface Props {
  messages: Message[];
  /**
   * ID of the assistant message currently receiving deltas, if any. The
   * bubble uses this to suppress the per-line confidence summary on the
   * currently-writing line (it only earns a summary once a new line
   * starts below it, or when the stream finishes).
   */
  streamingId?: string | null;
}

export default function MessageList({ messages, streamingId }: Props) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col gap-6 px-6 py-8">
      {messages.map((m) => (
        <MessageBubble
          key={m.id}
          message={m}
          isStreaming={m.id === streamingId}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

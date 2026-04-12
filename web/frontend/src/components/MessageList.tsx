import { useEffect, useRef } from 'react';
import type { Message } from '../types';
import MessageBubble from './MessageBubble';

interface Props {
  messages: Message[];
  mode: 'probly' | 'gemma';
}

export default function MessageList({ messages, mode }: Props) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col gap-6 px-6 py-8">
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} hideUncertainty={mode === 'gemma'} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

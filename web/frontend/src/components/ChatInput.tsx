import { useState, type KeyboardEvent } from 'react';

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
  modelLabel?: string;
}

export default function ChatInput({
  onSend,
  disabled,
  placeholder,
  modelLabel,
}: Props) {
  const [value, setValue] = useState('');

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const canSend = !disabled && value.trim() !== '';

  return (
    <div className="w-full rounded-2xl border border-rule bg-white shadow-[0_1px_2px_rgba(31,30,26,0.04),0_8px_24px_-12px_rgba(31,30,26,0.12)]">
      <textarea
        className="block w-full resize-none rounded-2xl bg-transparent px-5 pt-2 text-[15px] leading-6 text-ink placeholder:text-muted focus:outline-none"
        rows={2}
        placeholder={placeholder ?? 'How can I help you today?'}
        value={value}
        disabled={disabled}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <div className="flex items-center justify-between px-3 pb-2 pt-0">
        <button
          type="button"
          className="flex h-8 w-8 items-center justify-center rounded-full border border-rule text-ink/70 transition-colors hover:bg-canvas"
          aria-label="Add attachment"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14M5 12h14" />
          </svg>
        </button>

        <div className="flex items-center gap-2">
          <button
            type="button"
            className="flex items-center gap-1 rounded-md px-2 py-1 text-sm text-muted hover:text-ink"
          >
            <span>{modelLabel ?? 'Gemma 4'}</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>
          <button
            type="button"
            onClick={submit}
            disabled={!canSend}
            className={`flex h-8 w-8 items-center justify-center rounded-full transition-colors ${
              canSend
                ? 'bg-ink text-white hover:bg-ink/90'
                : 'bg-rule text-muted'
            }`}
            aria-label="Send message"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M13 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

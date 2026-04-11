import { useState, type CSSProperties } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Message, TokenConfidence } from '../types';

interface Props {
  message: Message;
}

type Feedback = 'up' | 'down' | null;
type ConfidenceMode = 'full' | 'concept' | 'word';

const CONFIDENCE_MODES: { key: ConfidenceMode; label: string }[] = [
  { key: 'full', label: 'Full Response' },
  { key: 'concept', label: 'Concept Level' },
  { key: 'word', label: 'Word Level' },
];

// SHAP red (#ff0d57). We use only this hue: low confidence → opaque red,
// high confidence → transparent, so "problem" tokens are the only thing
// that visually jumps out of an otherwise neutral bubble.
const SHAP_RED_RGB = '255, 13, 87';

/**
 * Map a [0, 1] confidence to a background color string. Uses a gamma=2
 * curve so very-low confidences pop while mid-range values stay subtle.
 * The optional ``maxAlpha`` override lets callers (e.g. the per-line
 * summary bar) render at full opacity even when in-text tints follow the
 * soft ramp.
 */
function confidenceToRgba(confidence: number, maxAlpha = 1): string {
  const clamped = Math.min(1, Math.max(0, confidence));
  const alpha = (1 - clamped) ** 2 * maxAlpha;
  return `rgba(${SHAP_RED_RGB}, ${alpha.toFixed(3)})`;
}

/**
 * Return a text color override when the confidence tint is dark enough
 * that default ink-on-red would be hard to read. Matches the alpha curve
 * used by ``confidenceToRgba``.
 */
function confidenceTextColor(confidence: number): string | undefined {
  const clamped = Math.min(1, Math.max(0, confidence));
  const alpha = (1 - clamped) ** 2;
  return alpha > 0.6 ? '#ffffff' : undefined;
}

/**
 * Length-weighted mean confidence across a list of tokens. Weighting by
 * character length prevents a stray `","` from dominating an otherwise
 * high-confidence line — and conversely, makes a long unsure word pull
 * the line's score down more than a one-letter conjunction.
 */
function weightedMean(tokens: TokenConfidence[]): number {
  let totalLen = 0;
  let weighted = 0;
  for (const t of tokens) {
    const len = t.text.length;
    totalLen += len;
    weighted += t.confidence * len;
  }
  return totalLen === 0 ? 1 : weighted / totalLen;
}

// Every tinted span transitions its background and text color with this
// easing so switching modes (or toggling the panel) fades rather than
// pops. Kept in one constant so the bubble background and per-token
// spans stay in sync.
const HIGHLIGHT_TRANSITION =
  'background-color 240ms cubic-bezier(0.4, 0, 0.2, 1), color 240ms cubic-bezier(0.4, 0, 0.2, 1)';

/**
 * Render the assistant reply as one span per token, always with identical
 * geometry (zero padding, no border radius), so that switching between
 * word, concept, full, and panel-closed states never causes the text to
 * reflow. The ``getTint`` callback decides which tint (if any) each token
 * receives in the current mode — returning ``null`` leaves the span
 * transparent, which lets `transition` animate it smoothly toward or away
 * from a colored state on the next render.
 *
 * Each token is split into its non-whitespace core and the trailing
 * whitespace that ``_chunk_reply`` glued onto it. Only the core gets the
 * tinted background, so highlights hug the word/punctuation and don't
 * bleed across the inter-word gaps.
 */
function TokenizedBody({
  tokens,
  getTint,
}: {
  tokens: TokenConfidence[];
  getTint: (index: number) => number | null;
}) {
  return (
    <p className="my-0 whitespace-pre-wrap leading-6">
      {tokens.map((t, i) => {
        const conf = getTint(i);
        const style: CSSProperties = { transition: HIGHLIGHT_TRANSITION };
        if (conf !== null) {
          style.backgroundColor = confidenceToRgba(conf);
          const textColor = confidenceTextColor(conf);
          if (textColor) style.color = textColor;
        }
        const match = t.text.match(/^(\S*)(\s*)$/);
        const word = match ? match[1] : t.text;
        const trailing = match ? match[2] : '';
        return (
          <span key={i}>
            <span style={style}>{word}</span>
            {trailing}
          </span>
        );
      })}
    </p>
  );
}

/**
 * Pick the right render path for the assistant's message body. Messages
 * without streamed ``tokens`` (no deltas yet, or a plain-text pipeline
 * without confidence) fall through to the existing Markdown renderer —
 * highlighting requires token-level data.
 *
 * Once tokens are present, every mode (and the panel-closed state) goes
 * through ``TokenizedBody`` so the DOM layout is identical in all four
 * cases — only the per-span background color changes. That is what lets
 * switching modes animate smoothly without shifting the text.
 */
function AssistantContent({
  message,
  mode,
  active,
}: {
  message: Message;
  mode: ConfidenceMode;
  active: boolean;
}) {
  if (!message.tokens || message.tokens.length === 0) {
    return (
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {message.content}
      </ReactMarkdown>
    );
  }
  const tokens = message.tokens;

  // Map the current (mode, active) state to a per-token tint lookup.
  // ``null`` means "no tint" and produces a fully transparent span, which
  // is what we want for the panel-closed state.
  let getTint: (index: number) => number | null;
  if (!active) {
    getTint = () => null;
  } else if (mode === 'full') {
    // Full mode tints every token with the same length-weighted mean score,
    // so the highlight lives on the text itself rather than on the
    // surrounding bubble.
    const fullScore = weightedMean(tokens);
    getTint = () => fullScore;
  } else {
    // Concept mode falls back to word-level tints (concept spans were
    // removed from the wire format for the real-time confidence demo).
    getTint = (i) => tokens[i].confidence;
  }

  return <TokenizedBody tokens={tokens} getTint={getTint} />;
}

interface MessageActionsProps {
  content: string;
  mode: ConfidenceMode;
  onModeChange: (mode: ConfidenceMode) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function MessageActions({
  content,
  mode,
  onModeChange,
  open,
  onOpenChange,
}: MessageActionsProps) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<Feedback>(null);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard unavailable — silently ignore in this mock UI.
    }
  };

  const baseBtn =
    'flex h-7 w-7 items-center justify-center rounded-md text-muted transition-colors hover:bg-panel hover:text-ink';

  return (
    <div className="mt-2 flex items-center gap-1">
      <button
        type="button"
        onClick={handleCopy}
        className={baseBtn}
        aria-label="Copy message"
        title={copied ? 'Copied' : 'Copy'}
      >
        {copied ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 6L9 17l-5-5" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
            <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
          </svg>
        )}
      </button>
      <button
        type="button"
        onClick={() => setFeedback((f) => (f === 'up' ? null : 'up'))}
        className={`${baseBtn} ${feedback === 'up' ? 'bg-panel text-ink' : ''}`}
        aria-label="Good response"
        aria-pressed={feedback === 'up'}
        title="Good response"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3zM7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3" />
        </svg>
      </button>
      <button
        type="button"
        onClick={() => setFeedback((f) => (f === 'down' ? null : 'down'))}
        className={`${baseBtn} ${feedback === 'down' ? 'bg-panel text-ink' : ''}`}
        aria-label="Bad response"
        aria-pressed={feedback === 'down'}
        title="Bad response"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3zM17 2h3a2 2 0 012 2v7a2 2 0 01-2 2h-3" />
        </svg>
      </button>
      <button
        type="button"
        onClick={() => onOpenChange(!open)}
        className={`flex h-7 items-center gap-1.5 rounded-md px-2 text-xs text-muted transition-colors hover:bg-panel hover:text-ink ${
          open ? 'bg-panel text-ink' : ''
        }`}
        aria-label="Show confidence"
        aria-expanded={open}
        title="Confidence"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="6" y1="20" x2="6" y2="14" />
          <line x1="12" y1="20" x2="12" y2="10" />
          <line x1="18" y1="20" x2="18" y2="4" />
        </svg>
        <span>Confidence</span>
      </button>
      <div
        className={`overflow-hidden transition-all duration-300 ease-out ${
          open ? 'ml-2 max-w-md opacity-100' : 'ml-0 max-w-0 opacity-0'
        }`}
        aria-hidden={!open}
      >
        <div className="relative flex rounded-full border border-rule bg-white/60 p-1 shadow-sm">
          <div
            className="pointer-events-none absolute bottom-1 left-1 top-1 w-28 rounded-full bg-ink shadow-sm transition-transform duration-300 ease-out"
            style={{
              transform: `translateX(${
                CONFIDENCE_MODES.findIndex((m) => m.key === mode) * 100
              }%)`,
            }}
            aria-hidden
          />
          {CONFIDENCE_MODES.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              onClick={() => onModeChange(key)}
              aria-pressed={mode === key}
              tabIndex={open ? 0 : -1}
              className={`relative z-10 w-28 whitespace-nowrap rounded-full px-3 py-1 text-center text-xs transition-colors duration-300 ease-out ${
                mode === key
                  ? 'text-white'
                  : 'text-muted hover:text-ink'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function MessageBubble({ message }: Props) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-panel px-4 py-2.5 text-[15px] leading-6 text-ink">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.role === 'system') {
    return (
      <div className="flex justify-center">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
          {message.content}
        </div>
      </div>
    );
  }

  return <AssistantBubble message={message} />;
}

function AssistantBubble({ message }: { message: Message }) {
  // State is lifted out of MessageActions so the selected mode can drive
  // the body renderer (word / concept / full tint) in addition to the
  // toggle UI in the action row.
  const [confidenceOpen, setConfidenceOpen] = useState(false);
  const [confidenceMode, setConfidenceMode] = useState<ConfidenceMode>('full');

  return (
    <div className="flex justify-start">
      <div className="flex max-w-[92%] flex-col">
        <img
          src="/gemma-color.png"
          alt="Gemma"
          className="mb-2 h-7 w-7 rounded-md object-contain"
        />
        <div
          className="
            prose prose-stone min-w-[36rem] max-w-none pt-0 text-[15px] leading-6 text-ink
            prose-p:my-2 prose-p:leading-6
            prose-headings:text-ink prose-headings:font-semibold
            prose-h1:text-xl prose-h2:text-lg prose-h3:text-base
            prose-strong:text-ink prose-strong:font-semibold
            prose-a:text-accent prose-a:no-underline hover:prose-a:underline
            prose-code:rounded prose-code:bg-panel prose-code:px-1 prose-code:py-0.5
            prose-code:text-[13.5px] prose-code:text-ink prose-code:before:content-none prose-code:after:content-none
            prose-pre:my-3 prose-pre:rounded-lg prose-pre:bg-ink prose-pre:text-canvas
            prose-pre:p-4 prose-pre:text-[13px] prose-pre:leading-5
            prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5
            prose-blockquote:border-l-2 prose-blockquote:border-rule
            prose-blockquote:pl-3 prose-blockquote:text-muted prose-blockquote:not-italic
            prose-hr:border-rule
          "
        >
          {message.thinking === 'active' ? (
            <span className="italic text-muted">
              Thinking<span className="animate-pulse">...</span>
            </span>
          ) : (
            <>
              {message.thinking === 'done' && (
                <div className="mb-1 text-xs italic text-muted">
                  Thought for {message.thoughtLabel ?? '<1s'}
                </div>
              )}
              <div className="rounded-2xl border border-rule/70 bg-white/50 px-4 py-2 shadow-sm">
                {message.content ? (
                  <AssistantContent
                    message={message}
                    mode={confidenceMode}
                    active={confidenceOpen}
                  />
                ) : (
                  // Streaming placeholder while waiting for the first chunk.
                  <span className="inline-block h-4 w-2 animate-pulse bg-muted align-middle" />
                )}
              </div>
              {message.content && (
                <MessageActions
                  content={message.content}
                  mode={confidenceMode}
                  onModeChange={setConfidenceMode}
                  open={confidenceOpen}
                  onOpenChange={setConfidenceOpen}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

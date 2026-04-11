import { useEffect, useState, type CSSProperties, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ConceptSpan, ConfidencePayload, Message } from '../types';

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
 */
function confidenceToRgba(confidence: number): string {
  const clamped = Math.min(1, Math.max(0, confidence));
  const alpha = (1 - clamped) ** 2;
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

// Every tinted span transitions its background and text color with this
// easing so switching modes (or toggling the panel) fades rather than
// pops. Kept in one constant so the bubble background and per-word
// spans stay in sync.
const HIGHLIGHT_TRANSITION =
  'background-color 240ms cubic-bezier(0.4, 0, 0.2, 1), color 240ms cubic-bezier(0.4, 0, 0.2, 1)';

/**
 * Information about which concept (if any) a given word participates in.
 * ``isLast`` flags the final word of the span so we can leave its
 * trailing whitespace untinted — making the concept band end cleanly at
 * the last glyph instead of bleeding into the next word's gap.
 */
interface ConceptInfo {
  concept: ConceptSpan;
  isLast: boolean;
}

function buildConceptIndex(
  wordCount: number,
  concepts: readonly ConceptSpan[],
): (ConceptInfo | null)[] {
  const out: (ConceptInfo | null)[] = new Array(wordCount).fill(null);
  for (const span of concepts) {
    for (let i = span.firstWord; i <= span.lastWord; i++) {
      if (i >= 0 && i < wordCount) {
        out[i] = { concept: span, isLast: i === span.lastWord };
      }
    }
  }
  return out;
}

/**
 * Render the assistant reply with per-mode confidence tints.
 *
 * Every word is rendered as an outer ``<span>`` that wraps the word's
 * full chunk (glyph + trailing whitespace). Only the background tint
 * differs between modes:
 *
 *   - **word**    — inner span around the glyph is tinted with the
 *                   word's confidence; trailing whitespace stays
 *                   transparent.
 *   - **concept** — if the word is inside a concept span, the outer
 *                   chunk span is tinted with the concept's confidence
 *                   (so consecutive concept words form a continuous
 *                   band through the inter-word whitespace). The last
 *                   word of a span only tints its inner glyph, so the
 *                   band ends cleanly at the last letter.
 *   - **full**    — outer chunk span is tinted with the single
 *                   whole-response confidence (whitespace included), so
 *                   every visual line ends up with the same uniform
 *                   tint.
 */
function WordBody({
  confidence,
  mode,
  active,
}: {
  confidence: ConfidencePayload;
  mode: ConfidenceMode;
  active: boolean;
}) {
  const { words, concepts, full } = confidence;
  const conceptIndex = buildConceptIndex(words.length, concepts);

  return (
    <p className="my-0 whitespace-pre-wrap leading-6">
      {words.map((w, i) => {
        const outerStyle: CSSProperties = { transition: HIGHLIGHT_TRANSITION };
        let innerContent: ReactNode = w.text;

        if (active) {
          if (mode === 'word') {
            const match = w.text.match(/^(\S*)(\s*)$/);
            const glyph = match ? match[1] : w.text;
            const trailing = match ? match[2] : '';
            const innerStyle: CSSProperties = {
              transition: HIGHLIGHT_TRANSITION,
              backgroundColor: confidenceToRgba(w.confidence),
            };
            const textColor = confidenceTextColor(w.confidence);
            if (textColor) innerStyle.color = textColor;
            const tintedGlyph = <span style={innerStyle}>{glyph}</span>;
            // Only show the dotted-underline + hover tooltip affordance in
            // word mode and only on words the backend actually seeded with
            // alternatives — keep the hint rare so it reads as a
            // spotlight, not noise on every hedged word.
            const alternatives = w.alternatives;
            const decoratedGlyph =
              alternatives && alternatives.length > 0 ? (
                <AlternativesTooltip alternatives={alternatives}>
                  <span className="underline decoration-dotted decoration-muted/60 underline-offset-2">
                    {tintedGlyph}
                  </span>
                </AlternativesTooltip>
              ) : (
                tintedGlyph
              );
            innerContent = (
              <>
                {decoratedGlyph}
                {trailing}
              </>
            );
          } else if (mode === 'concept') {
            const info = conceptIndex[i];
            if (info) {
              if (info.isLast) {
                // Tint only the glyph of the last concept word so the
                // band stops cleanly at the last letter.
                const match = w.text.match(/^(\S*)(\s*)$/);
                const glyph = match ? match[1] : w.text;
                const trailing = match ? match[2] : '';
                const innerStyle: CSSProperties = {
                  transition: HIGHLIGHT_TRANSITION,
                  backgroundColor: confidenceToRgba(info.concept.confidence),
                };
                const textColor = confidenceTextColor(info.concept.confidence);
                if (textColor) innerStyle.color = textColor;
                innerContent = (
                  <>
                    <span style={innerStyle}>{glyph}</span>
                    {trailing}
                  </>
                );
              } else {
                outerStyle.backgroundColor = confidenceToRgba(info.concept.confidence);
                const textColor = confidenceTextColor(info.concept.confidence);
                if (textColor) outerStyle.color = textColor;
              }
            }
          } else {
            // full mode — single whole-response confidence, applied
            // uniformly to every word chunk (glyph + trailing
            // whitespace), so every visual line paints the same tint.
            outerStyle.backgroundColor = confidenceToRgba(full);
            const textColor = confidenceTextColor(full);
            if (textColor) outerStyle.color = textColor;
          }
        }

        return (
          <span key={i} style={outerStyle}>
            {innerContent}
          </span>
        );
      })}
    </p>
  );
}

/**
 * Pick the right render path for the assistant's message body. While a
 * message is still streaming (or came from a backend with no confidence
 * data, like real Gemma), ``message.confidence`` is undefined and we
 * fall through to the plain Markdown renderer — the confidence toggle
 * stays greyed out in that state.
 *
 * Once ``confidence`` arrives, ``WordBody`` takes over and renders one
 * outer span per word so all three display modes can tint their spans
 * independently without reflowing the text.
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
  if (!message.confidence) {
    return (
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {message.content}
      </ReactMarkdown>
    );
  }
  return <WordBody confidence={message.confidence} mode={mode} active={active} />;
}

/**
 * Wrap a single control in a little floating tooltip bubble that shows
 * ``label`` on hover or keyboard focus. The wrapper is an
 * ``inline-flex`` so it doesn't disrupt the action row's flexbox layout,
 * and the tooltip itself is ``pointer-events-none`` + absolutely
 * positioned so it never blocks clicks or shifts surrounding elements.
 * A downward-pointing CSS triangle (border trick) glues the bubble to
 * the control below it. Uses Tailwind's ``group`` / ``group-hover`` +
 * ``group-focus-within`` so no extra React state is needed to drive the
 * fade.
 */
function WithTooltip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="group relative inline-flex">
      {children}
      <span
        role="tooltip"
        className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 -translate-x-1/2 whitespace-nowrap rounded-md bg-ink px-2 py-1 text-[11px] font-medium text-white opacity-0 shadow-md transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {label}
        <span
          aria-hidden
          className="absolute bottom-full left-1/2 h-0 w-0 -translate-x-1/2 border-4 border-transparent border-b-ink"
        />
      </span>
    </span>
  );
}

/**
 * Floating tooltip showing a vertical list of alternative words on hover.
 * Rendered in Word-Level confidence mode over words that the backend
 * seeded with an ``alternatives`` list. Visually mirrors ``WithTooltip``
 * (same dark pill + triangle) but renders a stacked list instead of a
 * single label, and uses an ``inline`` wrapper so it can sit in the
 * middle of a flowing ``<p>`` without breaking line wrap. The tooltip
 * itself is ``pointer-events-none`` so it never eats clicks on
 * neighbouring words.
 */
function AlternativesTooltip({
  alternatives,
  children,
}: {
  alternatives: readonly string[];
  children: ReactNode;
}) {
  return (
    <span className="group relative inline cursor-help">
      {children}
      <span
        role="tooltip"
        className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 -translate-x-1/2 flex flex-col gap-0.5 whitespace-nowrap rounded-md bg-ink px-2 py-1.5 text-[11px] font-medium text-white opacity-0 shadow-md transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {alternatives.map((alt) => (
          <span key={alt}>{alt}</span>
        ))}
        <span
          aria-hidden
          className="absolute bottom-full left-1/2 h-0 w-0 -translate-x-1/2 border-4 border-transparent border-b-ink"
        />
      </span>
    </span>
  );
}

interface MessageActionsProps {
  content: string;
  mode: ConfidenceMode;
  onModeChange: (mode: ConfidenceMode) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /**
   * When true the confidence toggle + mode switch are visible but
   * non-interactive (greyed out). Used while generation is still
   * streaming, or when the backend produced no confidence data.
   */
  confidenceDisabled: boolean;
  /**
   * When true the backend has flagged this response as low confidence,
   * and the Confidence button should draw a blinking-then-solid red
   * underline to surface the warning in the action row.
   */
  lowConfidence: boolean;
}

function MessageActions({
  content,
  mode,
  onModeChange,
  open,
  onOpenChange,
  confidenceDisabled,
  lowConfidence,
}: MessageActionsProps) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<Feedback>(null);
  // The underline blinks briefly when the low-confidence flag first
  // arrives (i.e. when the confidence frame lands, after generation
  // finishes), then settles into a solid red marker.
  const [lowConfBlinking, setLowConfBlinking] = useState(false);
  useEffect(() => {
    if (!lowConfidence) {
      setLowConfBlinking(false);
      return;
    }
    setLowConfBlinking(true);
    const timer = window.setTimeout(() => setLowConfBlinking(false), 2400);
    return () => window.clearTimeout(timer);
  }, [lowConfidence]);

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

  const confidenceTooltipLabel = confidenceDisabled
    ? 'Confidence available after generation finishes'
    : 'Display model confidence';

  return (
    <div className="mt-2 flex items-center gap-1">
      <WithTooltip label={copied ? 'Copied' : 'Copy'}>
        <button
          type="button"
          onClick={handleCopy}
          className={baseBtn}
          aria-label="Copy message"
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
      </WithTooltip>
      <WithTooltip label="Positive feedback">
        <button
          type="button"
          onClick={() => setFeedback((f) => (f === 'up' ? null : 'up'))}
          className={`${baseBtn} ${feedback === 'up' ? 'bg-panel text-ink' : ''}`}
          aria-label="Good response"
          aria-pressed={feedback === 'up'}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3zM7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3" />
          </svg>
        </button>
      </WithTooltip>
      <WithTooltip label="Negative feedback">
        <button
          type="button"
          onClick={() => setFeedback((f) => (f === 'down' ? null : 'down'))}
          className={`${baseBtn} ${feedback === 'down' ? 'bg-panel text-ink' : ''}`}
          aria-label="Bad response"
          aria-pressed={feedback === 'down'}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3zM17 2h3a2 2 0 012 2v7a2 2 0 01-2 2h-3" />
          </svg>
        </button>
      </WithTooltip>
      <WithTooltip label={confidenceTooltipLabel}>
        <button
          type="button"
          onClick={() => {
            if (confidenceDisabled) return;
            onOpenChange(!open);
          }}
          disabled={confidenceDisabled}
          className={`flex h-7 items-center gap-1.5 rounded-md px-2 text-xs transition-colors ${
            confidenceDisabled
              ? 'cursor-not-allowed text-muted/50'
              : `text-muted hover:bg-panel hover:text-ink ${open ? 'bg-panel text-ink' : ''}`
          }`}
          aria-label={
            lowConfidence
              ? 'Show confidence (low confidence response)'
              : 'Show confidence'
          }
          aria-expanded={open}
          aria-disabled={confidenceDisabled}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="6" y1="20" x2="6" y2="14" />
            <line x1="12" y1="20" x2="12" y2="10" />
            <line x1="18" y1="20" x2="18" y2="4" />
          </svg>
          <span>Confidence</span>
        </button>
        {lowConfidence && (
          <span
            aria-hidden
            className="pointer-events-none absolute inset-x-1 -bottom-0.5 h-0.5 rounded-full"
            style={{
              backgroundColor: `rgb(${SHAP_RED_RGB})`,
              animation: lowConfBlinking
                ? 'lowConfBlink 600ms steps(1) infinite'
                : undefined,
            }}
          />
        )}
      </WithTooltip>
      <div
        className={`overflow-hidden transition-all duration-300 ease-out ${
          open && !confidenceDisabled ? 'ml-2 max-w-md opacity-100' : 'ml-0 max-w-0 opacity-0'
        }`}
        aria-hidden={!open || confidenceDisabled}
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
              tabIndex={open && !confidenceDisabled ? 0 : -1}
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

  const confidenceAvailable = message.confidence !== undefined;
  const confidenceActive = confidenceOpen && confidenceAvailable;
  const lowConfidence =
    confidenceAvailable && message.confidence?.lowConfidence === true;

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
                    active={confidenceActive}
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
                  confidenceDisabled={!confidenceAvailable}
                  lowConfidence={lowConfidence}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

import { useEffect, useState, type CSSProperties, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ConceptSpan, Message, UncertaintyPayload } from '../types';

interface Props {
  message: Message;
}

type Feedback = 'up' | 'down' | null;
type UncertaintyMode = 'full' | 'concept' | 'word';

const UNCERTAINTY_MODES: { key: UncertaintyMode; label: string }[] = [
  { key: 'full', label: 'Full Response' },
  { key: 'concept', label: 'Concept Level' },
  { key: 'word', label: 'Word Level' },
];

// SHAP red (#ff0d57). We use only this hue: high uncertainty → opaque red,
// low uncertainty → transparent, so "problem" tokens are the only thing
// that visually jumps out of an otherwise neutral bubble.
const SHAP_RED_RGB = '255, 13, 87';

/**
 * Map a [0, 1] uncertainty to a background color string. Uses a gamma=2
 * curve so very-high uncertainties pop while mid-range values stay subtle.
 */
function uncertaintyToRgba(uncertainty: number): string {
  const clamped = Math.min(1, Math.max(0, uncertainty));
  const alpha = clamped ** 2;
  return `rgba(${SHAP_RED_RGB}, ${alpha.toFixed(3)})`;
}

/**
 * Return a text color override when the uncertainty tint is dark enough
 * that default ink-on-red would be hard to read. Matches the alpha curve
 * used by ``uncertaintyToRgba``.
 */
function uncertaintyTextColor(uncertainty: number): string | undefined {
  const clamped = Math.min(1, Math.max(0, uncertainty));
  const alpha = clamped ** 2;
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
 * Render the assistant reply with per-mode uncertainty tints.
 *
 * Every word is rendered as an outer ``<span>`` that wraps the word's
 * full chunk (glyph + trailing whitespace). Only the background tint
 * differs between modes:
 *
 *   - **word**    — inner span around the glyph is tinted with the
 *                   word's uncertainty; trailing whitespace stays
 *                   transparent.
 *   - **concept** — if the word is inside a concept span, the outer
 *                   chunk span is tinted with the concept's uncertainty
 *                   (so consecutive concept words form a continuous
 *                   band through the inter-word whitespace). The last
 *                   word of a span only tints its inner glyph, so the
 *                   band ends cleanly at the last letter.
 *   - **full**    — outer chunk span is tinted with the single
 *                   whole-response uncertainty (whitespace included), so
 *                   every visual line ends up with the same uniform
 *                   tint.
 */
function WordBody({
  uncertainty,
  mode,
  active,
  replacements,
  onReplaceWord,
}: {
  uncertainty: UncertaintyPayload;
  mode: UncertaintyMode;
  active: boolean;
  replacements: Map<number, string>;
  onReplaceWord: (wordIndex: number, newGlyph: string) => void;
}) {
  const { words, concepts, full } = uncertainty;
  const conceptIndex = buildConceptIndex(words.length, concepts);

  return (
    <p className="my-0 whitespace-pre-wrap leading-6">
      {words.map((w, i) => {
        const outerStyle: CSSProperties = { transition: HIGHLIGHT_TRANSITION };
        // Split every word into its non-whitespace glyph + trailing
        // whitespace so we can swap the glyph for a user-picked
        // alternative while leaving punctuation / spacing untouched.
        const rawMatch = w.text.match(/^(\S*)(\s*)$/);
        const originalGlyph = rawMatch ? rawMatch[1] : w.text;
        const trailing = rawMatch ? rawMatch[2] : '';
        const displayGlyph = replacements.get(i) ?? originalGlyph;
        let innerContent: ReactNode = displayGlyph + trailing;

        if (active) {
          if (mode === 'word') {
            const innerStyle: CSSProperties = {
              transition: HIGHLIGHT_TRANSITION,
              backgroundColor: uncertaintyToRgba(w.uncertainty),
            };
            const textColor = uncertaintyTextColor(w.uncertainty);
            if (textColor) innerStyle.color = textColor;
            const tintedGlyph = <span style={innerStyle}>{displayGlyph}</span>;
            // Only show the dotted-underline + hover tooltip affordance in
            // word mode and only on words the backend actually seeded with
            // alternatives — keep the hint rare so it reads as a
            // spotlight, not noise on every hedged word.
            const alternatives = w.alternatives;
            const decoratedGlyph =
              alternatives && alternatives.length > 0 ? (
                <AlternativesTooltip
                  alternatives={alternatives}
                  onSelect={(alt) => onReplaceWord(i, alt)}
                >
                  <span
                    tabIndex={0}
                    className="underline decoration-dotted decoration-muted/60 underline-offset-2 focus:outline-none focus-visible:rounded focus-visible:ring-2 focus-visible:ring-ink/40"
                  >
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
                const innerStyle: CSSProperties = {
                  transition: HIGHLIGHT_TRANSITION,
                  backgroundColor: uncertaintyToRgba(info.concept.uncertainty),
                };
                const textColor = uncertaintyTextColor(info.concept.uncertainty);
                if (textColor) innerStyle.color = textColor;
                innerContent = (
                  <>
                    <span style={innerStyle}>{displayGlyph}</span>
                    {trailing}
                  </>
                );
              } else {
                outerStyle.backgroundColor = uncertaintyToRgba(info.concept.uncertainty);
                const textColor = uncertaintyTextColor(info.concept.uncertainty);
                if (textColor) outerStyle.color = textColor;
              }
            }
          } else {
            // full mode — single whole-response uncertainty, applied
            // uniformly to every word chunk (glyph + trailing
            // whitespace), so every visual line paints the same tint.
            outerStyle.backgroundColor = uncertaintyToRgba(full);
            const textColor = uncertaintyTextColor(full);
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
 * message is still streaming (or came from a backend with no uncertainty
 * data, like real Gemma), ``message.uncertainty`` is undefined and we
 * fall through to the plain Markdown renderer — the uncertainty toggle
 * stays greyed out in that state.
 *
 * Once ``uncertainty`` arrives, ``WordBody`` takes over and renders one
 * outer span per word so all three display modes can tint their spans
 * independently without reflowing the text.
 */
function AssistantContent({
  message,
  mode,
  active,
  replacements,
  onReplaceWord,
}: {
  message: Message;
  mode: UncertaintyMode;
  active: boolean;
  replacements: Map<number, string>;
  onReplaceWord: (wordIndex: number, newGlyph: string) => void;
}) {
  if (!message.uncertainty) {
    return (
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {message.content}
      </ReactMarkdown>
    );
  }
  return (
    <WordBody
      uncertainty={message.uncertainty}
      mode={mode}
      active={active}
      replacements={replacements}
      onReplaceWord={onReplaceWord}
    />
  );
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
 * Floating picker showing a vertical list of alternative words above the
 * hovered word. Rendered in Word-Level uncertainty mode over words that
 * the backend seeded with an ``alternatives`` list.
 *
 * Visually distinct from the dark ``WithTooltip`` pill used by the
 * action row: this one is a light card with an ink border, so it reads
 * as an interactive picker rather than a passive hint.
 *
 * Each alternative is a real ``<button>``: clicking commits a
 * replacement via ``onSelect`` and closes the picker. The card animates
 * up from the word with a slide+scale, and each alternative fades in on
 * a staggered ``transitionDelay`` so the list cascades rather than
 * popping.
 *
 * The positioned wrapper uses invisible ``pb-2`` as a hover bridge so
 * the cursor can cross the gap between the word and the card without
 * triggering ``pointerleave``. Pointer events on the wrapper are gated
 * on ``open`` so the padding doesn't trap clicks when the picker is
 * hidden.
 */
function AlternativesTooltip({
  alternatives,
  onSelect,
  children,
}: {
  alternatives: readonly string[];
  onSelect: (alternative: string) => void;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);

  const handleSelect = (alt: string) => {
    onSelect(alt);
    setOpen(false);
  };

  return (
    <span
      className="relative inline cursor-help"
      onPointerEnter={() => setOpen(true)}
      onPointerLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      <span
        role="tooltip"
        className="absolute left-1/2 bottom-full z-20 -translate-x-1/2 pb-2"
        style={{ pointerEvents: open ? 'auto' : 'none' }}
      >
        <span
          className="relative flex flex-col gap-0.5 whitespace-nowrap rounded-lg border border-ink/15 bg-white px-1.5 py-1.5 text-[11px] font-medium text-ink shadow-lg ring-1 ring-ink/5"
          style={{
            opacity: open ? 1 : 0,
            transform: open
              ? 'translateY(0) scale(1)'
              : 'translateY(6px) scale(0.96)',
            transition:
              'opacity 180ms ease-out, transform 240ms cubic-bezier(0.2, 0.8, 0.2, 1)',
            transformOrigin: 'bottom center',
          }}
        >
          {/* Downward-pointing triangle glued to the bottom edge of the
              card so it aligns visually with the word below. */}
          <span
            aria-hidden
            className="absolute top-full left-1/2 h-0 w-0 -translate-x-1/2 border-4 border-transparent border-t-white"
          />
          <span
            aria-hidden
            className="absolute top-full left-1/2 h-0 w-0 -translate-x-1/2 border-4 border-transparent"
            style={{ borderTopColor: 'rgba(31, 30, 26, 0.15)', marginTop: 1 }}
          />
          {alternatives.map((alt, i) => (
            <button
              key={alt}
              type="button"
              onClick={() => handleSelect(alt)}
              className="rounded px-1.5 py-0.5 text-left transition-colors duration-150 hover:bg-panel focus:bg-panel focus:outline-none"
              style={{
                opacity: open ? 1 : 0,
                transform: open ? 'translateY(0)' : 'translateY(4px)',
                transition:
                  'opacity 160ms ease-out, transform 200ms cubic-bezier(0.2, 0.8, 0.2, 1)',
                transitionDelay: open ? `${i * 40}ms` : '0ms',
              }}
            >
              {alt}
            </button>
          ))}
        </span>
      </span>
    </span>
  );
}

interface MessageActionsProps {
  content: string;
  mode: UncertaintyMode;
  onModeChange: (mode: UncertaintyMode) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /**
   * When true the uncertainty toggle + mode switch are visible but
   * non-interactive (greyed out). Used while generation is still
   * streaming, or when the backend produced no uncertainty data.
   */
  uncertaintyDisabled: boolean;
  /**
   * When true the backend has flagged this response as high uncertainty,
   * and the Uncertainty button should draw a blinking-then-solid red
   * underline to surface the warning in the action row.
   */
  highUncertainty: boolean;
  /**
   * When true this message carries a mock-only alternative reply text,
   * so the action row should expose a small toggle button next to the
   * Uncertainty button. When false the regenerate button is not
   * rendered at all (it takes no layout space).
   */
  hasRegenerate: boolean;
  /** Current state of the regenerate toggle (original vs alternative). */
  showRegenerate: boolean;
  onToggleRegenerate: () => void;
}

function MessageActions({
  content,
  mode,
  onModeChange,
  open,
  onOpenChange,
  uncertaintyDisabled,
  highUncertainty,
  hasRegenerate,
  showRegenerate,
  onToggleRegenerate,
}: MessageActionsProps) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<Feedback>(null);
  // The underline blinks briefly when the high-uncertainty flag first
  // arrives (i.e. when the uncertainty frame lands, after generation
  // finishes), then settles into a solid red marker.
  const [highUncBlinking, setHighUncBlinking] = useState(false);
  useEffect(() => {
    if (!highUncertainty) {
      setHighUncBlinking(false);
      return;
    }
    setHighUncBlinking(true);
    const timer = window.setTimeout(() => setHighUncBlinking(false), 2400);
    return () => window.clearTimeout(timer);
  }, [highUncertainty]);

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

  const uncertaintyTooltipLabel = uncertaintyDisabled
    ? 'Uncertainty available after generation finishes'
    : 'Display model uncertainty';

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
      <WithTooltip label={uncertaintyTooltipLabel}>
        <button
          type="button"
          onClick={() => {
            if (uncertaintyDisabled) return;
            onOpenChange(!open);
          }}
          disabled={uncertaintyDisabled}
          className={`flex h-7 items-center gap-1.5 rounded-md px-2 text-xs transition-colors ${
            uncertaintyDisabled
              ? 'cursor-not-allowed text-muted/50'
              : `text-muted hover:bg-panel hover:text-ink ${open ? 'bg-panel text-ink' : ''}`
          }`}
          aria-label={
            highUncertainty
              ? 'Show uncertainty (high uncertainty response)'
              : 'Show uncertainty'
          }
          aria-expanded={open}
          aria-disabled={uncertaintyDisabled}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="6" y1="20" x2="6" y2="14" />
            <line x1="12" y1="20" x2="12" y2="10" />
            <line x1="18" y1="20" x2="18" y2="4" />
          </svg>
          <span className="relative inline-block">
            Uncertainty
            {highUncertainty && (
              <span
                aria-hidden
                className="pointer-events-none absolute inset-x-0 -bottom-0.5 h-px"
                style={{
                  backgroundColor: `rgba(${SHAP_RED_RGB}, 0.45)`,
                  animation: highUncBlinking
                    ? 'highUncBlink 600ms steps(1) infinite'
                    : undefined,
                }}
              />
            )}
          </span>
        </button>
      </WithTooltip>
      <div
        className={`overflow-hidden transition-all duration-300 ease-out ${
          open && !uncertaintyDisabled ? 'ml-2 max-w-md opacity-100' : 'ml-0 max-w-0 opacity-0'
        }`}
        aria-hidden={!open || uncertaintyDisabled}
      >
        <div className="relative flex rounded-full border border-rule bg-white/60 p-1 shadow-sm">
          <div
            className="pointer-events-none absolute bottom-1 left-1 top-1 w-28 rounded-full bg-ink shadow-sm transition-transform duration-300 ease-out"
            style={{
              transform: `translateX(${
                UNCERTAINTY_MODES.findIndex((m) => m.key === mode) * 100
              }%)`,
            }}
            aria-hidden
          />
          {UNCERTAINTY_MODES.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              onClick={() => onModeChange(key)}
              aria-pressed={mode === key}
              tabIndex={open && !uncertaintyDisabled ? 0 : -1}
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
      {/*
        Regenerate pill. Visually this is part of the Uncertainty cluster
        (it's a form of uncertainty-level intervention), so it lives
        inside its own expanding wrapper that opens in sync with the
        Uncertainty mode selector. The wrapper only takes layout space
        when (a) the uncertainty panel is open and (b) this message
        actually carries a regenerate alternative — otherwise it stays
        collapsed at ``max-w-0 opacity-0`` and contributes nothing.
      */}
      <div
        className={`overflow-hidden transition-all duration-300 ease-out ${
          open && !uncertaintyDisabled && hasRegenerate
            ? 'ml-2 max-w-xs opacity-100'
            : 'ml-0 max-w-0 opacity-0'
        }`}
        aria-hidden={!open || uncertaintyDisabled || !hasRegenerate}
      >
        {/*
          Outer pill container mirrors the mode-selector's wrapper
          (``rounded-full border border-rule bg-white/60 p-1 shadow-sm``)
          so the regenerate control visually reads as another segment of
          the same uncertainty control strip. The inner button matches
          the mode pills' ``rounded-full px-3 py-1 text-xs`` sizing, but
          is icon-only and square-ish so it sits at the same height.
        */}
        <div className="flex rounded-full border border-rule bg-white/60 p-1 shadow-sm">
          <WithTooltip
            label={showRegenerate ? 'Show original response' : 'Show regenerated response'}
          >
            <button
              type="button"
              onClick={onToggleRegenerate}
              tabIndex={open && !uncertaintyDisabled && hasRegenerate ? 0 : -1}
              aria-pressed={showRegenerate}
              aria-label={
                showRegenerate ? 'Show original response' : 'Show regenerated response'
              }
              className={`flex h-6 items-center justify-center rounded-full px-3 text-xs transition-colors duration-300 ease-out ${
                showRegenerate
                  ? 'bg-ink text-white shadow-sm'
                  : 'text-muted hover:text-ink'
              }`}
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="23 4 23 10 17 10" />
                <polyline points="1 20 1 14 7 14" />
                <path d="M3.51 9a9 9 0 0114.85-3.36L23 10" />
                <path d="M20.49 15a9 9 0 01-14.85 3.36L1 14" />
              </svg>
            </button>
          </WithTooltip>
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
  const [uncertaintyOpen, setUncertaintyOpen] = useState(false);
  const [uncertaintyMode, setUncertaintyMode] = useState<UncertaintyMode>('full');
  const [showRegenerate, setShowRegenerate] = useState(false);
  // Per-word glyph overrides picked by the user from the alternatives
  // tooltip. Keyed by index into ``message.uncertainty.words``; only the
  // non-whitespace glyph is replaced, trailing punctuation/whitespace is
  // preserved at render time in ``WordBody``.
  const [replacements, setReplacements] = useState<Map<number, string>>(
    () => new Map(),
  );
  const handleReplaceWord = (wordIndex: number, newGlyph: string) => {
    setReplacements((prev) => {
      const next = new Map(prev);
      next.set(wordIndex, newGlyph);
      return next;
    });
  };

  const uncertaintyAvailable = message.uncertainty !== undefined;
  const highUncertainty =
    uncertaintyAvailable && message.uncertainty?.highUncertainty === true;
  const regenerateText = message.uncertainty?.regenerate;
  const hasRegenerate = typeof regenerateText === 'string' && regenerateText.length > 0;
  // When the user is viewing the regenerate alternative, the per-word
  // uncertainty indices no longer line up with the rendered text, so we
  // fall back to plain (untinted) rendering for that view. The toggle
  // itself still works; only the body tinting is suppressed.
  const viewingRegenerate = showRegenerate && hasRegenerate;
  const uncertaintyActive = uncertaintyOpen && uncertaintyAvailable && !viewingRegenerate;
  const effectiveContent = viewingRegenerate ? (regenerateText as string) : message.content;

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
                  viewingRegenerate ? (
                    // Regenerate alternative has no per-word uncertainty
                    // data of its own, so we render it as plain Markdown
                    // rather than going through ``WordBody``.
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {effectiveContent}
                    </ReactMarkdown>
                  ) : (
                    <AssistantContent
                      message={message}
                      mode={uncertaintyMode}
                      active={uncertaintyActive}
                      replacements={replacements}
                      onReplaceWord={handleReplaceWord}
                    />
                  )
                ) : (
                  // Streaming placeholder while waiting for the first chunk.
                  <span className="inline-block h-4 w-2 animate-pulse bg-muted align-middle" />
                )}
              </div>
              {message.content && (
                <MessageActions
                  content={effectiveContent}
                  mode={uncertaintyMode}
                  onModeChange={setUncertaintyMode}
                  open={uncertaintyOpen}
                  onOpenChange={setUncertaintyOpen}
                  uncertaintyDisabled={!uncertaintyAvailable}
                  highUncertainty={highUncertainty}
                  hasRegenerate={hasRegenerate}
                  showRegenerate={viewingRegenerate}
                  onToggleRegenerate={() => setShowRegenerate((v) => !v)}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

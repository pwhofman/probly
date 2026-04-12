export type Role = 'user' | 'assistant' | 'system';

export type ChatMode = 'probly' | 'gemma';

/** One word (plus any trailing whitespace) with its backend-authored uncertainty. */
export interface Word {
  text: string;
  uncertainty: number;
  /** Hardcoded (mock-only) alternative words the model might have produced
   *  at this position. When present and non-empty, Word-Level display mode
   *  wraps this word in a dotted-underline hover tooltip listing the
   *  alternatives. Real Gemma never sets this; omit rather than emit an
   *  empty array. */
  alternatives?: string[];
}

/** Multi-word concept span with its own uncertainty.
 *
 * ``firstWord`` and ``lastWord`` are inclusive indices into the
 * ``UncertaintyPayload.words`` array. Concept mode on the frontend tints
 * every word in ``[firstWord, lastWord]`` (including inter-word
 * whitespace) with ``uncertainty``.
 */
export interface ConceptSpan {
  firstWord: number;
  lastWord: number;
  uncertainty: number;
}

/** Final uncertainty payload the backend sends once generation has stopped.
 *
 * Each field drives one of the display modes:
 *   - ``words``    — per-word tints in "word" mode.
 *   - ``concepts`` — multi-word spans in "concept" mode.
 *   - ``full``     — a single whole-response uncertainty that the frontend
 *                    paints as a uniform tint across every visual line
 *                    (whitespace included) in "full" mode.
 *
 * All uncertainty math lives in the backend — the frontend only picks
 * values and paints them.
 */
export interface UncertaintyPayload {
  words: Word[];
  concepts: ConceptSpan[];
  full: number;
  /** Mock-only: when true, the action row flags this response with a
   *  blinking-then-solid red underline under the Uncertainty button.
   *  Absent / false means "no warning". Real Gemma never sets this. */
  highUncertainty?: boolean;
  /** Mock-only alternative reply text. When present, the action row
   *  shows a small toggle button next to the Uncertainty button that
   *  swaps the rendered message body between the original content and
   *  this alternative. Real Gemma never sets this. */
  regenerate?: string;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  thinking?: 'active' | 'done';
  thoughtLabel?: string;
  /**
   * Present on assistant messages once the backend has sent its final
   * uncertainty frame (i.e. after generation has stopped). Absence means
   * uncertainty display is unavailable for this message — either still
   * streaming, or the backend (real Gemma) doesn't produce uncertainty
   * data at all.
   */
  uncertainty?: UncertaintyPayload;
}

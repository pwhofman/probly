export type Role = 'user' | 'assistant' | 'system';

/** Per-token confidence streamed alongside each delta from the backend. */
export interface TokenConfidence {
  text: string;
  confidence: number;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  thinking?: 'active' | 'done';
  thoughtLabel?: string;
  /**
   * Grows in-place as deltas arrive. Present on assistant messages once the
   * backend has emitted at least one delta-with-confidence frame. Absent
   * messages fall back to plain Markdown rendering and skip the per-line
   * summary column.
   */
  tokens?: TokenConfidence[];
}

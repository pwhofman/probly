// Static decorative sidebar. Nothing here is wired up yet —
// it's a skeleton mimicking the Claude.ai layout so the main
// chat area has the right surrounding shape.

import type { ReactNode } from 'react';

interface NavItemProps {
  icon: ReactNode;
  label: string;
  active?: boolean;
  onClick?: () => void;
}

function NavItem({ icon, label, active, onClick }: NavItemProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
        active
          ? 'bg-white text-ink shadow-sm'
          : 'text-ink/80 hover:bg-white/60'
      }`}
    >
      <span className="text-ink/70">{icon}</span>
      <span>{label}</span>
    </button>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="px-3 pb-1 pt-4 text-xs font-medium uppercase tracking-wide text-muted">
      {children}
    </div>
  );
}

function RecentItem({ title }: { title: string }) {
  return (
    <button
      type="button"
      className="w-full truncate rounded-lg px-3 py-1.5 text-left text-sm text-ink/75 hover:bg-white/60"
    >
      {title}
    </button>
  );
}

interface PastChat {
  id: string;
  title: string;
}

interface SidebarProps {
  onNewChat?: () => void;
  /**
   * Dynamic list of chats the user has already walked through,
   * newest first. Rendered above the hardcoded decorative recents.
   * Clicking an entry is a no-op — these are purely visual history.
   */
  pastChats?: PastChat[];
}

export default function Sidebar({ onNewChat, pastChats }: SidebarProps) {
  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-rule bg-panel">
      <div className="flex items-center gap-2 px-4 py-4">
        <img
          src="/gemma-color.png"
          alt="Gemma"
          className="h-7 w-7 shrink-0 rounded-md object-contain"
        />
        <span className="text-sm font-semibold text-ink">Gemma Chat</span>
      </div>

      <div className="flex flex-col gap-0.5 px-2">
        <NavItem
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 5v14M5 12h14" />
            </svg>
          }
          label="New chat"
          active
          onClick={onNewChat}
        />
        <NavItem
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="7" />
              <path d="M21 21l-4.3-4.3" />
            </svg>
          }
          label="Search"
        />
        <NavItem
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 11-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09a1.65 1.65 0 00-1-1.51 1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 11-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09a1.65 1.65 0 001.51-1 1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
            </svg>
          }
          label="Settings"
        />
      </div>

      <div className="mt-2 flex-1 overflow-y-auto px-2 pb-4">
        <SectionLabel>Recents</SectionLabel>
        <div className="flex flex-col gap-0.5">
          {pastChats?.map((c) => <RecentItem key={c.id} title={c.title} />)}
          <RecentItem title="Uncertainty on token logits" />
          <RecentItem title="Streaming TextStreamer wiring" />
          <RecentItem title="System prompt experiments" />
          <RecentItem title="Calibration ideas for Gemma" />
        </div>
      </div>

      <div className="border-t border-rule px-3 py-3">
        <div className="flex items-center gap-3">
          <img
            src="/max.png"
            alt="Max"
            className="h-8 w-8 shrink-0 rounded-full object-cover"
          />
          <div className="flex min-w-0 flex-col">
            <span className="truncate text-sm text-ink">Max</span>
            <span className="truncate text-xs text-muted">probly developer</span>
          </div>
        </div>
      </div>
    </aside>
  );
}

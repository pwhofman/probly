import { useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';

interface PastChat {
  id: string;
  title: string;
}

// Truncate a chat title to ~30 visible characters with an ellipsis so
// long first messages don't blow out the sidebar column.
function truncate(text: string, max: number): string {
  const trimmed = text.trim();
  if (trimmed.length <= max) return trimmed;
  return trimmed.slice(0, max).trimEnd() + '\u2026';
}

export default function App() {
  // Remount ChatWindow by bumping this key to wipe its internal state —
  // used by the Sidebar's "New chat" button to reset the conversation.
  const [chatKey, setChatKey] = useState(0);
  // Growing list of chats the user has already walked through, oldest at
  // the bottom. The user never returns to a finished chat; entries are
  // purely visual history rendered in the sidebar.
  const [pastChats, setPastChats] = useState<PastChat[]>([]);
  // First user message of the currently-active chat, captured via the
  // ChatWindow ``onFirstMessage`` callback. Reset to null after each
  // snapshot so an empty "New chat" click doesn't push a stale title.
  const [currentTitle, setCurrentTitle] = useState<string | null>(null);

  const handleNewChat = () => {
    if (currentTitle !== null) {
      setPastChats((prev) => [
        { id: Math.random().toString(36).slice(2), title: currentTitle },
        ...prev,
      ]);
    }
    setCurrentTitle(null);
    setChatKey((k) => k + 1);
  };

  return (
    <div className="flex h-full bg-canvas">
      <Sidebar onNewChat={handleNewChat} pastChats={pastChats} />
      <main className="flex h-full flex-1 flex-col">
        <ChatWindow
          key={chatKey}
          onFirstMessage={(text) => setCurrentTitle(truncate(text, 30))}
        />
      </main>
    </div>
  );
}

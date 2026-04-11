import { useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';

export default function App() {
  // Remount ChatWindow by bumping this key to wipe its internal state —
  // used by the Sidebar's "New chat" button to reset the conversation.
  const [chatKey, setChatKey] = useState(0);
  const handleNewChat = () => setChatKey((k) => k + 1);

  return (
    <div className="flex h-full bg-canvas">
      <Sidebar onNewChat={handleNewChat} />
      <main className="flex h-full flex-1 flex-col">
        <ChatWindow key={chatKey} />
      </main>
    </div>
  );
}

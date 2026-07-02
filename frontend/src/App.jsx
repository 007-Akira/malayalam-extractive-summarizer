import {
  Clipboard,
  Clock3,
  FileText,
  MessageSquarePlus,
  RotateCcw,
  Send,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const HISTORY_KEY = "malayalam-summarizer-chats";

function loadChats() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function saveChats(chats) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(chats));
}

function makeChat() {
  const now = new Date().toISOString();
  return {
    id: crypto.randomUUID(),
    title: "New summary",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function titleFromText(text) {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return "Untitled summary";
  return clean.length > 48 ? `${clean.slice(0, 48)}...` : clean;
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export default function App() {
  const [chats, setChats] = useState(() => {
    const stored = loadChats();
    return stored.length ? stored : [makeChat()];
  });
  const [activeChatId, setActiveChatId] = useState(() => {
    const stored = loadChats();
    return stored[0]?.id;
  });
  const [draft, setDraft] = useState("");
  const [undoStack, setUndoStack] = useState([]);
  const [sentenceCount, setSentenceCount] = useState(3);
  const [dynamicMmr, setDynamicMmr] = useState(true);
  const [diversity, setDiversity] = useState(0.3);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [copiedMessageId, setCopiedMessageId] = useState("");

  const activeChat = useMemo(() => {
    return chats.find((chat) => chat.id === activeChatId) || chats[0];
  }, [activeChatId, chats]);

  function persist(nextChats) {
    setChats(nextChats);
    saveChats(nextChats);
  }

  function createChat() {
    const chat = makeChat();
    const nextChats = [chat, ...chats];
    persist(nextChats);
    setActiveChatId(chat.id);
    setDraft("");
    setUndoStack([]);
    setError("");
  }

  function deleteChat(chatId) {
    const nextChats = chats.filter((chat) => chat.id !== chatId);

    if (!nextChats.length) {
      const chat = makeChat();
      persist([chat]);
      setActiveChatId(chat.id);
    } else {
      persist(nextChats);
      if (activeChat.id === chatId) {
        setActiveChatId(nextChats[0].id);
      }
    }

    setDraft("");
    setUndoStack([]);
    setError("");
  }

  function updateDraft(value) {
    setUndoStack((stack) => (stack.at(-1) === draft ? stack : [...stack, draft]));
    setDraft(value);
  }

  function undoDraft() {
    setUndoStack((stack) => {
      if (!stack.length) return stack;
      const nextStack = stack.slice(0, -1);
      setDraft(stack[stack.length - 1]);
      return nextStack;
    });
  }

  async function copySummary(message) {
    const text = message.sentences?.join("\n") || message.content;
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(message.id);
      window.setTimeout(() => setCopiedMessageId(""), 1600);
    } catch {
      setError("Copy failed. Select the summary text and copy it manually.");
    }
  }

  async function submitSummary() {
    const text = draft.trim();
    if (!text || isLoading) return;

    setIsLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          sentence_count: sentenceCount,
          diversity: dynamicMmr ? "auto" : diversity,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Unable to summarize this article.");
      }

      const now = new Date().toISOString();
      const nextChats = chats.map((chat) => {
        if (chat.id !== activeChat.id) return chat;
        const firstMessage = chat.messages.length === 0;

        return {
          ...chat,
          title: firstMessage ? titleFromText(text) : chat.title,
          updatedAt: now,
          messages: [
            ...chat.messages,
            {
              id: crypto.randomUUID(),
              role: "user",
              content: text,
              createdAt: now,
            },
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: payload.summary,
              sentences: payload.sentences,
              createdAt: now,
              settings: {
                sentenceCount,
                diversity: dynamicMmr ? "auto" : diversity,
              },
            },
          ],
        };
      });

      persist(nextChats);
      setDraft("");
      setUndoStack([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">M</div>
          <div>
            <strong>Malayalam AI</strong>
            <span>Extractive workspace</span>
          </div>
        </div>

        <div className="sidebar-section">
          <button className="new-chat-button" onClick={createChat}>
            <MessageSquarePlus size={18} />
            New chat
          </button>
        </div>

        <div className="sidebar-section">
          <div className="section-label">Previous chats</div>
          <div className="history-list">
            {chats.map((chat) => (
              <div
                className={`history-item ${chat.id === activeChat.id ? "active" : ""}`}
                key={chat.id}
              >
                <button className="history-main" onClick={() => setActiveChatId(chat.id)}>
                  <span>{chat.title}</span>
                  <small>
                    <Clock3 size={13} />
                    {formatDate(chat.updatedAt)}
                  </small>
                </button>
                <button
                  aria-label="Delete chat"
                  className="history-delete"
                  onClick={() => deleteChat(chat.id)}
                  title="Delete chat"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="sidebar-section settings">
          <div className="section-label">
            <SlidersHorizontal size={15} />
            Settings
          </div>

          <label>
            <span className="setting-topline">
              Sentence count
              <strong>{sentenceCount}</strong>
            </span>
            <input
              min="1"
              max="10"
              type="range"
              value={sentenceCount}
              onChange={(event) => setSentenceCount(Number(event.target.value))}
            />
          </label>

          <label className="check-row">
            <input
              checked={dynamicMmr}
              type="checkbox"
              onChange={(event) => setDynamicMmr(event.target.checked)}
            />
            Dynamic MMR
          </label>

          {!dynamicMmr && (
            <label>
              <span className="setting-topline">
                Diversity penalty
                <strong>{diversity.toFixed(1)}</strong>
              </span>
              <input
                min="0"
                max="1"
                step="0.1"
                type="range"
                value={diversity}
                onChange={(event) => setDiversity(Number(event.target.value))}
              />
            </label>
          )}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>Malayalam Extractive Summarizer</h1>
            <p>Summarize Malayalam articles in a clean chat workspace with saved history.</p>
          </div>
          <div className="topbar-status">
            <span>{activeChat.messages.filter((message) => message.role === "assistant").length}</span>
            summaries
          </div>
        </header>

        <section className="conversation">
          {activeChat.messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon">
                <FileText size={26} />
              </div>
              <h2>Start with an article</h2>
              <p>Paste a Malayalam news article below. Each result is saved as a chat so you can continue or revisit summaries later.</p>
            </div>
          ) : (
            activeChat.messages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="message-header">
                  <div>
                    <div className="message-label">{message.role === "user" ? "Article" : "Summary"}</div>
                    <span>{formatDate(message.createdAt)}</span>
                  </div>
                  {message.role === "assistant" && (
                    <button className="icon-button" onClick={() => copySummary(message)} title="Copy summary">
                      <Clipboard size={16} />
                      {copiedMessageId === message.id ? "Copied" : "Copy"}
                    </button>
                  )}
                </div>
                {message.role === "user" ? (
                  <p>{message.content}</p>
                ) : (
                  <ul>
                    {message.sentences.map((sentence, index) => (
                      <li key={`${message.id}-${index}`}>{sentence}</li>
                    ))}
                  </ul>
                )}
              </article>
            ))
          )}
          {isLoading && (
            <article className="message assistant typing-card">
              <div className="message-header">
                <div>
                  <div className="message-label">Summary</div>
                  <span>Generating</span>
                </div>
              </div>
              <div className="typing-lines" aria-label="Generating summary">
                <span />
                <span />
                <span />
              </div>
            </article>
          )}
        </section>

        <section className="composer">
          <div className="composer-header">
            <span>Article input</span>
            <small>{draft.trim().split(/\s+/).filter(Boolean).length} words</small>
          </div>
          <textarea
            value={draft}
            onChange={(event) => updateDraft(event.target.value)}
            placeholder="Paste the full Malayalam article here..."
          />
          {error && <div className="error">{error}</div>}
          <div className="composer-actions">
            <button className="secondary-button" disabled={!undoStack.length} onClick={undoDraft}>
              <RotateCcw size={17} />
              Undo
            </button>
            <button className="send-button" disabled={!draft.trim() || isLoading} onClick={submitSummary}>
              <Send size={17} />
              {isLoading ? "Summarizing..." : "Summarize"}
            </button>
          </div>
        </section>
      </section>
    </main>
  );
}

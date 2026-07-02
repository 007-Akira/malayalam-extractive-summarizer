import {
  Check,
  ChevronDown,
  ChevronRight,
  Clipboard,
  Clock3,
  FileText,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  RotateCcw,
  Send,
  SlidersHorizontal,
  Trash2,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const HISTORY_KEY = "malayalam-summarizer-chats";
const WORD_WARNING_LIMIT = 900;
const WORD_DANGER_LIMIT = 1200;
const MODEL_OPTIONS = [
  { value: "chotta_bheem_v2", label: "Chotta Bheem V2" },
  { value: "chotta_bheem", label: "Chotta Bheem" },
  { value: "hybrid_classifier", label: "Hybrid Classifier" },
  { value: "muril_classifier", label: "MuRIL Classifier" },
  { value: "sentence_classifier", label: "Sentence Classifier" },
];
const MODEL_LABELS = Object.fromEntries(MODEL_OPTIONS.map((option) => [option.value, option.label]));

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

function clampSentenceCount(value) {
  const parsed = Number(value);
  if (Number.isNaN(parsed)) return 1;
  return Math.min(10, Math.max(1, parsed));
}

function formatSummaryText(message) {
  if (message.settings?.format === "paragraph") {
    return message.content;
  }

  return message.sentences?.join("\n") || message.content;
}

function modelLabel(modelKey) {
  return MODEL_LABELS[modelKey] || modelKey || "Selected model";
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
  const [selectedModel, setSelectedModel] = useState("chotta_bheem");
  const [dynamicMmr, setDynamicMmr] = useState(true);
  const [diversity, setDiversity] = useState(0.3);
  const [summaryFormat, setSummaryFormat] = useState("bullets");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [copiedMessageId, setCopiedMessageId] = useState("");
  const [renamingChatId, setRenamingChatId] = useState("");
  const [renameValue, setRenameValue] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [settingsCollapsed, setSettingsCollapsed] = useState(false);

  const activeChat = useMemo(() => {
    return chats.find((chat) => chat.id === activeChatId) || chats[0];
  }, [activeChatId, chats]);
  const draftWordCount = useMemo(() => draft.trim().split(/\s+/).filter(Boolean).length, [draft]);
  const wordCountTone =
    draftWordCount >= WORD_DANGER_LIMIT ? "danger" : draftWordCount >= WORD_WARNING_LIMIT ? "warn" : "";

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

  function startRename(chat) {
    setRenamingChatId(chat.id);
    setRenameValue(chat.title);
  }

  function cancelRename() {
    setRenamingChatId("");
    setRenameValue("");
  }

  function saveRename(chatId) {
    const nextTitle = renameValue.trim();
    if (!nextTitle) {
      cancelRename();
      return;
    }

    const nextChats = chats.map((chat) => (chat.id === chatId ? { ...chat, title: nextTitle } : chat));
    persist(nextChats);
    cancelRename();
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
    const text = formatSummaryText(message);
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
          model: selectedModel,
        }),
      });

      const responseText = await response.text();
      let payload = {};

      try {
        payload = responseText ? JSON.parse(responseText) : {};
      } catch {
        payload = { detail: responseText };
      }

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
                format: summaryFormat,
                model: payload.model || selectedModel,
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
    <main className={`app ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">M</div>
          <div className="brand-copy">
            <strong>Malayalam News Extractive Summarizer</strong>
          </div>
          <button
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
          </button>
        </div>

        <div className="sidebar-section">
          <button className="new-chat-button" onClick={createChat} title="New chat">
            <MessageSquarePlus size={18} />
            <span>New chat</span>
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
                {renamingChatId === chat.id ? (
                  <div className="rename-row">
                    <input
                      autoFocus
                      value={renameValue}
                      onChange={(event) => setRenameValue(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") saveRename(chat.id);
                        if (event.key === "Escape") cancelRename();
                      }}
                    />
                    <button aria-label="Save chat name" onClick={() => saveRename(chat.id)} title="Save">
                      <Check size={14} />
                    </button>
                    <button aria-label="Cancel rename" onClick={cancelRename} title="Cancel">
                      <X size={14} />
                    </button>
                  </div>
                ) : (
                  <button className="history-main" onClick={() => setActiveChatId(chat.id)}>
                    <span>{chat.title}</span>
                    <small>
                      <Clock3 size={13} />
                      {formatDate(chat.updatedAt)}
                    </small>
                  </button>
                )}
                {renamingChatId !== chat.id && (
                  <button
                    aria-label="Rename chat"
                    className="history-action"
                    onClick={() => startRename(chat)}
                    title="Rename chat"
                  >
                    <Pencil size={15} />
                  </button>
                )}
                {renamingChatId !== chat.id && (
                  <button
                    aria-label="Delete chat"
                    className="history-action history-delete"
                    onClick={() => deleteChat(chat.id)}
                    title="Delete chat"
                  >
                    <Trash2 size={15} />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className={`sidebar-section settings ${settingsCollapsed ? "collapsed" : ""}`}>
          <button
            aria-expanded={!settingsCollapsed}
            className="settings-toggle"
            onClick={() => setSettingsCollapsed((collapsed) => !collapsed)}
            title={settingsCollapsed ? "Expand settings" : "Collapse settings"}
            type="button"
          >
            <span className="section-label">
              <SlidersHorizontal size={15} />
              Settings
            </span>
            {settingsCollapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
          </button>

          {!settingsCollapsed && (
            <div className="settings-body">
              <label>
                <span className="setting-topline">Model</span>
                <select
                  className="model-select"
                  disabled={isLoading}
                  value={selectedModel}
                  onChange={(event) => setSelectedModel(event.target.value)}
                >
                  {MODEL_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

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
                <input
                  aria-label="Exact sentence count"
                  className="number-input"
                  min="1"
                  max="10"
                  type="number"
                  value={sentenceCount}
                  onChange={(event) => setSentenceCount(clampSentenceCount(event.target.value))}
                />
              </label>

              <div className="format-toggle">
                <span>Format</span>
                <div>
                  <button
                    className={summaryFormat === "bullets" ? "active" : ""}
                    onClick={() => setSummaryFormat("bullets")}
                    type="button"
                  >
                    Bullets
                  </button>
                  <button
                    className={summaryFormat === "paragraph" ? "active" : ""}
                    onClick={() => setSummaryFormat("paragraph")}
                    type="button"
                  >
                    Paragraph
                  </button>
                </div>
              </div>

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
          )}
        </div>
      </aside>

      <section className="workspace">
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
                    <span>
                      {formatDate(message.createdAt)}
                      {message.role === "assistant" && (
                        <small className="model-chip">{modelLabel(message.settings?.model)}</small>
                      )}
                    </span>
                  </div>
                  <div className="message-actions">
                    <button
                      aria-label={`Copy ${message.role === "user" ? "article" : "summary"}`}
                      className={`icon-button ${copiedMessageId === message.id ? "copied" : ""}`}
                      onClick={() => copySummary(message)}
                      title={copiedMessageId === message.id ? "Copied" : "Copy"}
                    >
                      <Clipboard size={16} />
                    </button>
                  </div>
                </div>
                {message.role === "user" ? (
                  <p>{message.content}</p>
                ) : message.settings?.format === "paragraph" ? (
                  <p className="summary-paragraph">{message.content}</p>
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
            <small className={`word-count ${wordCountTone}`}>
              {draftWordCount} words
              {wordCountTone === "warn" && " - getting long"}
              {wordCountTone === "danger" && " - consider shortening"}
            </small>
          </div>
          <textarea
            value={draft}
            onChange={(event) => updateDraft(event.target.value)}
            placeholder="Paste the full Malayalam article here..."
          />
          {error && <div className="error">{error}</div>}
          <div className="composer-actions">
            <button className="secondary-button" disabled={!undoStack.length} onClick={undoDraft} title="Undo">
              <RotateCcw size={17} />
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

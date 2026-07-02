import json
from datetime import datetime
from html import escape
from pathlib import Path
from uuid import uuid4

import streamlit as st

from summarize import summarize_article


HISTORY_FILE = Path(".summarizer_chat_history.json")


def now_label():
    return datetime.now().strftime("%d %b %Y, %I:%M %p")


def make_chat(title="New summary"):
    timestamp = now_label()
    return {
        "id": uuid4().hex,
        "title": title,
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
    }


def load_chats():
    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    return data if isinstance(data, list) else []


def save_chats(chats):
    HISTORY_FILE.write_text(
        json.dumps(chats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_chat(chat_id):
    for chat in st.session_state.chats:
        if chat["id"] == chat_id:
            return chat
    return None


def create_new_chat():
    chat = make_chat()
    st.session_state.chats.insert(0, chat)
    st.session_state.current_chat_id = chat["id"]
    st.session_state.article_input = ""
    st.session_state.last_draft = ""
    st.session_state.draft_stack = []
    save_chats(st.session_state.chats)


def remember_draft():
    current = st.session_state.get("article_input", "")
    last = st.session_state.get("last_draft", "")

    if current != last:
        stack = st.session_state.setdefault("draft_stack", [])
        if not stack or stack[-1] != last:
            stack.append(last)
        st.session_state.last_draft = current


def undo_draft():
    stack = st.session_state.setdefault("draft_stack", [])
    if stack:
        previous = stack.pop()
        st.session_state.article_input = previous
        st.session_state.last_draft = previous


def title_from_article(text):
    cleaned = " ".join(text.split())
    if not cleaned:
        return "Untitled summary"
    return cleaned[:46] + ("..." if len(cleaned) > 46 else "")


def summary_html(sentences):
    items = "".join(f"<li>{escape(sentence)}</li>" for sentence in sentences)
    return f"<ul>{items}</ul>"


st.set_page_config(
    page_title="Malayalam Extractive Summarizer",
    page_icon="M",
    layout="wide",
)

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "current_chat_id" not in st.session_state:
    if st.session_state.chats:
        st.session_state.current_chat_id = st.session_state.chats[0]["id"]
    else:
        create_new_chat()

if "article_input" not in st.session_state:
    st.session_state.article_input = ""

if "last_draft" not in st.session_state:
    st.session_state.last_draft = st.session_state.article_input

if "draft_stack" not in st.session_state:
    st.session_state.draft_stack = []

if st.session_state.pop("clear_composer_next", False):
    st.session_state.article_input = ""
    st.session_state.last_draft = ""
    st.session_state.draft_stack = []

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }

    .stApp {
        background: #f7f8fb;
        color: #18202f;
    }

    .block-container {
        max-width: 1180px;
        padding: 1.6rem 2rem 2.4rem;
    }

    [data-testid="stHeader"], #MainMenu, footer {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e6ef;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.85rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #121826;
        letter-spacing: 0;
    }

    .app-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        border-bottom: 1px solid #e2e6ef;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }

    .app-header h1 {
        font-size: clamp(1.7rem, 3vw, 2.6rem);
        line-height: 1.05;
        margin: 0;
        font-weight: 760;
    }

    .app-header p {
        color: #5a6475;
        line-height: 1.6;
        margin: 0.55rem 0 0;
        max-width: 740px;
    }

    .welcome {
        display: grid;
        min-height: 34vh;
        place-items: center;
        text-align: center;
    }

    .welcome-inner {
        max-width: 680px;
    }

    .welcome-inner h2 {
        font-size: clamp(2rem, 5vw, 3.8rem);
        line-height: 1;
        margin: 0 0 0.8rem;
        font-weight: 760;
    }

    .welcome-inner p {
        color: #5b6678;
        font-size: 1rem;
        line-height: 1.7;
        margin: 0 auto;
        max-width: 560px;
    }

    .chat-list {
        display: grid;
        gap: 1rem;
        margin: 0.5rem 0 1.15rem;
    }

    .message {
        border-radius: 8px;
        border: 1px solid #e1e6ef;
        box-shadow: 0 14px 38px rgba(20, 31, 53, 0.06);
        padding: 1rem 1.1rem;
    }

    .message-user {
        background: #ffffff;
    }

    .message-assistant {
        background: #f9fbff;
        border-left: 4px solid #275efe;
    }

    .message-meta {
        color: #667085;
        font-size: 0.78rem;
        font-weight: 720;
        letter-spacing: 0.05em;
        margin-bottom: 0.65rem;
        text-transform: uppercase;
    }

    .article-preview {
        color: #243047;
        line-height: 1.7;
        max-height: 9.5rem;
        overflow: auto;
        white-space: pre-wrap;
    }

    .message-assistant ul {
        color: #243047;
        line-height: 1.75;
        margin: 0;
        padding-left: 1.2rem;
    }

    .message-assistant li {
        margin-bottom: 0.65rem;
    }

    .message-assistant li:last-child {
        margin-bottom: 0;
    }

    .composer {
        position: relative;
        background: #ffffff;
        border: 1px solid #e1e6ef;
        border-radius: 8px;
        box-shadow: 0 16px 42px rgba(20, 31, 53, 0.08);
        padding: 1rem;
        margin-top: 0.6rem;
    }

    .composer::after {
        content: "";
        position: absolute;
        top: 46%;
        right: -13px;
        width: 24px;
        height: 24px;
        background: #ffffff;
        border-top: 1px solid #e1e6ef;
        border-right: 1px solid #e1e6ef;
        transform: translateY(-50%) rotate(45deg);
        box-shadow: 8px -8px 18px rgba(20, 31, 53, 0.04);
        z-index: 2;
    }

    .composer::before {
        content: "";
        position: absolute;
        top: 46%;
        right: -4px;
        width: 8px;
        height: 72px;
        background: #ffffff;
        transform: translateY(-50%);
        z-index: 3;
    }

    .composer-label {
        color: #172033;
        font-size: 0.78rem;
        font-weight: 760;
        letter-spacing: 0.06em;
        margin-bottom: 0.85rem;
        text-transform: uppercase;
    }

    .stTextArea textarea {
        background: #fbfcff !important;
        color: #18202f !important;
        border: 1px solid #d8dee9 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        font-size: 1rem !important;
        line-height: 1.65 !important;
        min-height: 230px;
    }

    .stTextArea textarea:focus {
        border-color: #275efe !important;
        box-shadow: 0 0 0 3px rgba(39, 94, 254, 0.12) !important;
    }

    .stTextArea label, .stSlider label, .stCheckbox label, .stSelectbox label {
        color: #283348 !important;
        font-weight: 650;
    }

    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 760;
        padding: 0.78rem 0.95rem;
    }

    .stButton button[kind="primary"] {
        background: #172033 !important;
        border: 1px solid #172033 !important;
        color: #ffffff !important;
        box-shadow: 0 12px 24px rgba(23, 32, 51, 0.13);
    }

    .stButton button[kind="primary"]:hover {
        background: #26344f !important;
        border-color: #26344f !important;
        color: #ffffff !important;
    }

    .stAlert {
        border-radius: 8px;
    }

    .sidebar-title {
        color: #121826;
        font-size: 0.78rem;
        font-weight: 780;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .history-meta {
        color: #667085;
        font-size: 0.82rem;
        line-height: 1.55;
        margin-top: -0.25rem;
    }

    @media (max-width: 860px) {
        .block-container {
            padding: 1.2rem 1rem 2rem;
        }

        .app-header {
            display: block;
        }

        .composer::before,
        .composer::after {
            display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-title">Workspace</div>', unsafe_allow_html=True)
    st.button("New chat", use_container_width=True, on_click=create_new_chat)

    st.markdown('<div class="sidebar-title">Previous chats</div>', unsafe_allow_html=True)
    if st.session_state.chats:
        chat_options = {chat["id"]: chat["title"] for chat in st.session_state.chats}
        selected_chat = st.selectbox(
            "Chat history",
            options=list(chat_options.keys()),
            format_func=lambda chat_id: chat_options[chat_id],
            index=list(chat_options.keys()).index(st.session_state.current_chat_id),
            label_visibility="collapsed",
        )
        st.session_state.current_chat_id = selected_chat
        active_chat = get_chat(selected_chat)
        if active_chat:
            st.markdown(
                f"<div class='history-meta'>Updated {escape(active_chat['updated_at'])}</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)
    k_sentences = st.slider(
        "Sentence count",
        min_value=1,
        max_value=10,
        value=3,
        help="Choose how many sentences the summarizer returns.",
    )
    use_dmmr = st.checkbox(
        "Dynamic MMR",
        value=True,
        help="Automatically balances relevance and redundancy.",
    )
    if use_dmmr:
        diversity_param = "auto"
        st.info("Auto diversity tuning is enabled.")
    else:
        diversity_param = st.slider(
            "Diversity penalty",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            help="Higher values reduce redundancy more aggressively.",
        )

current_chat = get_chat(st.session_state.current_chat_id)
if current_chat is None:
    create_new_chat()
    current_chat = get_chat(st.session_state.current_chat_id)

st.markdown(
    """
    <div class="app-header">
        <div>
            <h1>Malayalam Extractive Summarizer</h1>
            <p>A focused chat workspace for turning Malayalam articles into concise extractive summaries.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

messages = current_chat.get("messages", [])

if not messages:
    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-inner">
                <h2>Start with an article</h2>
                <p>Paste a Malayalam news article below. Each summary is saved as a chat so you can return to earlier work from the sidebar.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="chat-list">', unsafe_allow_html=True)
    for message in messages:
        if message["role"] == "user":
            st.markdown(
                f"""
                <div class="message message-user">
                    <div class="message-meta">Article</div>
                    <div class="article-preview">{escape(message["content"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="message message-assistant">
                    <div class="message-meta">Summary</div>
                    {summary_html(message.get("sentences", []))}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="composer"><div class="composer-label">Article Input</div>', unsafe_allow_html=True)
st.text_area(
    "Malayalam article text",
    key="article_input",
    height=250,
    label_visibility="collapsed",
    on_change=remember_draft,
    placeholder="Paste the full Malayalam article here...",
)

undo_col, submit_col = st.columns([1, 3])
with undo_col:
    st.button(
        "Undo",
        use_container_width=True,
        disabled=not st.session_state.draft_stack,
        on_click=undo_draft,
    )
with submit_col:
    summarize_clicked = st.button("Summarize", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if summarize_clicked:
    raw_text = st.session_state.article_input.strip()

    if not raw_text:
        st.warning("Paste the Malayalam article text before summarizing.")
    else:
        with st.spinner("Analyzing article..."):
            try:
                summary_text, extracted_list = summarize_article(
                    raw_text,
                    k=k_sentences,
                    diversity=diversity_param,
                )

                if "Article is too short" in summary_text:
                    st.warning(summary_text)
                else:
                    timestamp = now_label()
                    if not current_chat["messages"]:
                        current_chat["title"] = title_from_article(raw_text)

                    current_chat["messages"].append(
                        {
                            "role": "user",
                            "content": raw_text,
                            "created_at": timestamp,
                        }
                    )
                    current_chat["messages"].append(
                        {
                            "role": "assistant",
                            "content": summary_text,
                            "sentences": extracted_list,
                            "created_at": timestamp,
                            "settings": {
                                "k": k_sentences,
                                "diversity": diversity_param,
                            },
                        }
                    )
                    current_chat["updated_at"] = timestamp

                    st.session_state.clear_composer_next = True
                    save_chats(st.session_state.chats)
                    st.rerun()

            except Exception as e:
                st.error(f"An error occurred while summarizing: {e}")

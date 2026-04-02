import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Ebad's Assistant", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', 'Segoe UI', sans-serif; }

    .stApp {
        background: #0f1117;
    }

    /* Header */
    .header-container {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        border-bottom: 1px solid #2a2d3a;
        margin-bottom: 1.5rem;
    }
    .header-title {
        font-size: 2.2em;
        font-weight: 700;
        color: #e8eaf6;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }
    .header-title span {
        background: linear-gradient(90deg, #5b8dee, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-subtitle {
        color: #6b7280;
        font-size: 0.95em;
        font-weight: 400;
        letter-spacing: 1px;
    }
    .badge {
        display: inline-block;
        background: #1e2130;
        border: 1px solid #2a2d3a;
        border-radius: 20px;
        padding: 3px 12px;
        color: #9ca3af;
        font-size: 0.78em;
        margin: 0.4rem 0.2rem;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        font-size: 16px;
        line-height: 1.6;
        border: 1px solid #2a2d3a;
    }
    [data-testid="stChatMessage"][data-testid*="user"],
    div[data-testid="stChatMessage"]:has(img[alt="user"]) {
        background: #1a1f2e;
    }
    div[data-testid="stChatMessage"]:has(img[alt="assistant"]) {
        background: #161b27;
    }
    [data-testid="stChatMessage"] p {
        color: #e2e8f0 !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border: 1px solid #4A90E2 !important;
        border-radius: 12px !important;
        background: #1e2130 !important;
        padding: 12px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #e2e8f0 !important;
        font-size: 15px !important;
        background: transparent !important;
    }

    /* Sidebar */
    div[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #2a2d3a;
    }
    div[data-testid="stSidebar"] * {
        color: #9ca3af !important;
    }
    .stSelectbox label { color: #9ca3af !important; }

    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: #2a2d3a;
        margin: 1rem 0;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        color: #4b5563;
        padding: 3rem 0;
        font-size: 0.95em;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h3 style='color:#e2e8f0 !important; font-size:1.1em;'>⚙️ Settings</h3>", unsafe_allow_html=True)
    model_name = st.selectbox("AI Model", [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768"
    ])
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.context = ""
        st.rerun()
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4b5563 !important; font-size:0.78em; text-align:center;'>Powered by Groq + LangChain<br>Built by Ebad</p>", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class='header-container'>
        <div class='header-title'>🤖 <span>Ebad's</span> Assistant</div>
        <div class='header-subtitle'>PERSONAL AI — PRIVATE · FAST · FREE</div>
        <div style='margin-top:0.8rem'>
            <span class='badge'>⚡ Groq</span>
            <span class='badge'>🔗 LangChain</span>
            <span class='badge'>🦙 Llama 3</span>
            <span class='badge'>🔒 Private</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Model + Chain
template = """You are Ebad's personal AI assistant. You are professional, smart, and friendly.
Always refer to the user as 'Ebad (Human)' in your greetings and responses.
Example: 'Hi Ebad (Human), how can I help you today?'

Conversation history:
{context}

Ebad (Human): {question}
Assistant:"""

api_key = st.secrets["GROQ_API_KEY"]
model = ChatGroq(model=model_name, api_key=api_key)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Session state
if "context" not in st.session_state:
    st.session_state.context = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history
if not st.session_state.messages:
    st.markdown("<div class='empty-state'>👋 Hello Ebad (Human)! Ask me anything...</div>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"<p style='color:#e2e8f0'>{msg['content']}</p>", unsafe_allow_html=True)

# Input
if question := st.chat_input("Message Ebad's Assistant..."):
    with st.chat_message("user"):
        st.markdown(f"<p style='color:#e2e8f0'>{question}</p>", unsafe_allow_html=True)
    with st.spinner("Thinking..."):
        response = chain.invoke({"context": st.session_state.context, "question": question})
        answer = response.content
    with st.chat_message("assistant"):
        st.markdown(f"<p style='color:#e2e8f0'>{answer}</p>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.context += f"\nEbad (Human): {question}\nAssistant: {answer}"
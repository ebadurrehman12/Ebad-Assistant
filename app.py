import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Ebad's Assistant", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Rajdhani:wght@400;600&display=swap');
    
    .stApp { 
        background: linear-gradient(135deg, #0a0a0a 0%, #0d1b2a 50%, #0a0a0a 100%);
    }
    
    .hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    
    .hero h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8em;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff006e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2em;
    }
    
    .hero p {
        color: #888;
        font-size: 1.1em;
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 2px;
    }
    
    .badge {
        display: inline-block;
        background: linear-gradient(90deg, #00d4ff22, #7b2ff722);
        border: 1px solid #00d4ff55;
        border-radius: 20px;
        padding: 4px 16px;
        color: #00d4ff;
        font-size: 0.85em;
        margin: 0.5rem 0.3rem;
    }
    
    [data-testid="stChatMessage"] { 
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px; 
        padding: 14px;
        border-left: 3px solid #00d4ff;
        margin: 8px 0;
    }
    
    [data-testid="stChatMessage"] p { 
        color: #e0e0e0 !important; 
        font-size: 15px;
        font-family: 'Rajdhani', sans-serif;
    }
    
    .stChatInputContainer {
        border-top: 1px solid #00d4ff33 !important;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a, #0a0a1a);
        border-right: 1px solid #00d4ff33;
    }
    
    .stSelectbox label { color: #00d4ff !important; }
    .stButton button {
        background: linear-gradient(90deg, #ff006e, #7b2ff7) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff006e);
        border-radius: 2px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='color:#00d4ff; font-family:Orbitron'>⚙️ Controls</h2>", unsafe_allow_html=True)
    model_name = st.selectbox("🧠 AI Model", ["llama3.2", "llama2"])
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.context = ""
        st.rerun()
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#555; font-size:0.8em; text-align:center'>Powered by<br><span style='color:#00d4ff'>Llama 3.2 + LangChain</span><br>Running 100% locally</p>", unsafe_allow_html=True)

st.markdown("""
    <div class='hero'>
        <h1>🤖 Ebad's Assistant</h1>
        <p>PERSONAL AI — PRIVATE • FAST • FREE</p>
        <div>
            <span class='badge'>⚡ Local AI</span>
            <span class='badge'>🔒 Private</span>
            <span class='badge'>🧠 LangChain</span>
            <span class='badge'>🦙 Llama 3.2</span>
        </div>
    </div>
    <div class='divider'></div>
""", unsafe_allow_html=True)

template = "You are Ebad's helpful personal assistant. Be smart, concise and friendly. History: {context} Question: {question} Answer:"
model = OllamaLLM(model=model_name)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

if "context" not in st.session_state:
    st.session_state.context = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("<p style='text-align:center; color:#555; padding:2rem'>👋 Hello! Ask me anything...</p>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"<p style='color:#e0e0e0'>{msg['content']}</p>", unsafe_allow_html=True)

if question := st.chat_input("💬 Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(f"<p style='color:#e0e0e0'>{question}</p>", unsafe_allow_html=True)
    with st.spinner("🧠 Thinking..."):
        response = chain.invoke({"context": st.session_state.context, "question": question})
    with st.chat_message("assistant"):
        st.markdown(f"<p style='color:#e0e0e0'>{response}</p>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.context += f"You: {question} Bot: {response}"
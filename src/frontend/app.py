import os
import uuid
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
RUN_ENDPOINT = f"{BACKEND_URL}/api/v1/agent/run"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"

st.set_page_config(page_title="Agent Chat", page_icon="🤖", layout="centered")
st.title("AI Agent")

@st.cache_data(ttl=10.0)
def ping_health(url: str) -> str:
    try:
        r = requests.get(url, timeout=5)
        if r.ok:
            return "ok"
        return f"bad ({r.status_code})"
    except Exception as e:
        return f"error: {e}"

status = ping_health(HEALTH_ENDPOINT)
st.caption(f"Backend health: **{status}** · {BACKEND_URL}")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! Type down your request, i'll pass it to the agent!"}
    ]

def call_agent_api(prompt: str) -> str:
    try:
        resp = requests.post(
            RUN_ENDPOINT,
            json={"input": prompt},
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("output", "")
        return f"Backend Error: {resp.status_code} — {resp.text[:500]}"
    except requests.RequestException as e:
        return f"Failed to access backend: {e}"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_text = st.chat_input("Send message to agent...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking..._")
        reply = call_agent_api(user_text)
        placeholder.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

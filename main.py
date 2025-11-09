import streamlit as st
import asyncio
import requests
from openai import OpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents.run import RunConfig

# OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set.")
openai_client = OpenAI(api_key=openai_api_key)

model = OpenAIChatCompletionsModel(model="gpt‑4", openai_client=openai_client)
config = RunConfig(model=model, model_provider=openai_client, tracing_disabled=True)
agent = Agent(
    name="Assistant",
    instructions=("You are a helpful assistant. If user wants web‑search or deep research, "
                  "you should fetch real results or elaborate deeply accordingly."),
    model=model
)

async def get_agent_response(user_input):
    result = await Runner.run(agent, user_input, run_config=config)
    return result.final_output

def run_asyncio_task(task):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        return asyncio.ensure_future(task)
    else:
        return asyncio.run(task)

# UI
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")
st.title("🤖 AI Chat Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "attachments" not in st.session_state:
    st.session_state.attachments = []

if "mode" not in st.session_state:
    st.session_state.mode = "Normal"  # Normal or Research

# Mode selector
mode = st.radio("Choose mode:", ["Normal", "Research"], index=0)
st.session_state.mode = mode

# Attachment upload
with st.expander("+ Attach files/images"):
    uploaded_files = st.file_uploader("Choose files/images", accept_multiple_files=True)
    if uploaded_files:
        st.session_state.attachments = uploaded_files

# Display chat history
for chat in st.session_state.chat_history:
    if chat["sender"] == "user":
        st.markdown(f"**🧑‍💻 You:** {chat['message']}")
    else:
        st.markdown(f"**🤖 Assistant:** {chat['message']}")
        if chat.get("attachments"):
            for file in chat["attachments"]:
                if file.type.startswith("image"):
                    st.image(file)
                else:
                    st.markdown(f"[Download {file.name}](data:file;base64,{file.getvalue().hex()})")

user_input = st.text_input("💬 Ask Your Query")

if st.button("✨ Send"):
    if user_input:
        st.session_state.chat_history.append({"sender": "user", "message": user_input, "mode": mode})
        # Build input
        full_input = f"Mode: {mode}\nUser: {user_input}"
        if st.session_state.attachments:
            file_names = [f.name for f in st.session_state.attachments]
            full_input += f"\nUploaded files: {', '.join(file_names)}"
        # If research mode => perform web search first
        if mode == "Research":
            search_query = user_input
            # Example: simple web search using requests (you will need proper API)
            res = requests.get(f"https://api.example.com/search?q={search_query}")  # placeholder
            # parse results
            results = res.json().get("results", [])
            summary = "Search results:\n" + "\n".join([f"{r['title']}: {r['link']}" for r in results[:3]])
            full_input += "\n\n" + summary
        with st.spinner("Agent is thinking... 💭"):
            response = run_asyncio_task(get_agent_response(full_input))
            if asyncio.isfuture(response):
                response = asyncio.run(response)
        chat_data = {"sender": "assistant", "message": response}
        if st.session_state.attachments:
            chat_data["attachments"] = st.session_state.attachments
            st.session_state.attachments = []
        st.session_state.chat_history.append(chat_data)
        st.experimental_rerun()
    else:
        st.warning("Please type something 😘")

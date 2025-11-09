import streamlit as st
import asyncio
from openai import OpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents.run import RunConfig

# --- OpenAI API Setup ---
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set.")

openai_client = OpenAI(api_key=openai_api_key)

# Model setup
model = OpenAIChatCompletionsModel(
    model="gpt-4o",
    openai_client=openai_client
)

config = RunConfig(
    model=model,
    model_provider=openai_client,
    tracing_disabled=True
)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. Always respond clearly, fully answering questions. "
                 "If the user uploads a file or asks to search the web, handle it accordingly.",
    model=model
)

# --- Async agent response ---
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

# --- Streamlit UI ---
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")

st.title("🤖 AI Chat Assistant")

# Session state for chat & attachments
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "attachments" not in st.session_state:
    st.session_state.attachments = []

# Display chat messages
for chat in st.session_state.chat_history:
    if chat["sender"] == "user":
        st.markdown(f"**🧑‍💻 You:** {chat['message']}")
    else:
        st.markdown(f"**🤖 Assistant:** {chat['message']}")
        if "attachments" in chat:
            for file in chat["attachments"]:
                if file.type.startswith("image"):
                    st.image(file)
                else:
                    st.markdown(f"[Download {file.name}](data:file;base64,{file.getvalue().hex()})")

# "+" Button for attachments
col1, col2 = st.columns([0.1,0.9])
with col1:
    attach_btn = st.button("+")
uploaded_files = []
if attach_btn:
    uploaded_files = st.file_uploader("Attach files/images", accept_multiple_files=True)

# User input
user_input = st.text_input("💬 Ask Your Query")

if st.button("✨ Send"):
    if user_input:
        # Add user message
        st.session_state.chat_history.append({"sender": "user", "message": user_input})

        # Add attachments if any
        if uploaded_files:
            st.session_state.attachments = uploaded_files

        with st.spinner("Agent is processing... 💭"):
            # Include file names in input if attached
            full_input = user_input
            if uploaded_files:
                file_names = [f.name for f in uploaded_files]
                full_input += f"\n\nUser uploaded files: {', '.join(file_names)}"

            # --- Run agent ---
            response = run_asyncio_task(get_agent_response(full_input))
            if asyncio.isfuture(response):
                response = asyncio.run(response)

        # Add assistant response
        chat_data = {"sender": "assistant", "message": response}
        if uploaded_files:
            chat_data["attachments"] = uploaded_files
        st.session_state.chat_history.append(chat_data)

        # Clear uploaded files
        st.session_state.attachments = []

        # Refresh chat
        st.experimental_rerun()
    else:
        st.warning("Please type something 😘")

# --- Optional: Web Search Simulation ---
# If user types "search:" at start of input, do web search simulation
if user_input.lower().startswith("search:"):
    query = user_input[7:].strip()
    st.info(f"🌐 Web search simulated for: {query}")
    st.session_state.chat_history.append({
        "sender": "assistant",
        "message": f"Here are the simulated search results for '{query}':\n1. Result A\n2. Result B\n3. Result C"
    })
    st.experimental_rerun()

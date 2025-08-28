import streamlit as st
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import asyncio

# Gemini API Key (move to env or secrets for production)
gemini_api_key =  # "Enter your api_key"

# Check API key
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

# Client setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=model
)

# Async function to get agent's response
async def get_agent_response(user_input):
    result = await Runner.run(agent, user_input, run_config=config)
    return result.final_output

# Helper to run async safely in Streamlit
def run_asyncio_task(task):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.ensure_future(task)
    else:
        return asyncio.run(task)

# Streamlit UI
st.title("AI Chat Assistant")

# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display existing chat history
for chat in st.session_state.chat_history:
    if chat["sender"] == "user":
        st.markdown(f"**🧑‍💻 You:** {chat['message']}")
    else:
        st.markdown(f"**🤖 Assistant:** {chat['message']}")

# User input box
user_input = st.text_input("💬 Type your message:")

if st.button("✨ Send"):
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"sender": "user", "message": user_input})

        with st.spinner("Agent has Process... 💭"):
            response = run_asyncio_task(get_agent_response(user_input))
            if asyncio.isfuture(response):
                response = asyncio.run(response)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"sender": "assistant", "message": response})

        # Rerun to refresh chat display
        st.rerun()

    else:
        st.warning("Please type  😘")

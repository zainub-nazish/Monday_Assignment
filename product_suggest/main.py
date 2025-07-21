import os
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Setup model
client = AsyncOpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
config = RunConfig(model=model, model_provider=client, tracing_disabled=True)

# Define agent
agent = Agent(
    name="Smart Store Agent",
    instructions="Suggest a medicine based on user symptoms with a short reason. Format: 🤖 Suggestion: [name]\n📌 Reason: [reason]",
    model=model
)

# Required callback for Chainlit
@cl.on_message
async def handle_message(message: cl.Message):
    result = await Runner.run(agent, input=message.content, run_config=config)
    await cl.Message(content=result.final_output).send()
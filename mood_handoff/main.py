import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# 🔐 Load the Gemini API key from .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# 🔗 Connect to Gemini via OpenAI-compatible wrapper
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 🧠 Model: Gemini Flash 2.0
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# ⚙️ Configuration for agents
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# 🧠 Agent 1 – Mood Detector
mood_agent = Agent(
    name="Mood Detector",
    instructions=(
        "You are a mood detection assistant. Read the user's message and respond ONLY with one of these moods: "
        "happy, sad, angry, excited, stressed, or neutral. No explanation or extra text."
    ),
    model=model
)

# 💡 Agent 2 – Activity Suggester
activity_agent = Agent(
    name="Uplift Buddy",
    instructions=(
        "If someone is sad, stressed, or angry, suggest a short and gentle activity to uplift their mood. "
        "Use a warm and encouraging tone.\n\n"
        "Format:\n"
        "🧘 Suggested Activity: [one-line suggestion]\n"
        "💬 Note: [comforting message]"
    ),
    model=model
)

# 🚀 Main App Logic
def main():
    print("🌈 Welcome to the Mood Analyzer & Support Agent (Made by:Zainub)")
    print("🧠 Describe how you're feeling. (Type 'exit' to quit)\n")

    while True:
        user_input = input("🗣️ You: ").strip()

        if not user_input:
            print("⚠️ Please enter something to analyze.\n")
            continue

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Stay positive! You're never alone. See you soon!\n")
            break

        # 🔍 Step 1: Detect Mood
        mood_result = Runner.run_sync(mood_agent, input=user_input, run_config=config)
        mood = mood_result.final_output.strip().lower()

        print(f"🔍 Detected Mood: {mood}")

        # 🎯 Step 2: Suggest activity for negative moods
        if mood in ["sad", "stressed", "angry"]:
            activity_result = Runner.run_sync(activity_agent, input=user_input, run_config=config)
            print(activity_result.final_output + "\n")
        elif mood in ["happy", "excited", "neutral"]:
            print("✅ You seem to be doing well! Keep smiling and spreading joy! 🌟\n")
        else:
            print("⚠️ Couldn't detect a valid mood. Try expressing it differently.\n")

if __name__ == "__main__":
    main()

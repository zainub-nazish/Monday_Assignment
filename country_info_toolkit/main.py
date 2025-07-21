import os
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ✅ Setup Gemini Flash 2.5 model
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

run_config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# ✅ Tool Agent: Capital Finder
capital_agent = Agent(
    name="Capital Finder",
    model=model,
    instructions="You are an expert in world geography. Return only the CAPITAL of the given country. Don't explain, just return the capital clearly."
)

# ✅ Tool Agent: Language Finder
language_agent = Agent(
    name="Language Finder",
    model=model,
    instructions="You are a linguistic expert. Return only the MAIN LANGUAGE spoken in the given country. Don't explain, just return the language."
)

# ✅ Tool Agent: Population Finder
population_agent = Agent(
    name="Population Finder",
    model=model,
    instructions="You are a population statistics expert. Return only the POPULATION of the given country in a short format like '241 million'."
)

# ✅ Main Agent: Orchestrator
orchestrator_agent = Agent(
    name="Country Info Orchestrator",
    model=model,
    instructions="""
You are a smart assistant that collects country information using 3 expert agents:
1. Capital Finder
2. Language Finder
3. Population Finder

You will receive their results and combine them into a beautiful sentence like:
'The capital of Pakistan is Islamabad, the language is Urdu, and the population is 241 million.'

If any of the data is missing, reply with:
'I cannot fulfill that request. I need a valid country name to look up the information. Can you provide me with one?'
"""
)

# ✅ Run the program
def main():
    print("🌍 Welcome to Country Info Bot")
    country = input("Enter country name: ").strip().title()  # Standardize input like "pakistan" → "Pakistan"

    try:
        # Run each tool agent
        capital_result = Runner.run_sync(capital_agent, input=country, run_config=run_config).final_output.strip()
        language_result = Runner.run_sync(language_agent, input=country, run_config=run_config).final_output.strip()
        population_result = Runner.run_sync(population_agent, input=country, run_config=run_config).final_output.strip()

        # Debugging (optional)
        # print("🧪", capital_result, language_result, population_result)

        # Create final response
        summary_input = f"Capital: {capital_result}, Language: {language_result}, Population: {population_result}"

        final_result = Runner.run_sync(orchestrator_agent, input=summary_input, run_config=run_config)
        print("\n📘 Country Info Summary:\n")
        print(final_result.final_output)

    except Exception as e:
        print("❌ Error occurred:", e)

if __name__ == "__main__":
    main()
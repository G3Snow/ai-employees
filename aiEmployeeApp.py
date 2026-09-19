import os

import chainlit as cl
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

load_dotenv()

# Models are env-overridable so you can swap providers without rewriting the app.
FINANCE_MODEL = os.getenv("FINANCE_MODEL", "openai/gpt-4o")
OPS_MODEL = os.getenv("OPS_MODEL", "anthropic/claude-3-5-sonnet-20240620")

finance_llm = LLM(model=FINANCE_MODEL)
ops_llm = LLM(model=OPS_MODEL)

finance_agent = Agent(
    role="Finance Specialist",
    goal="Analyze budgets, ROI, and financial risk, then recommend cost-effective options.",
    backstory=(
        "You are a meticulous CFO with 20 years of experience. "
        "You quantify tradeoffs, flag hidden costs, and keep recommendations grounded in numbers."
    ),
    llm=finance_llm,
    allow_delegation=False,
    verbose=True,
)

ops_agent = Agent(
    role="Operations Specialist",
    goal="Design efficient internal processes that can actually be run day to day.",
    backstory=(
        "You are an operations lead who automates workflows. "
        "You turn strategy into steps, owners, tools, and a realistic rollout."
    ),
    llm=ops_llm,
    allow_delegation=False,
    verbose=True,
)


def missing_provider_keys() -> list[str]:
    missing = []
    if "openai/" in FINANCE_MODEL or "openai/" in OPS_MODEL:
        if not os.getenv("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
    if "anthropic/" in FINANCE_MODEL or "anthropic/" in OPS_MODEL:
        if not os.getenv("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")
    return missing


def build_crew(user_request: str) -> Crew:
    finance_task = Task(
        description=(
            "Review this workplace request from a finance perspective. "
            "Cover budget impact, cost savings, ROI, and financial risks.\n\n"
            f"User request:\n{user_request}"
        ),
        expected_output=(
            "A concise finance briefing with assumptions, numbers or ranges, "
            "tradeoffs, and a clear recommendation."
        ),
        agent=finance_agent,
    )
    ops_task = Task(
        description=(
            "Using the finance briefing as context, design a practical operating plan "
            "that respects those financial constraints.\n\n"
            f"Original user request:\n{user_request}"
        ),
        expected_output=(
            "A single reply for the user that combines finance findings with an "
            "operations plan: steps, owners, tools, timeline, and risks."
        ),
        agent=ops_agent,
        context=[finance_task],
    )
    return Crew(
        agents=[finance_agent, ops_agent],
        tasks=[finance_task, ops_task],
        process=Process.sequential,
        verbose=True,
    )


@cl.on_chat_start
async def on_chat_start():
    missing = missing_provider_keys()
    if missing:
        await cl.Message(
            content=(
                "This chat UI is ready, but the server is missing API keys: "
                + ", ".join(missing)
                + ". Set them as environment variables on the host (not on your Mac) "
                "so the finance and operations agents can run 24/7."
            )
        ).send()
        return

    await cl.Message(
        content=(
            "You are talking to a two-person AI team in one chat:\n"
            f"- **Finance Specialist** (`{FINANCE_MODEL}`)\n"
            f"- **Operations Specialist** (`{OPS_MODEL}`)\n\n"
            "Send a workplace question. Finance analyzes first; Operations turns that "
            "into an execution plan."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    missing = missing_provider_keys()
    if missing:
        await cl.Message(
            content="Cannot run the crew until these keys are set: " + ", ".join(missing)
        ).send()
        return

    status = cl.Message(
        content="Finance is reviewing this, then Operations will draft the plan..."
    )
    await status.send()

    crew = build_crew(message.content)
    result = await crew.kickoff_async()

    status.content = str(result)
    await status.update()

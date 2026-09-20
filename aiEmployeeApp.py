import os

import chainlit as cl
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

load_dotenv()

# Models are env-overridable so you can swap providers without rewriting the app.
EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", "openai/gpt-4o")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "anthropic/claude-3-5-sonnet-20240620")
TECHNICAL_MODEL = os.getenv("TECHNICAL_MODEL", "openai/gpt-4o")

COLLABORATION_PROTOCOL = """
Team protocol (follow this on every turn):
- You work with two peers: Executor (project plan), Evaluator (feasibility and errors),
  and Technical (platforms, code, and realistic tech capability).
- Start by briefly saying who is best positioned to lead this specific request and why.
  If a prior teammate already assigned roles and you agree, say so in one sentence and move on.
  Only disagree if you have a concrete, role-specific reason.
- Fact-check the others using your specialty. Correct mistakes; do not rubber-stamp.
- Be willing to discuss when the user asks for debate or when a real conflict would
  change the plan. Do not invent extra rounds, recaps, or Socratic Q&A for a simple ask.
- Stay reasonable as an individual: concise, practical, no padding.
- After agreement, do your job in pipeline order: Executor plan -> Evaluator review
  -> Technical specification. Do not skip your role or take over someone else's.
"""

executor_llm = LLM(model=EXECUTOR_MODEL)
evaluator_llm = LLM(model=EVALUATOR_MODEL)
technical_llm = LLM(model=TECHNICAL_MODEL)

executor_agent = Agent(
    role="Executor",
    goal=(
        "Turn the user's request into a quick, executable project plan, "
        "then hand it off for feasibility and technical review."
    ),
    backstory=(
        "You are a pragmatic project executor. You take messy input and turn it into "
        "a short plan someone can actually start: objective, steps, owners, tools, "
        "and a realistic sequence. You do not over-engineer simple requests.\n"
        f"{COLLABORATION_PROTOCOL}"
    ),
    llm=executor_llm,
    allow_delegation=False,
    max_iter=3,
    verbose=True,
)

evaluator_agent = Agent(
    role="Evaluator",
    goal=(
        "Stress-test the Executor's plan for executability, errors, and whether "
        "the user can realistically do it; then return an updated plan."
    ),
    backstory=(
        "You are a skeptical evaluator. You look for missing steps, false assumptions, "
        "scope that is too big, skills the user may not have, and failure points. "
        "You say clearly if the plan is doable, what would break, and how to shrink it "
        "so a single person can execute it.\n"
        f"{COLLABORATION_PROTOCOL}"
    ),
    llm=evaluator_llm,
    allow_delegation=False,
    max_iter=3,
    verbose=True,
)

technical_agent = Agent(
    role="Technical",
    goal=(
        "Convert the Evaluator's updated plan into a technically realistic version: "
        "correct code, honest platform limits, and a plan that is highly likely to work."
    ),
    backstory=(
        "You are a hands-on technical specialist. You check whether the proposed stack "
        "can actually do the job, fix incorrect APIs/code/architecture, and replace "
        "wishful tooling with something that exists and fits the constraints. "
        "You speculate thoroughly about failure modes, then output a tightened plan "
        "you would bet on working.\n"
        f"{COLLABORATION_PROTOCOL}"
    ),
    llm=technical_llm,
    allow_delegation=False,
    max_iter=3,
    verbose=True,
)


def missing_provider_keys() -> list[str]:
    models = f"{EXECUTOR_MODEL} {EVALUATOR_MODEL} {TECHNICAL_MODEL}"
    missing = []
    if "openai/" in models and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if "anthropic/" in models and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    return missing


def build_crew(user_request: str) -> Crew:
    executor_task = Task(
        description=(
            "Intake this user request with your teammates' roles in mind. "
            "First, in 1-3 sentences, agree who should lead and whether this is a "
            "simple request (keep the plan short) or needs a fuller plan. "
            "Then build a quick but executable project plan: goal, steps, tools, "
            "order of work, and what 'done' looks like. Do not write code unless "
            "the user asked for it.\n\n"
            f"User request:\n{user_request}"
        ),
        expected_output=(
            "A short role-agreement note, then an executable project plan with "
            "objective, sequenced steps, tools, and definition of done."
        ),
        agent=executor_agent,
    )
    evaluator_task = Task(
        description=(
            "Review the Executor's plan. Start by confirming or briefly correcting "
            "who should lead this request. Then evaluate: could this actually be "
            "executed as written, what errors or gaps exist, and whether it is "
            "doable by this user working alone. Fact-check claims in your lane. "
            "If the request is simple, keep the review short. Return an updated plan "
            "that fixes the issues you found.\n\n"
            f"Original user request:\n{user_request}"
        ),
        expected_output=(
            "A feasibility verdict (doable / doable with changes / not doable as written), "
            "the main errors or risks, and an updated executable plan."
        ),
        agent=evaluator_agent,
        context=[executor_task],
    )
    technical_task = Task(
        description=(
            "Take the Evaluator's updated plan. Confirm agreement on roles in one "
            "sentence unless you must dissent. Analyze technical specifications and "
            "whether the proposed platforms can actually accomplish the work. "
            "Correct any code, APIs, architecture, or tooling that would fail. "
            "Speculate on realistic failure modes, then output a final plan that has "
            "been tightened until it is highly likely to work. Match length to the "
            "request: simple asks get a short final plan, not a design document.\n\n"
            f"Original user request:\n{user_request}"
        ),
        expected_output=(
            "A final, technically realistic plan for the user: stack and platform "
            "verdict, corrected specs or code where needed, residual risks, and "
            "sequenced steps that should work."
        ),
        agent=technical_agent,
        context=[executor_task, evaluator_task],
    )
    return Crew(
        agents=[executor_agent, evaluator_agent, technical_agent],
        tasks=[executor_task, evaluator_task, technical_task],
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
                "so the Executor, Evaluator, and Technical agents can run 24/7."
            )
        ).send()
        return

    await cl.Message(
        content=(
            "You are talking to a three-person AI team in one chat:\n"
            f"- **Executor** (`{EXECUTOR_MODEL}`) — quick executable project plan\n"
            f"- **Evaluator** (`{EVALUATOR_MODEL}`) — feasibility, errors, is it doable by you\n"
            f"- **Technical** (`{TECHNICAL_MODEL}`) — specs, platforms, corrected code, final plan\n\n"
            "They first agree who should lead, fact-check each other, and keep simple "
            "requests short. Then they work in order: Executor → Evaluator → Technical."
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
        content=(
            "Team is agreeing who should lead, then Executor → Evaluator → Technical..."
        )
    )
    await status.send()

    crew = build_crew(message.content)
    result = await crew.kickoff_async()

    status.content = str(result)
    await status.update()

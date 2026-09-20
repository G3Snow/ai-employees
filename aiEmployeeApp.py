import asyncio
import hmac
import os
from typing import Optional

from persistence import (
    configure_runtime,
    database_url,
    ensure_schema,
    uses_postgres,
)

configure_runtime()

import chainlit as cl
from chainlit.data import get_data_layer as chainlit_get_data_layer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", "openai/gpt-4o")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "anthropic/claude-3-5-sonnet-20240620")
TECHNICAL_MODEL = os.getenv("TECHNICAL_MODEL", "openai/gpt-4o")
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "180"))

COLLABORATION_PROTOCOL = """
You are in a live group chat with the user and two teammates:
Executor (project plan), Evaluator (feasibility), Technical (platforms and code).
Write one complete chat message now. Do not wait for anyone else to speak.
- Open with one short sentence on who should lead this request and why, then do your job.
- If a teammate already assigned roles and you agree, say so in one clause and continue.
- Fact-check others using your specialty. Correct mistakes; do not rubber-stamp.
- Stay concise and practical. Simple asks get a short message, not a design document.
"""

ROLE_SPECS = [
    (
        "Executor",
        "is sketching the plan…",
        "Intake this user request. Lead with one sentence on who should lead, "
        "then post an executable project plan: goal, steps, tools, order of work, "
        "and what done looks like. Do not write code unless the user asked for it.",
        {
            "role": "Executor",
            "goal": (
                "Turn the user's request into a quick, executable project plan, "
                "then hand it off for feasibility and technical review."
            ),
            "backstory": (
                "You are a pragmatic project executor. You take messy input and turn it into "
                "a short plan someone can actually start: objective, steps, owners, tools, "
                "and a realistic sequence. You do not over-engineer simple requests.\n"
                f"{COLLABORATION_PROTOCOL}"
            ),
            "model": EXECUTOR_MODEL,
        },
    ),
    (
        "Evaluator",
        "is reviewing feasibility…",
        "Read the chat so far. Confirm or correct who should lead in one clause, "
        "then evaluate whether the Executor's plan can actually be executed, what "
        "errors or gaps exist, and whether it is doable by this user working alone. "
        "Return a feasibility verdict and an updated plan.",
        {
            "role": "Evaluator",
            "goal": (
                "Stress-test the Executor's plan for executability, errors, and whether "
                "the user can realistically do it; then return an updated plan."
            ),
            "backstory": (
                "You are a skeptical evaluator. You look for missing steps, false assumptions, "
                "scope that is too big, skills the user may not have, and failure points. "
                "You say clearly if the plan is doable, what would break, and how to shrink it "
                "so a single person can execute it.\n"
                f"{COLLABORATION_PROTOCOL}"
            ),
            "model": EVALUATOR_MODEL,
        },
    ),
    (
        "Technical",
        "is checking the stack…",
        "Read the chat so far. Confirm roles in one clause unless you must dissent. "
        "Check whether the proposed platforms can do the work. Correct APIs, code, "
        "or tooling that would fail. End with a final plan the user can follow.",
        {
            "role": "Technical",
            "goal": (
                "Convert the Evaluator's updated plan into a technically realistic version: "
                "correct code, honest platform limits, and a plan that is highly likely to work."
            ),
            "backstory": (
                "You are a hands-on technical specialist. You check whether the proposed stack "
                "can actually do the job, fix incorrect APIs/code/architecture, and replace "
                "wishful tooling with something that exists and fits the constraints. "
                "You speculate thoroughly about failure modes, then output a tightened plan "
                "you would bet on working.\n"
                f"{COLLABORATION_PROTOCOL}"
            ),
            "model": TECHNICAL_MODEL,
        },
    ),
]

_TEAM = None


def use_fake_team() -> bool:
    flag = os.getenv("DEV_FAKE_TEAM", "").lower() in {"1", "true", "yes"}
    if flag:
        return True
    try:
        import crewai  # noqa: F401
    except ImportError:
        return True
    return False


def get_team():
    global _TEAM
    if _TEAM is not None:
        return _TEAM
    if use_fake_team():
        _TEAM = [(name, None, thinking, instructions) for name, thinking, instructions, _cfg in ROLE_SPECS]
        return _TEAM
    from crewai import Agent, LLM

    built = []
    for name, thinking, instructions, cfg in ROLE_SPECS:
        agent = Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=LLM(model=cfg["model"]),
            allow_delegation=False,
            max_iter=3,
            verbose=True,
        )
        built.append((name, agent, thinking, instructions))
    _TEAM = built
    return _TEAM


def missing_provider_keys() -> list[str]:
    models = f"{EXECUTOR_MODEL} {EVALUATOR_MODEL} {TECHNICAL_MODEL}"
    missing = []
    if "openai/" in models and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if "anthropic/" in models and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    return missing


def _build_task(agent, instructions: str, user_request: str, prior_chat: str):
    from crewai import Task

    history = prior_chat.strip() or "(No teammate messages yet.)"
    return Task(
        description=(
            f"{instructions}\n\n"
            f"User request:\n{user_request}\n\n"
            f"Group chat so far:\n{history}"
        ),
        expected_output="One complete group-chat message with your role's deliverable.",
        agent=agent,
    )


async def _name_thread(user_text: str) -> None:
    if cl.user_session.get("thread_named"):
        return
    layer = chainlit_get_data_layer()
    thread_id = getattr(cl.context.session, "thread_id", None)
    if layer and thread_id:
        title = " ".join(user_text.strip().split())[:80] or "New chat"
        try:
            await layer.update_thread(thread_id=thread_id, name=title)
        except Exception:
            pass
    cl.user_session.set("thread_named", True)


async def run_agent_turn(
    author: str,
    agent,
    thinking: str,
    instructions: str,
    user_request: str,
    prior_chat: str,
) -> str:
    bubble = cl.Message(author=author, content=f"_{thinking}_")
    await bubble.send()
    if use_fake_team():
        await asyncio.sleep(0.4)
        text = (
            f"I'll take a pass as **{author}**.\n\n"
            f"Request: {user_request.strip()[:400]}\n\n"
            "This is a local UI stand-in because CrewAI is not available in this environment."
        )
        bubble.content = text
        await bubble.update()
        return text
    from crewai import Crew, Process

    task = _build_task(agent, instructions, user_request, prior_chat)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    try:
        result = await asyncio.wait_for(
            cl.make_async(crew.kickoff)(),
            timeout=AGENT_TIMEOUT_SECONDS,
        )
        text = str(result).strip() or "(No message)"
        bubble.content = text
        await bubble.update()
        return text
    except asyncio.TimeoutError:
        text = (
            f"I timed out after {AGENT_TIMEOUT_SECONDS}s and did not finish this turn. "
            "Try a shorter request, or raise AGENT_TIMEOUT_SECONDS on the server."
        )
        bubble.content = text
        await bubble.update()
        return text
    except Exception as exc:
        text = f"I hit an error and could not finish this turn:\n\n```\n{type(exc).__name__}: {exc}\n```"
        bubble.content = text
        await bubble.update()
        return text


@cl.data_layer
def get_data_layer():
    connect_args = {}
    ssl_require = uses_postgres()
    return SQLAlchemyDataLayer(
        conninfo=database_url(),
        connect_args=connect_args,
        ssl_require=ssl_require,
        user_thread_limit=200,
        show_logger=False,
    )


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    expected_user = os.getenv("CHAINLIT_USERNAME", "jack")
    expected_pass = os.getenv("CHAINLIT_PASSWORD", "")
    if not username.strip():
        return None
    if expected_pass:
        user_ok = username == expected_user
        pass_ok = len(password) == len(expected_pass) and hmac.compare_digest(
            password, expected_pass
        )
        if user_ok and pass_ok:
            return cl.User(identifier=username, metadata={"role": "user"})
        return None
    return cl.User(identifier=username.strip(), metadata={"role": "user"})


@cl.on_chat_start
async def on_chat_start():
    await ensure_schema()
    cl.user_session.set("thread_named", False)
    missing = missing_provider_keys()
    intro = (
        "You're in a group chat with a three-person AI team. Each person posts "
        "their own message, in order:\n"
        f"- **Executor** (`{EXECUTOR_MODEL}`) — executable project plan\n"
        f"- **Evaluator** (`{EVALUATOR_MODEL}`) — feasibility and gaps\n"
        f"- **Technical** (`{TECHNICAL_MODEL}`) — stack, specs, final plan\n\n"
        "Past chats stay in the left sidebar. Open one to pick up where you left off."
    )
    if missing and not use_fake_team():
        intro += (
            "\n\nThis UI is ready, but the server is missing API keys: "
            + ", ".join(missing)
            + ". Set them as environment variables on the host so the team can reply."
        )
    await cl.Message(content=intro, author="Front desk").send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    cl.user_session.set("thread_named", True)


@cl.on_message
async def main(message: cl.Message):
    if not use_fake_team():
        missing = missing_provider_keys()
        if missing:
            await cl.Message(
                content="Cannot run the team until these keys are set: " + ", ".join(missing),
                author="Front desk",
            ).send()
            return

    await _name_thread(message.content)

    prior_chat = f"User: {message.content}"
    for author, agent, thinking, instructions in get_team():
        reply = await run_agent_turn(
            author=author,
            agent=agent,
            thinking=thinking,
            instructions=instructions,
            user_request=message.content,
            prior_chat=prior_chat,
        )
        prior_chat += f"\n\n{author}: {reply}"

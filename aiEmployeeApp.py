"""A group chat where three AI employees each answer in their own message."""

import asyncio
import hmac
import logging
import os
from dataclasses import dataclass
from typing import Optional

from persistence import (
    bootstrap,
    connect_args,
    database_url,
    history_is_durable,
    storage_error,
)

# Must run before Chainlit is imported: it reads the auth secret at import time.
bootstrap()

import chainlit as cl
from chainlit.data import get_data_layer as chainlit_data_layer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

logger = logging.getLogger("aiEmployees")

APP_BUILD = "group-chat-3"

EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", "openai/gpt-4o")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "anthropic/claude-3-5-sonnet-20240620")
TECHNICAL_MODEL = os.getenv("TECHNICAL_MODEL", "openai/gpt-4o")
TURN_TIMEOUT_SECONDS = int(os.getenv("TURN_TIMEOUT_SECONDS", "240"))
# Bounds the provider call itself; the turn timeout alone cannot stop one already running.
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))

TRANSCRIPT_KEY = "transcript"
MAX_TRANSCRIPT_ENTRIES = 12
MAX_ENTRY_CHARS = 1500

COLLABORATION_PROTOCOL = """
You are in a live group chat with the user and two teammates:
Executor (project plan), Evaluator (feasibility), Technical (platforms and code).
Write one complete chat message now. Never wait for anyone else to speak.
- Open with one short sentence on who should lead this request and why, then do your job.
- If a teammate already assigned roles and you agree, say so in one clause and continue.
- Fact-check the others using your specialty. Correct mistakes; do not rubber-stamp.
- Talk like a colleague in chat: concise and practical, no headers for a simple ask.
"""


@dataclass(frozen=True)
class Employee:
    name: str
    model: str
    thinking: str
    goal: str
    backstory: str
    brief: str


TEAM = (
    Employee(
        name="Executor",
        model=EXECUTOR_MODEL,
        thinking="is sketching the plan…",
        goal=(
            "Turn the user's request into a quick, executable project plan, "
            "then hand it off for feasibility and technical review."
        ),
        backstory=(
            "You are a pragmatic project executor. You take messy input and turn it into "
            "a short plan someone can actually start: objective, steps, owners, tools, "
            "and a realistic sequence. You do not over-engineer simple requests."
        ),
        brief=(
            "Lead with one sentence on who should lead this, then post an executable "
            "plan: goal, steps, tools, order of work, and what done looks like. "
            "Do not write code unless the user asked for it."
        ),
    ),
    Employee(
        name="Evaluator",
        model=EVALUATOR_MODEL,
        thinking="is pressure-testing it…",
        goal=(
            "Stress-test the Executor's plan for executability, errors, and whether "
            "the user can realistically do it; then return an updated plan."
        ),
        backstory=(
            "You are a skeptical evaluator. You look for missing steps, false assumptions, "
            "scope that is too big, skills the user may not have, and failure points. "
            "You say clearly if the plan is doable, what would break, and how to shrink it "
            "so a single person can execute it."
        ),
        brief=(
            "Confirm or correct who should lead in one clause, then say whether the "
            "Executor's plan can actually be executed, what errors or gaps exist, and "
            "whether one person can do it alone. Give a verdict and an updated plan."
        ),
    ),
    Employee(
        name="Technical",
        model=TECHNICAL_MODEL,
        thinking="is checking the stack…",
        goal=(
            "Convert the Evaluator's updated plan into a technically realistic version: "
            "correct code, honest platform limits, and a plan that is highly likely to work."
        ),
        backstory=(
            "You are a hands-on technical specialist. You check whether the proposed stack "
            "can actually do the job, fix incorrect APIs, code, or architecture, and replace "
            "wishful tooling with something that exists and fits the constraints. "
            "You think through failure modes, then tighten the plan until you would bet on it."
        ),
        brief=(
            "Confirm roles in one clause unless you must dissent. Check whether the "
            "proposed platforms can do the work, correct anything that would fail, and "
            "end with the final plan the user should follow."
        ),
    ),
)


def missing_provider_keys() -> list[str]:
    models = " ".join(employee.model for employee in TEAM)
    missing = []
    if "openai/" in models and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if "anthropic/" in models and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    return missing


def _transcript() -> list[str]:
    return cl.user_session.get(TRANSCRIPT_KEY) or []


def _remember(speaker: str, message: str) -> None:
    entry = f"{speaker}: {message.strip()[:MAX_ENTRY_CHARS]}"
    history = [*_transcript(), entry][-MAX_TRANSCRIPT_ENTRIES:]
    cl.user_session.set(TRANSCRIPT_KEY, history)


def _build_crew(employee: Employee, user_request: str, stream: bool):
    from crewai import Agent, Crew, LLM, Process, Task

    agent = Agent(
        role=employee.name,
        goal=employee.goal,
        backstory=f"{employee.backstory}\n{COLLABORATION_PROTOCOL}",
        llm=LLM(model=employee.model, stream=stream, timeout=REQUEST_TIMEOUT_SECONDS),
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )
    history = "\n\n".join(_transcript()) or "(nothing yet)"
    task = Task(
        description=(
            f"{employee.brief}\n\n"
            f"Newest message from the user:\n{user_request}\n\n"
            f"Group chat so far:\n{history}"
        ),
        expected_output="One complete group-chat message containing your role's work.",
        agent=agent,
    )
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        stream=stream,
        verbose=False,
    )


def _result_text(output) -> str:
    """CrewOutput, or the finished result of a stream; `.result` re-raises crew errors."""
    result = output
    if hasattr(type(output), "result"):
        try:
            result = output.result
        except RuntimeError:
            return ""
    raw = getattr(result, "raw", None)
    return (raw if isinstance(raw, str) else str(result)).strip()


def _strip_scaffolding(text: str) -> str:
    """Streamed tokens carry the agent's `Thought:`/`Final Answer:` framing; the result does not."""
    marker = "Final Answer:"
    if marker in text:
        text = text.rsplit(marker, 1)[1]
    return text.strip()


def _is_text_chunk(chunk) -> bool:
    kind = getattr(getattr(chunk, "chunk_type", None), "value", "text")
    return kind == "text"


def _nameplate(speaker: str) -> str:
    """Chainlit 2.x only puts the author in the avatar, so name them in the text."""
    return f"**{speaker}**\n\n"


def _bubble(author: str, content: str) -> cl.Message:
    """Chainlit parents new messages to the `on_message` run step, which is never
    saved, so a resumed thread would drop every reply as an orphan."""
    message = cl.Message(author=author, content=content)
    message.parent_id = None
    return message


async def _stream_reply(employee: Employee, user_request: str, show) -> str:
    output = await _build_crew(employee, user_request, stream=True).akickoff()
    if not hasattr(output, "__aiter__"):
        return _result_text(output)

    streamed: list[str] = []
    try:
        async for chunk in output:
            token = getattr(chunk, "content", "") or ""
            if token and _is_text_chunk(chunk):
                streamed.append(token)
                await show(token)
    finally:
        # An abandoned stream keeps its worker threads; closing it at least stops
        # this consumer from being resumed against a session that has gone away.
        await output.aclose()
    return _result_text(output) or _strip_scaffolding("".join(streamed))


async def _speak(employee: Employee, bubble: cl.Message, user_request: str) -> str:
    """Stream one employee's reply into its own chat bubble."""
    started = False

    async def show(token: str) -> None:
        nonlocal started
        if not started:
            # Drop the "is thinking" placeholder the moment real words arrive.
            started = True
            bubble.content = _nameplate(employee.name)
            await bubble.update()
        await bubble.stream_token(token)

    try:
        return await _stream_reply(employee, user_request, show)
    except (AttributeError, TypeError) as exc:
        # A CrewAI release that reshapes the streaming API should cost live typing,
        # not the answer itself.
        logger.warning("Streaming unavailable (%s); falling back to a single reply.", exc)
        output = await _build_crew(employee, user_request, stream=False).akickoff()
        return _result_text(output)


def _failure_message(employee: Employee, exc: Exception) -> str:
    name = type(exc).__name__
    detail = str(exc).strip() or name
    hints = {
        "AuthenticationError": f"The API key for `{employee.model}` was rejected.",
        "PermissionDeniedError": f"That key is not allowed to use `{employee.model}`.",
        "NotFoundError": f"`{employee.model}` does not exist for this account.",
        "RateLimitError": "The provider is rate limiting, or the account is out of credit.",
        "APIConnectionError": "The provider could not be reached from the server.",
        "APITimeoutError": "The provider stopped responding.",
        "BadRequestError": f"The provider rejected the request for `{employee.model}`.",
        "ValueError": f"`{employee.model}` returned an empty response.",
        "ImportError": f"The server is missing the package for `{employee.model}`.",
    }
    hint = hints.get(name, f"Check the key and credit for `{employee.model}`.")
    return f"I could not finish. {hint}\n\n`{name}: {detail[:400]}`"


async def take_turn(employee: Employee, user_request: str) -> tuple[str, bool]:
    bubble = _bubble(employee.name, f"{_nameplate(employee.name)}_{employee.thinking}_")
    await bubble.send()
    try:
        reply = await asyncio.wait_for(
            _speak(employee, bubble, user_request), timeout=TURN_TIMEOUT_SECONDS
        )
        ok = True
    except asyncio.TimeoutError:
        reply = (
            f"I ran out of time after {TURN_TIMEOUT_SECONDS}s and stopped mid-answer. "
            "Try a smaller request, or raise TURN_TIMEOUT_SECONDS on the server."
        )
        ok = False
    except asyncio.CancelledError:
        # The browser went away; writing to its session would raise again.
        raise
    except Exception as exc:
        logger.exception("%s turn failed", employee.name)
        reply = _failure_message(employee, exc)
        ok = False

    # Replace the streamed draft with the clean final answer.
    reply = reply.strip() or "(no reply)"
    bubble.content = _nameplate(employee.name) + reply
    await bubble.update()
    return reply, ok


async def _name_thread(user_text: str) -> None:
    """Title the conversation from its first message only; a later one must not rename it."""
    if cl.user_session.get("thread_named"):
        return
    cl.user_session.set("thread_named", True)
    title = " ".join(user_text.split())[:80] or "New chat"
    try:
        layer = chainlit_data_layer()
        thread_id = getattr(cl.context.session, "thread_id", None)
        if layer and thread_id:
            await layer.update_thread(thread_id=thread_id, name=title)
    except Exception as exc:
        logger.warning("Could not title the thread (%s).", exc)


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo=database_url(),
        connect_args=connect_args(),
        user_thread_limit=200,
        show_logger=False,
    )


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Sign-in is what gives each person their own saved chat history."""
    username = username.strip()
    if not username:
        return None
    expected_user = os.getenv("CHAINLIT_USERNAME", "").strip()
    expected_pass = os.getenv("CHAINLIT_PASSWORD", "")
    if expected_pass:
        matches = hmac.compare_digest(username, expected_user or username) and hmac.compare_digest(
            password, expected_pass
        )
        if not matches:
            return None
    return cl.User(identifier=username, metadata={"role": "user"})


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set(TRANSCRIPT_KEY, [])
    cl.user_session.set("thread_named", False)

    lines = [
        _nameplate("Front desk")
        + "You're in a group chat with three AI teammates. Each one writes its own "
        "message, live, in this order:",
        f"- **Executor** (`{EXECUTOR_MODEL}`) — the plan",
        f"- **Evaluator** (`{EVALUATOR_MODEL}`) — what breaks, and whether you can do it",
        f"- **Technical** (`{TECHNICAL_MODEL}`) — stack, corrections, final plan",
        "",
        "Your past chats are in the left sidebar. Open one to continue it.",
        f"_Build {APP_BUILD}_",
    ]
    if missing := missing_provider_keys():
        lines.append(
            "\n**Not ready yet.** The server is missing " + ", ".join(f"`{k}`" for k in missing)
            + ". Add them in the Render dashboard under Environment, then redeploy."
        )
    if failure := storage_error():
        lines.append(
            f"\n**History is not saving.** The database could not be prepared: `{failure}`"
        )
    elif not history_is_durable():
        lines.append(
            "\n**History is temporary.** Set `DATABASE_URL` to a Postgres database so "
            "these chats survive restarts."
        )
    await _bubble("Front desk", "\n".join(lines)).send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Reload an old conversation so the team keeps its memory of it."""
    cl.user_session.set("thread_named", True)
    roles = {employee.name for employee in TEAM}
    history: list[str] = []
    for step in thread.get("steps") or []:
        speaker = step.get("name") or ""
        content = (step.get("output") or "").strip()
        if speaker and content.startswith(f"**{speaker}**"):
            content = content[len(f"**{speaker}**") :].strip()
        if not content:
            continue
        if step.get("type") == "user_message":
            history.append(f"You: {content[:MAX_ENTRY_CHARS]}")
        elif speaker in roles:
            history.append(f"{speaker}: {content[:MAX_ENTRY_CHARS]}")
    cl.user_session.set(TRANSCRIPT_KEY, history[-MAX_TRANSCRIPT_ENTRIES:])


@cl.on_message
async def on_message(message: cl.Message):
    if missing := missing_provider_keys():
        await _bubble(
            "Front desk",
            _nameplate("Front desk")
            + "The team can't reply until these keys are set on the server: "
            + ", ".join(f"`{key}`" for key in missing),
        ).send()
        return

    await _name_thread(message.content)
    _remember("You", message.content)

    failed: list[str] = []
    for employee in TEAM:
        reply, ok = await take_turn(employee, message.content)
        if ok:
            _remember(employee.name, reply)
        else:
            # The rest of the team still runs, but it should not treat an error as advice.
            failed.append(employee.name)
            _remember(
                "Chat moderator",
                f"{employee.name} could not reply this turn. Cover that ground yourself "
                "and do not refer to its answer.",
            )

    if failed:
        await _bubble(
            "Front desk",
            _nameplate("Front desk")
            + f"{', '.join(failed)} could not reply this turn, so the plan above is "
            "missing their input. Fix the error shown in their message and send again.",
        ).send()

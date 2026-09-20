"""A group chat where three AI employees each answer in their own message."""

import asyncio
import hmac
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from persistence import (
    bootstrap,
    connect_args,
    database_url,
    history_is_durable,
    storage_error,
)
from files import (
    FileWorkspace,
    IncomingFiles,
    chainlit_elements,
    employee_tools,
    ingest,
)

# Must run before Chainlit is imported: it reads the auth secret at import time.
bootstrap()

import chainlit as cl
from chainlit.data import get_data_layer as chainlit_data_layer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

logger = logging.getLogger("aiEmployees")

APP_BUILD = "group-chat-6"

EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", "openai/gpt-6-astra")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "anthropic/claude-fable-5-1")
TECHNICAL_MODEL = os.getenv("TECHNICAL_MODEL", "anthropic/claude-opus-5")
TURN_TIMEOUT_SECONDS = int(os.getenv("TURN_TIMEOUT_SECONDS", "360"))
# Bounds the provider call itself; the turn timeout alone cannot stop one already running.
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "240"))
MAX_AGREEMENT_ROUNDS = max(1, int(os.getenv("MAX_AGREEMENT_ROUNDS", "2")))

TRANSCRIPT_KEY = "transcript"
MAX_TRANSCRIPT_ENTRIES = 50
MAX_ENTRY_CHARS = 4000

_AGREEMENT_LINE = re.compile(r"(?im)^\s*AGREEMENT:\s*(yes|no)\b")

COLLABORATION_PROTOCOL = """
You are in a live group chat with the user and two teammates:
Executor (researched plan), Evaluator (project manager and fact-checker),
Technical (deployment plan and implementation).
Write one complete chat message now. Never wait for anyone else to speak.
- Open with one short sentence on who should lead this request and why, then do your job.
- If a teammate already assigned roles and you agree, say so in one clause and continue.
- Fact-check the others using your specialty. Correct mistakes; do not rubber-stamp.
- Do not trust a teammate's work or citations until you have checked them yourself.
- Talk like a colleague in chat: concise and practical, no headers for a simple ask.
- The user may attach images, spreadsheets, and other files. Use them.
- If a downloadable file would help (spreadsheet, csv, code, notes), create it with
  write_text_file or write_spreadsheet. It is attached to your message automatically.
- For facts about products, APIs, vendors, or procedures: use search_web and fetch_url.
  Prefer official company, OEM, and vendor documentation. Reddit is allowed only when
  later replies in the same thread confirm the approach actually worked.
"""

AGREEMENT_FOOTER = """
End your message with exactly one of these lines, on its own:
AGREEMENT: no
AGREEMENT: yes
Use yes only if you independently verified the latest revised plan and accept it
as correct, viable, and still serving the user's larger goal. Never rubber-stamp.
"""

EXECUTOR_DRAFT = """
This is your first pass. Research before you plan. Use search_web and fetch_url
whenever the request depends on external products, APIs, vendors, or procedures.
Cite URLs. Prefer official company and OEM pages. Reddit only if replies in that
thread confirm success. Do not mark AGREEMENT: yes yet.
""" + AGREEMENT_FOOTER

EVALUATOR_FIRST = """
This is your first review. Do not mark AGREEMENT: yes yet.
Primary job: read EVERY message in this thread. Catch hallucinations, dropped
constraints, and drift away from the user's larger project. You are the project
manager. Translate the Executor's draft into a plan that still serves that goal.
Assume the Executor made mistakes. Independently fact-check their claims with
official sources via search_web and fetch_url. Do not trust their citations
without fetching them. Keep the feasibility stress-test: what breaks, and whether
the user can actually do it.
""" + AGREEMENT_FOOTER

EXECUTOR_REBUTTAL = """
Evaluator challenged your plan. Do not trust that critique by default.
Independently verify their claims with official sources. Concede only what you
confirm is wrong; argue for what still holds. Post the revised plan with citations.
""" + AGREEMENT_FOOTER

EVALUATOR_REBUTTAL = """
Continue as project manager. Re-read the whole thread. Fact-check Executor again
with official sources. Argue remaining issues. You two must both independently
agree before Technical is allowed to see this. Keep the plan inside the user's
larger goal.
""" + AGREEMENT_FOOTER

TECHNICAL_AFTER_AGREEMENT = """
Executor and Evaluator have reached firm agreement. Use only that agreed revised
plan. Translate it into a technical deployment plan the way a highly skilled
engineer would: proven production practices, current official docs, nothing
experimental (no beta APIs, unproven libraries, or clever unpublished tricks).
Check platforms, correct anything that would fail, and think through failure modes.
If the user asked for action (update code, edit a document, produce a file),
carry it out now with write_text_file or write_spreadsheet. Ship the actual
artifact, not a description of the change.
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
        thinking="is researching the plan…",
        goal=(
            "Turn the user's request into a researched, executable project plan "
            "grounded in official sources, then defend it until Evaluator independently agrees."
        ),
        backstory=(
            "You are a pragmatic project executor. You take messy input and turn it into "
            "a plan someone can actually start: objective, steps, owners, tools, and a "
            "realistic sequence. You build the big plan from current public knowledge: "
            "official company and OEM websites first. You may use Reddit only when later "
            "replies in the same thread confirm the approach worked. You do not "
            "over-engineer simple requests, and you never invent citations."
        ),
        brief=(
            "Lead with one sentence on who should lead this, then post an executable "
            "plan: goal, steps, tools, order of work, and what done looks like. "
            "Cite the official/OEM URLs you used. Do not write code unless the user "
            "asked for it."
        ),
    ),
    Employee(
        name="Evaluator",
        model=EVALUATOR_MODEL,
        thinking="is auditing the thread…",
        goal=(
            "Keep the whole thread honest and on-project: catch hallucinations and drift, "
            "fact-check the Executor (assuming mistakes), and translate their draft into "
            "a plan that still serves the user's larger goal."
        ),
        backstory=(
            "You are a skeptical project manager, not a rubber stamp. Your primary job is "
            "to read the entire chat, notice when the team is hallucinating or losing the "
            "plot, and pull the work back to the user's actual larger goal. You assume the "
            "Executor got things wrong. You independently verify claims with official "
            "company, OEM, and vendor documentation. You also stress-test executability: "
            "missing steps, false assumptions, scope that is too big, skills the user may "
            "not have, and failure points. You say clearly if the plan is doable, what "
            "would break, and how to shrink it so a single person can execute it."
        ),
        brief=(
            "Confirm or correct who should lead in one clause. Audit the whole thread "
            "for hallucinations and project drift. Fact-check the Executor with official "
            "sources. Say whether the plan can actually be executed, what errors or gaps "
            "exist, and whether one person can do it alone. Give a verdict and a revised "
            "plan that stays inside the user's larger goal."
        ),
    ),
    Employee(
        name="Technical",
        model=TECHNICAL_MODEL,
        thinking="is writing the deployment plan…",
        goal=(
            "Turn the agreed Executor/Evaluator plan into a technically realistic "
            "deployment plan, then carry it out if the user asked for action."
        ),
        backstory=(
            "You are a highly skilled hands-on engineer. You review the plan passed to "
            "you only after Executor and Evaluator have both agreed it is correct. You "
            "check whether the proposed stack can actually do the job, fix incorrect APIs, "
            "code, or architecture, and replace wishful tooling with something that exists "
            "and is production-proven. You refuse experimental or unproven approaches. "
            "You think through failure modes, then tighten the plan until you would bet "
            "on it. When the user asked for a change, you make the change."
        ),
        brief=(
            "Confirm roles in one clause unless you must dissent. Translate the agreed "
            "plan into a technical deployment plan using only proven practices. Check "
            "whether the proposed platforms can do the work, correct anything that would "
            "fail, and end with the final plan the user should follow. If they asked you "
            "to update code or a document, produce the actual files now."
        ),
    ),
)


def _same_secret(left: str, right: str) -> bool:
    a, b = left.encode("utf-8"), right.encode("utf-8")
    if len(a) != len(b):
        hmac.compare_digest(a, a)
        return False
    return hmac.compare_digest(a, b)


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


def _build_crew(
    employee: Employee,
    user_request: str,
    stream: bool,
    incoming: IncomingFiles,
    workspace: FileWorkspace,
    extra_brief: str = "",
):
    from crewai import Agent, Crew, LLM, Process, Task

    tools = employee_tools(workspace)
    vision = incoming.vision_files(employee.model)
    llm_kwargs = dict(
        model=employee.model, stream=stream, timeout=REQUEST_TIMEOUT_SECONDS
    )
    if employee.model.startswith("anthropic/"):
        llm_kwargs["max_tokens"] = 8192
    agent = Agent(
        role=employee.name,
        goal=employee.goal,
        backstory=f"{employee.backstory}\n{COLLABORATION_PROTOCOL}",
        llm=LLM(**llm_kwargs),
        allow_delegation=False,
        max_iter=8,
        tools=tools,
        verbose=False,
    )
    history = "\n\n".join(_transcript()) or "(nothing yet)"
    files_block = f"\n\n{incoming.prompt_block}" if incoming.prompt_block else ""
    if incoming.input_files and not vision:
        files_block += (
            "\n(This model cannot view the image pixels; use the description above.)"
        )
    stage = f"\n\n{extra_brief.strip()}" if extra_brief.strip() else ""
    task = Task(
        description=(
            f"{employee.brief}\n"
            f"{stage}\n\n"
            f"Newest message from the user:\n{user_request}\n\n"
            f"Group chat so far (full project record — audit it):\n{history}"
            f"{files_block}"
        ),
        expected_output="One complete group-chat message containing your role's work.",
        agent=agent,
        tools=tools,
        input_files=vision,
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


async def _stream_reply(
    employee: Employee,
    user_request: str,
    show,
    incoming: IncomingFiles,
    workspace: FileWorkspace,
    extra_brief: str = "",
) -> str:
    vision = incoming.vision_files(employee.model) or None
    crew = _build_crew(employee, user_request, True, incoming, workspace, extra_brief)
    output = await crew.akickoff(input_files=vision)
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
        await output.aclose()
    return _result_text(output) or _strip_scaffolding("".join(streamed))


async def _speak(
    employee: Employee,
    bubble: cl.Message,
    user_request: str,
    incoming: IncomingFiles,
    workspace: FileWorkspace,
    extra_brief: str = "",
) -> str:
    """Stream one employee's reply into its own chat bubble."""
    started = False

    async def show(token: str) -> None:
        nonlocal started
        if not started:
            started = True
            bubble.content = _nameplate(employee.name)
            await bubble.update()
        await bubble.stream_token(token)

    try:
        return await _stream_reply(
            employee, user_request, show, incoming, workspace, extra_brief
        )
    except (AttributeError, TypeError) as exc:
        logger.warning("Streaming unavailable (%s); falling back to a single reply.", exc)
        vision = incoming.vision_files(employee.model) or None
        output = await _build_crew(
            employee, user_request, False, incoming, workspace, extra_brief
        ).akickoff(input_files=vision)
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


async def take_turn(
    employee: Employee,
    user_request: str,
    incoming: IncomingFiles,
    extra_brief: str = "",
) -> tuple[str, bool]:
    bubble = _bubble(employee.name, f"{_nameplate(employee.name)}_{employee.thinking}_")
    await bubble.send()
    workspace = FileWorkspace()
    try:
        try:
            reply = await asyncio.wait_for(
                _speak(employee, bubble, user_request, incoming, workspace, extra_brief),
                timeout=TURN_TIMEOUT_SECONDS,
            )
            ok = True
        except asyncio.TimeoutError:
            reply = (
                f"I ran out of time after {TURN_TIMEOUT_SECONDS}s and stopped mid-answer. "
                "Try a smaller request, or raise TURN_TIMEOUT_SECONDS on the server."
            )
            ok = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("%s turn failed", employee.name)
            reply = _failure_message(employee, exc)
            ok = False

        reply = reply.strip() or "(no reply)"
        attachments = chainlit_elements(workspace.created)
        if attachments:
            names = ", ".join(path.name for path in workspace.created)
            reply = f"{reply}\n\nAttached: {names}"
            bubble.elements = attachments
        bubble.content = _nameplate(employee.name) + reply
        await bubble.update()
        return reply, ok
    finally:
        workspace.close()


def _named(name: str) -> Employee:
    for employee in TEAM:
        if employee.name == name:
            return employee
    raise KeyError(name)


def _agrees(text: str) -> bool:
    matches = _AGREEMENT_LINE.findall(text or "")
    return bool(matches) and matches[-1].lower() == "yes"


def _both_agree(executor_text: str, evaluator_text: str) -> bool:
    return _agrees(executor_text) and _agrees(evaluator_text)


async def _run_turn(
    employee: Employee,
    request: str,
    incoming: IncomingFiles,
    failed: list[str],
    extra_brief: str = "",
) -> tuple[str, bool]:
    reply, ok = await take_turn(employee, request, incoming, extra_brief)
    if ok:
        _remember(employee.name, reply)
    else:
        failed.append(employee.name)
        _remember(
            "Chat moderator",
            f"{employee.name} could not reply this turn. Cover that ground yourself "
            "and do not refer to its answer.",
        )
    return reply, ok


async def _collaborate(
    request: str, incoming: IncomingFiles
) -> tuple[list[str], bool]:
    """Executor drafts, Evaluator reviews, they debate, Technical only after both agree."""
    failed: list[str] = []
    agreed = False
    executor = _named("Executor")
    evaluator = _named("Evaluator")
    technical = _named("Technical")

    _, executor_ok = await _run_turn(
        executor, request, incoming, failed, EXECUTOR_DRAFT
    )
    if executor_ok:
        _, evaluator_ok = await _run_turn(
            evaluator, request, incoming, failed, EVALUATOR_FIRST
        )
        if evaluator_ok:
            last_round = MAX_AGREEMENT_ROUNDS
            for round_index in range(1, last_round + 1):
                last_hint = ""
                if round_index == last_round:
                    last_hint = (
                        "\nThis is the last debate round. Do not fake consensus. "
                        "If you cannot independently agree, keep AGREEMENT: no.\n"
                    )
                executor_reply, executor_ok = await _run_turn(
                    executor,
                    request,
                    incoming,
                    failed,
                    EXECUTOR_REBUTTAL + last_hint,
                )
                if not executor_ok:
                    break
                evaluator_reply, evaluator_ok = await _run_turn(
                    evaluator,
                    request,
                    incoming,
                    failed,
                    EVALUATOR_REBUTTAL + last_hint,
                )
                if not evaluator_ok:
                    break
                if _both_agree(executor_reply, evaluator_reply):
                    agreed = True
                    break

    if not failed and agreed:
        await _run_turn(
            technical, request, incoming, failed, TECHNICAL_AFTER_AGREEMENT
        )
    return failed, agreed


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
        matches = _same_secret(username, expected_user or username) and _same_secret(
            password, expected_pass
        )
        if not matches:
            return None
    return cl.User(identifier=username, metadata={"role": "user"})


def _welcome_lines() -> list[str]:
    lines = [
        _nameplate("Front desk")
        + "You're in a group chat with three AI teammates. Each one writes its own "
        "message, live:",
        f"- **Executor** (`{EXECUTOR_MODEL}`) — researches the plan from official company "
        "and OEM sources (Reddit only when replies confirmed success)",
        f"- **Evaluator** (`{EVALUATOR_MODEL}`) — project manager: audits the whole thread, "
        "fact-checks hard, and keeps the work on your larger goal",
        "- They argue and fact-check each other until both independently agree. Only then:",
        f"- **Technical** (`{TECHNICAL_MODEL}`) — technical deployment plan, and does the "
        "work if you asked for action",
        "",
        "Attach images or spreadsheets with the paperclip. The team can send files back too.",
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
    return lines


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set(TRANSCRIPT_KEY, [])
    cl.user_session.set("thread_named", False)
    await _bubble("Front desk", "\n".join(_welcome_lines())).send()


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
        elif speaker in roles or speaker in {"Front desk", "Chat moderator"}:
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

    request = (message.content or "").strip() or "Please review the attached files."
    stash = FileWorkspace()
    try:
        incoming = ingest(list(message.elements or []), stash.uploads)
        title_source = (message.content or "").strip() or incoming.summary or request
        await _name_thread(title_source)
        _remember("You", request + (f"\n[{incoming.summary}]" if incoming.summary else ""))

        failed, agreed = await _collaborate(request, incoming)

        if failed:
            await _bubble(
                "Front desk",
                _nameplate("Front desk")
                + f"{', '.join(failed)} could not reply this turn, so the plan above is "
                "missing their input. Fix the error shown in their message and send again.",
            ).send()
        elif not agreed:
            note = (
                "Executor and Evaluator did not both independently agree, so Technical "
                "did not start. Send a follow-up if you want them to keep arguing the plan."
            )
            _remember("Chat moderator", note)
            await _bubble("Front desk", _nameplate("Front desk") + note).send()
    finally:
        stash.close()

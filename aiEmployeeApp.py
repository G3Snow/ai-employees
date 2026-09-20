"""A group chat where three AI employees each answer in their own message."""

import asyncio
import hmac
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Chainlit inserts this directory on sys.path during load, then pops it.
_APP_DIR = str(Path(__file__).resolve().parent)
if sys.path[-1:] != [_APP_DIR]:
    sys.path.append(_APP_DIR)

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

APP_BUILD = "group-chat-10"

EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", "openai/gpt-6-astra")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "anthropic/claude-fable-5-1")
TECHNICAL_MODEL = os.getenv("TECHNICAL_MODEL", "anthropic/claude-opus-5")
TURN_TIMEOUT_SECONDS = int(os.getenv("TURN_TIMEOUT_SECONDS", "900"))
# Bounds the provider call itself; the turn timeout alone cannot stop one already running.
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "840"))
MAX_AGREEMENT_ROUNDS = max(1, int(os.getenv("MAX_AGREEMENT_ROUNDS", "3")))
# GPT-6 Astra rejects reasoning_effort="none". Chat Completions also cannot
# combine its function tools with any other effort, so those models use the
# Responses API instead. low | medium | high | xhigh | max
_OPENAI_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_DEFAULT_OPENAI_REASONING_EFFORT = "high"

TRANSCRIPT_KEY = "transcript"
PENDING_KEY = "pending_agreement"
CHECKIN_MSG_KEY = "agreement_checkin_msg"
BUSY_KEY = "team_busy"
STALLED_KEY = "agreement_stalled"
MAX_TRANSCRIPT_ENTRIES = 50
MAX_ENTRY_CHARS = 4000

_AGREEMENT_LINE = re.compile(r"(?im)^\s*AGREEMENT:\s*(yes|no)\b")
EMPLOYEE_NAMES = ("Executor", "Evaluator", "Technical")
_NAME_TOKEN = r"Executor|Evaluator|Technical"
_OPENING_ADDRESS = re.compile(
    rf"^\s*(?:(?:hey|hi|hello|ok|okay|please)\s+)?"
    rf"((?:@?(?:{_NAME_TOKEN}))(?:\s*(?:,|\band\b|&|/)\s*@?(?:{_NAME_TOKEN}))*)"
    rf"\s*[,:]\s*",
    re.I,
)
_VOCATIVE = re.compile(
    rf"(?:^|(?<=\n)|(?<=[.!?]\s))"
    rf"(?:@({_NAME_TOKEN})(?:\s*[,:]|\s+)|({_NAME_TOKEN})\s*[,:]\s+)",
    re.I,
)
_AT_NAME = re.compile(rf"@({_NAME_TOKEN})\b", re.I)
_CHECKIN_MARKER = "have not independently agreed after"

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
- The user may address you by name (`Executor, …` or `@Evaluator`). Follow that
  instruction; it outranks a teammate's last request, but not the user's original goal.
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

CONTINUE_HINT = """
The user asked you to keep working toward independent agreement. Do not fake
consensus. Continue the debate and fact-check with official sources.
"""

COMPROMISE_EXECUTOR = """
The user asked you to lower the bar a little so a plan can ship. Keep what you
and Evaluator already share. Drop remaining disagreements that are not unsafe,
factually wrong, or against the user's goal. Post that most-agreed plan.
""" + AGREEMENT_FOOTER

COMPROMISE_EVALUATOR = """
The user asked you to lower the bar a little. Do not rubber-stamp something
false or off-goal. Do drop remaining non-deal-breaker objections so the version
you already share the most can proceed. AGREEMENT: yes only if you can stand
behind that compromise.
""" + AGREEMENT_FOOTER

TECHNICAL_HANDOFF = """
Executor and Evaluator did not fully agree. The user told Front desk to send you
what they have anyway. Use the latest plans in the thread. Prefer the overlap
they already share. Treat remaining disagreements as risks to call out, then
write a technically realistic deployment plan from proven practices. If the user
asked for action, carry it out.
"""

TECHNICAL_DIRECTED = """
The user addressed you by name. Follow their instruction. Use the current thread.
If Executor and Evaluator never fully agreed, prefer their overlap and flag
leftovers as risks. Still use proven practices only. If they asked for action,
carry it out.
"""

CHOICE_LABELS = {
    "continue": "Keep arguing",
    "compromise": "Lower the bar a little",
    "technical": "Send what they have to Technical",
}


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


def _model_leaf(model: str) -> str:
    return model.rsplit("/", 1)[-1].lower()


def _needs_openai_responses(model: str) -> bool:
    """GPT-6 Astra can call tools only on the Responses API.

    CrewAI defaults to Chat Completions. That path gets a 400 for tools +
    reasoning, then retries with reasoning_effort="none", which Astra rejects.
    """
    name = _model_leaf(model)
    return name.startswith("gpt-6") or "astra" in name


def _stream_supported(model: str) -> bool:
    """CrewAI 1.15.x Responses streaming returns '' when the model calls a tool.

    Chat Completions streaming already hands those tool calls back to the agent.
    The Responses stream path does not, so CrewAI raises
    ``ValueError: Invalid response from LLM call - None or empty.``
    Astra can only use tools on Responses, so it cannot stream until that is fixed.
    """
    return not _needs_openai_responses(model)


def _openai_reasoning_effort() -> str:
    raw = os.getenv("OPENAI_REASONING_EFFORT", _DEFAULT_OPENAI_REASONING_EFFORT)
    effort = (raw or "").strip().lower()
    if effort in _OPENAI_REASONING_EFFORTS:
        return effort
    logger.warning(
        "OPENAI_REASONING_EFFORT=%r is not supported; using %s.",
        raw,
        _DEFAULT_OPENAI_REASONING_EFFORT,
    )
    return _DEFAULT_OPENAI_REASONING_EFFORT


def _llm_kwargs(employee: Employee, stream: bool) -> dict:
    if not _stream_supported(employee.model):
        stream = False
    kwargs: dict = dict(
        model=employee.model, stream=stream, timeout=REQUEST_TIMEOUT_SECONDS
    )
    if employee.model.startswith("anthropic/"):
        kwargs["max_tokens"] = 8192
    if _needs_openai_responses(employee.model):
        kwargs["api"] = "responses"
        kwargs["reasoning_effort"] = _openai_reasoning_effort()
    return kwargs


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
    stream = stream and _stream_supported(employee.model)
    llm_kwargs = _llm_kwargs(employee, stream)
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


def _bubble(author: str, content: str, actions=None) -> cl.Message:
    """Chainlit parents new messages to the `on_message` run step, which is never
    saved, so a resumed thread would drop every reply as an orphan."""
    message = cl.Message(author=author, content=content, actions=actions or [])
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
    crew = _build_crew(
        employee,
        user_request,
        _stream_supported(employee.model),
        incoming,
        workspace,
        extra_brief,
    )
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

    streamed = _stream_supported(employee.model)
    try:
        return await _stream_reply(
            employee, user_request, show, incoming, workspace, extra_brief
        )
    except (AttributeError, TypeError, ValueError) as exc:
        empty = isinstance(exc, ValueError) and "None or empty" in str(exc)
        if isinstance(exc, ValueError) and not empty:
            raise
        if not streamed:
            raise
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
        "ImportError": "The server is missing a Python package.",
        "ModuleNotFoundError": "The server is missing a Python module.",
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


def _canon_name(raw: str) -> Optional[str]:
    for name in EMPLOYEE_NAMES:
        if name.lower() == (raw or "").strip().lower():
            return name
    return None


def _names_in(blob: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(_NAME_TOKEN, blob or "", re.I):
        name = _canon_name(match.group(0))
        if name and name not in found:
            found.append(name)
    return found


def _vocative_name(match: re.Match) -> Optional[str]:
    return _canon_name(match.group(1) or match.group(2) or "")


def _apply_vocatives(notes: dict[str, str], text: str) -> None:
    matches = list(_VOCATIVE.finditer(text or ""))
    for index, match in enumerate(matches):
        name = _vocative_name(match)
        if not name:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        notes[name] = text[match.end() : end].strip()


def parse_named_instructions(text: str) -> dict[str, str]:
    """Map teammate name → instruction when the user addresses them vocatively."""
    text = (text or "").strip()
    notes: dict[str, str] = {}
    if not text:
        return notes

    opening = _OPENING_ADDRESS.match(text)
    body = text
    if opening:
        body = text[opening.end() :]
        vocatives = list(_VOCATIVE.finditer(body))
        shared = body[: vocatives[0].start()].strip() if vocatives else body.strip()
        for name in _names_in(opening.group(1)):
            notes[name] = shared
        _apply_vocatives(notes, body)
    else:
        _apply_vocatives(notes, text)

    for match in _AT_NAME.finditer(text):
        name = _canon_name(match.group(1))
        if name and name not in notes:
            notes[name] = text.strip()

    return {name: " ".join(instruction.split()).strip(" ,") for name, instruction in notes.items()}


def parse_agreement_choice(text: str) -> Optional[str]:
    """Return continue / compromise / technical from a free-text reply, if any."""
    stripped = (text or "").strip()
    lowered = stripped.lower()
    for key, label in CHOICE_LABELS.items():
        if lowered == label.lower():
            return key
    if re.search(
        r"\b((send|give|pass|hand)\b.{0,40}\btechnical\b|take what (they|you|we) have|"
        r"give it to technical|just ship|as[- ]is to technical)\b",
        lowered,
        re.S,
    ):
        return "technical"
    if re.search(
        r"\b(compromise|lower the bar|reduce (the )?standards|relax (the )?(standards|bar)|"
        r"most agreed|good enough)\b",
        lowered,
    ):
        return "compromise"
    if re.search(
        r"\b(keep (going|arguing)|continue|another (round|try)|try again|keep working)\b",
        lowered,
    ):
        return "continue"
    return None


def _user_note_brief(name: str, instruction: str) -> str:
    if not instruction:
        return f"The user addressed you ({name}) by name this turn.\n"
    return (
        f"The user addressed you ({name}) by name this turn. Follow this instruction; "
        f"it outranks your teammate's last request, but not the user's original goal:\n"
        f"{instruction}\n"
    )


def _directed_brief(employee: Employee, instruction: str, wants_agreement: bool) -> str:
    note = _user_note_brief(employee.name, instruction)
    if employee.name == "Technical":
        return note + TECHNICAL_DIRECTED
    if not wants_agreement:
        return (
            note
            + "The rest of the team is silent this turn unless also named. Reply now. "
            "Skip the AGREEMENT line unless you are actually accepting a plan."
        )
    extra = EXECUTOR_REBUTTAL if employee.name == "Executor" else EVALUATOR_REBUTTAL
    return note + extra


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


async def _agreement_rounds(
    request: str,
    incoming: IncomingFiles,
    failed: list[str],
    *,
    rounds: int,
    executor_extra: str,
    evaluator_extra: str,
    lead: Optional[str] = None,
    last_round_hint: str,
) -> bool:
    executor = _named("Executor")
    evaluator = _named("Evaluator")
    order = [evaluator, executor] if lead == "Evaluator" else [executor, evaluator]
    replies = {"Executor": "", "Evaluator": ""}
    for round_index in range(1, rounds + 1):
        hint = last_round_hint if round_index == rounds else ""
        for employee in order:
            extra = executor_extra if employee.name == "Executor" else evaluator_extra
            reply, ok = await _run_turn(
                employee, request, incoming, failed, extra + hint
            )
            if not ok:
                return False
            replies[employee.name] = reply
        if _both_agree(replies["Executor"], replies["Evaluator"]):
            return True
    return False


async def _send_to_technical(
    request: str,
    incoming: IncomingFiles,
    failed: list[str],
    extra_brief: str,
) -> None:
    await _run_turn(_named("Technical"), request, incoming, failed, extra_brief)


async def _collaborate(
    request: str, incoming: IncomingFiles
) -> tuple[list[str], bool]:
    """Executor drafts, Evaluator reviews, they debate, Technical only after both agree."""
    failed: list[str] = []
    executor = _named("Executor")
    evaluator = _named("Evaluator")
    agreed = False

    _, executor_ok = await _run_turn(
        executor, request, incoming, failed, EXECUTOR_DRAFT
    )
    if executor_ok:
        _, evaluator_ok = await _run_turn(
            evaluator, request, incoming, failed, EVALUATOR_FIRST
        )
        if evaluator_ok:
            agreed = await _agreement_rounds(
                request,
                incoming,
                failed,
                rounds=MAX_AGREEMENT_ROUNDS,
                executor_extra=EXECUTOR_REBUTTAL,
                evaluator_extra=EVALUATOR_REBUTTAL,
                last_round_hint=(
                    "\nThis is the last debate round. Do not fake consensus. "
                    "If you cannot independently agree, keep AGREEMENT: no.\n"
                ),
            )

    if not failed and agreed:
        await _send_to_technical(
            request, incoming, failed, TECHNICAL_AFTER_AGREEMENT
        )
    return failed, agreed


async def _directed_turns(
    request: str, incoming: IncomingFiles, notes: dict[str, str]
) -> tuple[list[str], bool]:
    """Only the teammates the user named speak this turn."""
    failed: list[str] = []
    named = {name for name in notes if name in EMPLOYEE_NAMES}
    wants_agreement = "Executor" in named and "Evaluator" in named
    replies = {"Executor": "", "Evaluator": ""}
    for employee in TEAM:
        if employee.name not in named:
            continue
        reply, ok = await _run_turn(
            employee,
            request,
            incoming,
            failed,
            _directed_brief(employee, notes[employee.name], wants_agreement),
        )
        if not ok:
            continue
        if employee.name in replies:
            replies[employee.name] = reply

    agreed = wants_agreement and not failed and _both_agree(
        replies["Executor"], replies["Evaluator"]
    )
    if agreed and "Technical" not in named:
        await _send_to_technical(
            request, incoming, failed, TECHNICAL_AFTER_AGREEMENT
        )
    return failed, agreed


def _checkin_text() -> str:
    n = MAX_AGREEMENT_ROUNDS
    return (
        f"{_nameplate('Front desk')}"
        f"Executor and Evaluator {_CHECKIN_MARKER} {n} attempts, so I have not "
        "sent this to Technical.\n\n"
        "How do you want to proceed?\n"
        "- **Keep arguing** — they continue until they can both stand behind a plan.\n"
        "- **Lower the bar a little** — they converge on the version they already "
        "share the most, dropping remaining non-deal-breaker disagreements.\n"
        "- **Send to Technical** — Technical takes the current plan as-is and treats "
        "leftover disagreements as risks.\n\n"
        "Click a button, or reply in chat. You can also instruct someone by name, "
        "for example:\n"
        "- `Evaluator, drop the multi-region requirement`\n"
        "- `Executor, keep SQLite. Evaluator, that is acceptable.`\n"
        "- `@Technical implement the overlap they already have`"
    )


def _checkin_actions() -> list:
    return [
        cl.Action(
            name="agreement_choice",
            payload={"choice": "continue"},
            label="Keep arguing",
            icon="messages-square",
            tooltip="They keep debating toward independent agreement.",
        ),
        cl.Action(
            name="agreement_choice",
            payload={"choice": "compromise"},
            label="Lower the bar a little",
            icon="scale",
            tooltip="Ship the most agreed-upon version.",
        ),
        cl.Action(
            name="agreement_choice",
            payload={"choice": "technical"},
            label="Send to Technical",
            icon="hammer",
            tooltip="Take what they have and hand it to Technical.",
        ),
    ]


def _is_checkin_text(content: str) -> bool:
    text = content or ""
    return _CHECKIN_MARKER in text and "How do you want to proceed?" in text


async def _ask_how_to_proceed() -> None:
    note = _checkin_text()
    _remember("Front desk", note)
    bubble = _bubble("Front desk", note, actions=_checkin_actions())
    await bubble.send()
    cl.user_session.set(PENDING_KEY, True)
    cl.user_session.set(STALLED_KEY, True)
    cl.user_session.set(CHECKIN_MSG_KEY, bubble)


async def _dismiss_checkin() -> None:
    bubble = cl.user_session.get(CHECKIN_MSG_KEY)
    cl.user_session.set(PENDING_KEY, False)
    cl.user_session.set(CHECKIN_MSG_KEY, None)
    actions = list(getattr(bubble, "actions", None) or [])
    for action in actions:
        try:
            await action.remove()
        except Exception as exc:
            logger.warning("Could not remove check-in button (%s).", exc)


async def _announce_failures(failed: list[str]) -> None:
    await _bubble(
        "Front desk",
        _nameplate("Front desk")
        + f"{', '.join(failed)} could not reply this turn, so the plan above is "
        "missing their input. Fix the error shown in their message and send again.",
    ).send()


async def _wrap_up(failed: list[str], agreed: bool, *, check_in: bool) -> None:
    if failed:
        await _announce_failures(failed)
        return
    if agreed:
        cl.user_session.set(STALLED_KEY, False)
        return
    if check_in:
        await _ask_how_to_proceed()


async def _handle_agreement_decision(
    request: str,
    incoming: IncomingFiles,
    *,
    choice: Optional[str] = None,
) -> None:
    notes = parse_named_instructions(request)
    parsed = parse_agreement_choice(request)
    explicit_path = choice is not None or parsed is not None
    if choice is None:
        choice = parsed
    if choice is None:
        choice = "technical" if list(notes) == ["Technical"] else "continue"
        explicit_path = list(notes) == ["Technical"]

    named_steer = bool(notes) and choice == "continue" and not explicit_path
    if named_steer:
        recipients = [name for name in notes if name != "Technical"]
        who = ", ".join(recipients) or "the team"
        ack = (
            f"{_nameplate('Front desk')}I'll pass that to {who} and have Executor and "
            "Evaluator take another pass."
        )
        if "Technical" in notes:
            ack += " I'll hold Technical until they finish."
    else:
        label = CHOICE_LABELS[choice]
        ack = f"{_nameplate('Front desk')}Got it — {label.lower()}."
        recipients = [
            name for name in notes if choice == "technical" or name != "Technical"
        ]
        if recipients:
            ack += f" I'll also pass your instruction to {', '.join(recipients)}."
        if choice != "technical" and "Technical" in notes:
            ack += " I'll hold Technical until Executor and Evaluator finish."
    _remember("Front desk", ack)
    await _bubble("Front desk", ack).send()

    failed: list[str] = []
    if choice == "technical":
        for employee in TEAM:
            if employee.name == "Technical" or employee.name not in notes:
                continue
            await _run_turn(
                employee,
                request,
                incoming,
                failed,
                _directed_brief(employee, notes[employee.name], False),
            )
        tech_brief = TECHNICAL_HANDOFF
        if notes.get("Technical"):
            tech_brief = _user_note_brief("Technical", notes["Technical"]) + tech_brief
        if not failed:
            await _send_to_technical(request, incoming, failed, tech_brief)
        await _wrap_up(failed, True, check_in=False)
        return

    if choice == "compromise":
        executor_extra = COMPROMISE_EXECUTOR
        evaluator_extra = COMPROMISE_EVALUATOR
        rounds = max(1, min(2, MAX_AGREEMENT_ROUNDS))
        last_hint = (
            "\nThis is the last compromise round. Agree if you can stand behind the "
            "most-shared plan; keep AGREEMENT: no if it is still wrong or off-goal.\n"
        )
    else:
        executor_extra = EXECUTOR_REBUTTAL + CONTINUE_HINT
        evaluator_extra = EVALUATOR_REBUTTAL + CONTINUE_HINT
        rounds = 1 if named_steer else MAX_AGREEMENT_ROUNDS
        last_hint = (
            "\nThis is the last debate round. Do not fake consensus. "
            "If you cannot independently agree, keep AGREEMENT: no.\n"
        )

    if notes.get("Executor"):
        executor_extra = _user_note_brief("Executor", notes["Executor"]) + executor_extra
    if notes.get("Evaluator"):
        evaluator_extra = (
            _user_note_brief("Evaluator", notes["Evaluator"]) + evaluator_extra
        )

    lead = None
    if "Evaluator" in notes and "Executor" not in notes:
        lead = "Evaluator"
    elif "Executor" in notes and "Evaluator" not in notes:
        lead = "Executor"

    agreed = await _agreement_rounds(
        request,
        incoming,
        failed,
        rounds=rounds,
        executor_extra=executor_extra,
        evaluator_extra=evaluator_extra,
        lead=lead,
        last_round_hint=last_hint,
    )
    if not failed and agreed:
        await _send_to_technical(
            request, incoming, failed, TECHNICAL_AFTER_AGREEMENT
        )
    await _wrap_up(failed, agreed, check_in=True)


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
        f"- They argue up to {MAX_AGREEMENT_ROUNDS} times. If they still don't independently "
        "agree, I check in with you before Technical starts.",
        f"- **Technical** (`{TECHNICAL_MODEL}`) — technical deployment plan, and does the "
        "work if you asked for action",
        "",
        "Address someone by name to instruct just them, e.g. `Executor, drop Kubernetes` "
        "or `@Evaluator you're too strict on tests`.",
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
    cl.user_session.set(PENDING_KEY, False)
    cl.user_session.set(CHECKIN_MSG_KEY, None)
    cl.user_session.set(BUSY_KEY, False)
    cl.user_session.set(STALLED_KEY, False)
    await _bubble("Front desk", "\n".join(_welcome_lines())).send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Reload an old conversation so the team keeps its memory of it."""
    cl.user_session.set("thread_named", True)
    cl.user_session.set(CHECKIN_MSG_KEY, None)
    cl.user_session.set(BUSY_KEY, False)
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
    pending = bool(history) and history[-1].startswith("Front desk:") and _is_checkin_text(
        history[-1]
    )
    cl.user_session.set(PENDING_KEY, pending)
    cl.user_session.set(STALLED_KEY, pending)


async def _busy_guard() -> bool:
    if cl.user_session.get(BUSY_KEY):
        await _bubble(
            "Front desk",
            _nameplate("Front desk")
            + "The team is still working on the last instruction. Wait for them to finish.",
        ).send()
        return True
    return False


async def _run_user_request(request: str, incoming: IncomingFiles) -> None:
    cl.user_session.set(BUSY_KEY, True)
    try:
        if cl.user_session.get(PENDING_KEY):
            await _dismiss_checkin()
            await _handle_agreement_decision(request, incoming)
            return

        notes = parse_named_instructions(request)
        if notes:
            failed, agreed = await _directed_turns(request, incoming, notes)
            stalled = bool(cl.user_session.get(STALLED_KEY))
            both_named = "Executor" in notes and "Evaluator" in notes
            await _wrap_up(
                failed, agreed, check_in=stalled and both_named and not agreed
            )
            return

        failed, agreed = await _collaborate(request, incoming)
        await _wrap_up(failed, agreed, check_in=not agreed)
    finally:
        cl.user_session.set(BUSY_KEY, False)


@cl.action_callback("agreement_choice")
async def on_agreement_choice(action: cl.Action):
    if missing := missing_provider_keys():
        await _bubble(
            "Front desk",
            _nameplate("Front desk")
            + "The team can't reply until these keys are set on the server: "
            + ", ".join(f"`{key}`" for key in missing),
        ).send()
        return
    if await _busy_guard():
        return
    if not cl.user_session.get(PENDING_KEY):
        return
    choice = (action.payload or {}).get("choice")
    if choice not in CHOICE_LABELS:
        return
    label = CHOICE_LABELS[choice]
    _remember("You", label)
    incoming = IncomingFiles(prompt_block="", summary="")
    cl.user_session.set(BUSY_KEY, True)
    try:
        await _dismiss_checkin()
        await _handle_agreement_decision(label, incoming, choice=choice)
    finally:
        cl.user_session.set(BUSY_KEY, False)


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
    if await _busy_guard():
        return

    request = (message.content or "").strip() or "Please review the attached files."
    stash = FileWorkspace()
    try:
        incoming = ingest(list(message.elements or []), stash.uploads)
        title_source = (message.content or "").strip() or incoming.summary or request
        await _name_thread(title_source)
        _remember("You", request + (f"\n[{incoming.summary}]" if incoming.summary else ""))
        await _run_user_request(request, incoming)
    finally:
        stash.close()

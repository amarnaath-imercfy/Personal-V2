"""Personal Tutor RAG Agent — locked to a single lesson file per session."""

import logging
from typing import Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

from . import tools

logger = logging.getLogger(__name__)

# ── Keys expected in the structured first message ──────────────────────────
_EXPECTED_KEYS = [
    "Student_Name",
    "Student_Board",
    "Student_Grade",
    "Subject_Name",
    "Lesson_Name",
    "File_Id",
]


def _parse_student_context(callback_context: CallbackContext) -> Optional[genai_types.Content]:
    """Before-agent callback: parse the structured first message and lock context.

    On the very first message, parses student info, locks it in state, and
    returns the greeting Content directly — bypassing the agent entirely so
    there is exactly ONE clean response with zero tool calls.

    On every subsequent turn, student_name is already set so None is returned
    and the agent runs normally.
    """
    # Already initialized — let the agent handle this turn normally
    if callback_context.state.get("student_name"):
        return None

    user_content = callback_context.user_content
    if not user_content or not user_content.parts:
        return None

    raw_text = ""
    for part in user_content.parts:
        if hasattr(part, "text") and part.text:
            raw_text = part.text
            break

    parsed: dict[str, str] = {}
    for line in raw_text.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            parsed[key.strip()] = value.strip()

    # Only proceed when all required keys are present
    if not all(k in parsed for k in _EXPECTED_KEYS):
        return None

    # Lock all student info into session state (immutable for the session)
    callback_context.state["student_name"] = parsed["Student_Name"]
    callback_context.state["student_board"] = parsed["Student_Board"]
    callback_context.state["student_grade"] = parsed["Student_Grade"]
    callback_context.state["subject_name"] = parsed["Subject_Name"]
    callback_context.state["lesson_name"] = parsed["Lesson_Name"]
    callback_context.state["file_id"] = parsed["File_Id"]

    logger.info(
        "Student context locked — name=%s, file_id=%s",
        parsed["Student_Name"],
        parsed["File_Id"],
    )

    # Return greeting Content directly — agent is skipped, single clean response
    greeting = (
        f"Hi {parsed['Student_Name']}! I've got **{parsed['Lesson_Name']}** ready. "
        "If you have a specific topic you'd like to know about, feel free to ask, "
        "or I can suggest some important concepts!"
    )
    return genai_types.Content(
        role="model",
        parts=[genai_types.Part.from_text(text=greeting)],
    )


# ── Agent instruction ──────────────────────────────────────────────────────
INSTRUCTION = """\
You are a personal tutor for a school student.

Session (locked — never change):
Student: {student_name} | Board: {student_board} | Grade: {student_grade}
Subject: {subject_name} | Lesson: {lesson_name}

Rules:
1. Always call a tool before answering. Base reply ONLY on retrieved content.
2. If tool returns no_results → say "I couldn't find that in the lesson, try rephrasing."
3. Never use your own knowledge — lesson file is the only source of truth.
4. Keep language simple for Grade {student_grade}. Stay on topic: {lesson_name} only.

If student says "yes" / asks for topics / wants suggestions:
  → Call query_lesson with query="{lesson_name}" → extract headings/section titles from the returned chunks → numbered list → ask which to start.

For any question:
  → Call query_lesson → explain from retrieved content only → short comprehension check.
"""


root_agent = Agent(
    model="gemini-2.5-flash",
    name="personal_tutor",
    description=(
        "A personal tutor that answers school students' questions strictly "
        "from their assigned lesson file using Vertex AI RAG."
    ),
    instruction=INSTRUCTION,
    tools=[
        tools.query_lesson,
    ],
    before_agent_callback=_parse_student_context,
)

"""Every prompt text used by the experiments (design §5.2, §3.3 M1/M4, §10.9-6). The paper quotes this module.

Rendering: the messages below go through the model's chat template with add_generation_prompt=True and
enable_thinking=False (Qwen3; templates without that variable ignore it). The student prompt is the English
student prompt of the English record, unchanged; only the question language changes (no language instruction).
"""
from __future__ import annotations

from .util import LANGUAGE_NAMES

# English record (continual-pair-opd lm_route_runtime.RouteRuntime.prompt), kept verbatim.
STUDENT_INSTRUCTION = "Explain your reasoning briefly. End your response with #### followed by the final numeric answer."

# T-en: a system instruction on the teacher side only (design §3.3 M1: "teacher 시스템 지시").
TEACHER_EN_SYSTEM = "Write your reasoning in English, even when the question is in another language."

# T-priv: the teacher also sees the English parallel question (design §3.3 M4).
TEACHER_PRIV_BRIDGE = "The same problem in English:"

# S0-b (design §5.3, §10.9-6): one extra line after the student instruction. a_l = acc(solve_en) - acc(solve_native).
S0B_LINES = {
    "solve_en": "Write your reasoning in English.",
    "solve_native": "Write your reasoning in {language}.",
    "control_format": "End your response with a line of the form \"#### N\", where N is only the final number.",
    "control_language": "Write your reasoning in the same language as the question.",
}

TEACHER_CONDITIONS = ("T-def", "T-en", "T-priv")


def student_messages(question, extra_line=None):
    content = question + "\n" + STUDENT_INSTRUCTION
    if extra_line:
        content += "\n" + extra_line
    return [{"role": "user", "content": content}]


def s0b_messages(question, condition, language):
    line = S0B_LINES[condition].format(language=LANGUAGE_NAMES[language])
    return student_messages(question, line)


def teacher_messages(question, condition, english_question=None):
    """T-def: the student prompt. T-en: the student prompt plus TEACHER_EN_SYSTEM as a system message.
    T-priv: the question, then TEACHER_PRIV_BRIDGE and the English parallel question, then the student instruction.
    For an English question T-priv equals T-def (there is no other parallel text)."""
    if condition == "T-def":
        return student_messages(question)
    if condition == "T-en":
        return [{"role": "system", "content": TEACHER_EN_SYSTEM}] + student_messages(question)
    if condition == "T-priv":
        if english_question is None:
            raise ValueError("T-priv needs the English parallel question")
        if english_question.strip() == question.strip():
            return student_messages(question)
        content = question + "\n\n" + TEACHER_PRIV_BRIDGE + "\n" + english_question + "\n" + STUDENT_INSTRUCTION
        return [{"role": "user", "content": content}]
    raise ValueError(f"unknown teacher condition {condition}")


def all_prompt_texts():
    """For the paper appendix and the run manifest."""
    return {"student_instruction": STUDENT_INSTRUCTION, "teacher_en_system": TEACHER_EN_SYSTEM,
            "teacher_priv_bridge": TEACHER_PRIV_BRIDGE, "s0b_lines": dict(S0B_LINES),
            "student_message_example": student_messages("<question>"),
            "teacher_messages_example": {c: teacher_messages("<question>", c, "<English question>")
                                         for c in TEACHER_CONDITIONS},
            "rendering": "chat template, add_generation_prompt=True, enable_thinking=False"}

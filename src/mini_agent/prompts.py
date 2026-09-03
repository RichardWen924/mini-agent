from collections.abc import Sequence

from .models import Message, Role


def trim_history(
    history: Sequence[Message],
    max_messages: int,
) -> list[Message]:
    """保留最近的历史消息，并返回一个新的列表。"""
    if max_messages < 0:
        raise ValueError("max_messages must be greater than or equal to zero")

    if max_messages == 0:
        return []

    if len(history) <= max_messages:
        return list(history)

    return list(history[-max_messages:])


def build_prompt(
    system_prompt: str,
    history: Sequence[Message],
    user_input: str,
    max_history: int,
) -> list[Message]:
    """组合系统提示词、有限历史和当前用户问题。"""
    if not system_prompt.strip():
        raise ValueError("system_prompt must not be empty")

    if not user_input.strip():
        raise ValueError("user_input must not be empty")

    trimmed_history = trim_history(history, max_history)
    prompt_messages = [Message(role=Role.SYSTEM, content=system_prompt)]
    prompt_messages.extend(trimmed_history)
    prompt_messages.append(Message(role=Role.USER, content=user_input))

    return prompt_messages

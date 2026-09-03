from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        """在消息创建后校验角色和内容的运行时类型。"""

        if not isinstance(self.role, Role):
            raise TypeError(f"role must be a Role, got {type(self.role).__name__}")

        if not isinstance(self.content, str):
            raise TypeError(
                f"content must be a string, got {type(self.content).__name__}"
            )

    def is_empty(self) -> bool:
        """判断消息内容是否为空或只包含空白字符。"""

        return not self.content.strip()

    def to_dict(self) -> dict[str, str]:
        """将消息转换为模型接口常用的字典格式。"""

        return {"role": self.role.value, "content": self.content}

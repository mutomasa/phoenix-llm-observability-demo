"""Tool Executor から呼ばれる安全なローカルツール（外部APIは使わない）。"""

import ast
import json
import operator
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Protocol


class ToolExecutionError(RuntimeError):
    """ツール実行に失敗した場合に送出する。"""


class Tool(Protocol):
    name: str
    description: str

    def run(self, tool_input: str) -> str: ...


_BINARY_OPERATORS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type, Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](
            _evaluate_node(node.left), _evaluate_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
    raise ToolExecutionError(f"計算できない式です: {ast.dump(node)}")


def normalize_expression(expression: str) -> str:
    return expression.replace("×", "*").replace("÷", "/").replace("＊", "*").strip()


class Calculator:
    name = "calculator"
    description = "四則演算などの数式を計算する。入力例: 128*46"

    def run(self, tool_input: str) -> str:
        try:
            tree = ast.parse(normalize_expression(tool_input), mode="eval")
        except SyntaxError as exc:
            raise ToolExecutionError(f"数式を解析できません: {tool_input!r}") from exc
        value = _evaluate_node(tree.body)
        return str(int(value) if float(value).is_integer() else value)


class KnowledgeSearch:
    name = "knowledge_search"
    description = "ローカルの知識ベースから用語の説明を検索する。入力例: Phoenix"

    def __init__(self, path: Path) -> None:
        self._entries = json.loads(path.read_text(encoding="utf-8"))

    def run(self, tool_input: str) -> str:
        query = tool_input.lower()
        for entry in self._entries:
            if any(keyword in query for keyword in entry["keywords"]):
                return f"{entry['title']}: {entry['content']}"
        return "該当する情報は見つかりませんでした。"


class CurrentDate:
    name = "current_date"
    description = "今日の日付を返す。入力は不要。"

    def __init__(self, today: Callable[[], date] = date.today) -> None:
        self._today = today

    def run(self, tool_input: str) -> str:
        return self._today().isoformat()


class FailingTool:
    """エラーシナリオ用に、元のツールの代わりに必ず例外を送出する。"""

    def __init__(self, tool: Tool) -> None:
        self.name = tool.name
        self.description = tool.description

    def run(self, tool_input: str) -> str:
        raise ToolExecutionError(f"[injected] {self.name} の実行に失敗しました")


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolExecutionError(f"未知のツールです: {name}")
        return self._tools[name]

    def with_failure(self, name: str) -> "ToolRegistry":
        return ToolRegistry([FailingTool(t) if t.name == name else t for t in self._tools.values()])


DEFAULT_KNOWLEDGE_PATH = Path("datasets/knowledge.json")


def create_default_registry(knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH) -> ToolRegistry:
    return ToolRegistry([Calculator(), KnowledgeSearch(knowledge_path), CurrentDate()])

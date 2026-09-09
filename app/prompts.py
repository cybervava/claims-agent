"""Prompts live as Markdown templates under prompts/ — never inline in code.

Templates use `${name}` placeholders (string.Template) so JSON braces in prompt
bodies are safe. Each render records the template's content hash so audit rows
can pin exactly which prompt version produced an output.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Template


@dataclass(frozen=True)
class RenderedPrompt:
    name: str
    version: str
    text: str


class PromptStore:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.is_dir():
            raise FileNotFoundError(f"prompts dir not found: {self.prompts_dir}")

    def _path(self, name: str) -> Path:
        path = self.prompts_dir / f"{name}.md"
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")
        return path

    def raw(self, name: str) -> str:
        return self._path(name).read_text(encoding="utf-8")

    def version(self, name: str) -> str:
        return hashlib.sha256(self.raw(name).encode("utf-8")).hexdigest()[:12]

    def render(self, name: str, /, **variables: object) -> RenderedPrompt:
        raw = self.raw(name)
        text = Template(raw).safe_substitute({k: str(v) for k, v in variables.items()})
        return RenderedPrompt(name=name, version=self.version(name), text=text)

    def list(self) -> list[dict[str, str]]:
        return [
            {"name": p.stem, "version": self.version(p.stem)}
            for p in sorted(self.prompts_dir.glob("*.md"))
        ]

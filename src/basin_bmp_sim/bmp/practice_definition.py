"""Model-neutral metadata for BMP practice definitions."""
from dataclasses import dataclass

@dataclass(frozen=True)
class PracticeDefinition:
    cps: int
    name: str
    response_family: str

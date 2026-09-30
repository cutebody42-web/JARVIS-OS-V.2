"""NEXUS persona definitions."""

from .persona_spec import PersonaRegistry, PersonaSpec
from .profiles import BUILTIN_PERSONAS, FRIDAY, JARVIS, TABY

__all__ = [
    "BUILTIN_PERSONAS",
    "FRIDAY",
    "JARVIS",
    "TABY",
    "PersonaRegistry",
    "PersonaSpec",
]

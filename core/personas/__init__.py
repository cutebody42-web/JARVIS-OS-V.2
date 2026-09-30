"""Built-in NEXUS persona specifications."""

from core.personas.friday import FRIDAY
from core.personas.jarvis import JARVIS
from core.personas.persona_spec import PersonaSpec
from core.personas.taby import TABY

BUILTIN_PERSONAS = {
    TABY.name: TABY,
    FRIDAY.name: FRIDAY,
    JARVIS.name: JARVIS,
}

__all__ = [
    "BUILTIN_PERSONAS",
    "FRIDAY",
    "JARVIS",
    "PersonaSpec",
    "TABY",
]

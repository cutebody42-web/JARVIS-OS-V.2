"""Built-in NEXUS personas."""

from core.personas.friday import FRIDAY
from core.personas.jarvis import JARVIS
from core.personas.persona_spec import PersonaRegistry, PersonaSpec
from core.personas.taby import TABY

DEFAULT_PERSONAS = PersonaRegistry(
    {
        TABY.name: TABY,
        FRIDAY.name: FRIDAY,
        JARVIS.name: JARVIS,
    }
)

__all__ = [
    "DEFAULT_PERSONAS",
    "FRIDAY",
    "JARVIS",
    "PersonaRegistry",
    "PersonaSpec",
    "TABY",
]

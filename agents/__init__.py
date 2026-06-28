"""Agent Prompts Package — 四代理协作系统。"""

from agents.alice import create_alice_prompt
from agents.bob import create_bob_prompt
from agents.charlie import create_charlie_prompt
from agents.diana import create_diana_prompt

__all__ = [
    "create_alice_prompt",
    "create_bob_prompt",
    "create_charlie_prompt",
    "create_diana_prompt",
]

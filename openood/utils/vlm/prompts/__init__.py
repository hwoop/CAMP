from .prompts import (
    register_prompt,
    get_templates,
    list_registered_prompts,
)

# Import side effects: register built-in prompt sets.
from . import imagenet
from . import csp

__all__ = [
    "register_prompt",
    "get_templates",
    "list_registered_prompts",
]

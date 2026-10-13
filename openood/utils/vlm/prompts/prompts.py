from typing import Dict, List


_PROMPT_REGISTRY: Dict[str, List[str]] = {}


def register_prompt(
    name: str,
    templates: List[str],
) -> None:

    name = name.lower()

    if not isinstance(templates, list) or len(templates) == 0:
        raise ValueError("templates must be a non-empty list of strings.")

    for template in templates:
        if not isinstance(template, str):
            raise TypeError(f"Each template must be str, got {type(template)}.")

        if "{}" not in template:
            raise ValueError(
                f"Prompt template must contain '{{}}' placeholder: {template}"
            )

    if name in _PROMPT_REGISTRY:
        raise ValueError(f"Prompt '{name}' is already registered.")

    _PROMPT_REGISTRY[name] = templates


def get_templates(
    prompt_name: str,
) -> List[str]:

    prompt_name = prompt_name.lower()
    if prompt_name not in _PROMPT_REGISTRY:
        available = list_registered_prompts()
        raise NotImplementedError(
            f"Unknown prompt '{prompt_name}'. Available prompts: {available}"
        )

    return _PROMPT_REGISTRY[prompt_name]


def list_registered_prompts() -> List[str]:
    return sorted(_PROMPT_REGISTRY.keys())

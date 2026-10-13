from openood.utils.vlm.prompts.prompts import register_prompt


NICE_TEMPLATES = [
    "The nice {}.",
]


# MCM paper, Sec. 3.
MCM_TEMPLATES = [
    "this is a photo of a {}",
]


NEG_ADJ_TEMPLATES = [
    "This is a {} photo.",
]


def register_imagenet_prompts() -> None:

    register_prompt(name="nice", templates=NICE_TEMPLATES)

    register_prompt(name="neg_adj", templates=NEG_ADJ_TEMPLATES)

    register_prompt(name="mcm", templates=MCM_TEMPLATES)


register_imagenet_prompts()

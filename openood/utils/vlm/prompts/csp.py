from openood.utils.vlm.prompts.prompts import register_prompt


CSP_NOUN_TEMPLATES = [
    "a close-up photo of the nice {}.",
    "a photo with the nice {}.",
    "a good photo with the nice {}.",
    "a photo of the nice {}.",
    "{}",
    "the {}",
    "the good {}",
    "the nice {}",
]


CSP_ADJ_TEMPLATES = [
    "{}",
    "a nice photo of {}",
    "a good photo of {}",
    "a close-up photo of {}",
]


CSP_SUPERCLASS_TEMPLATES = [
    "{} environment",
    "{} space",
    "{} object",
    "{} scene",
    "{} place",
    "{} item",
    "{} area",
    "{} view",
    "{} thing",
    "{} pattern",
    "{} structure",
    "{} landscape",
    "{} vista",
    "{} creature",
]


def register_csp_prompts() -> None:

    register_prompt(name="csp_noun", templates=CSP_NOUN_TEMPLATES)
    register_prompt(name="csp_adj", templates=CSP_ADJ_TEMPLATES)
    register_prompt(name="csp_superclass", templates=CSP_SUPERCLASS_TEMPLATES)


register_csp_prompts()

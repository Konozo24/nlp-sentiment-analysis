"""
Named-entity helpers: BIO tag parsing and Streamlit-markdown rendering.
"""

import re

# entity type -> streamlit markdown color, used for both highlight and badge
ENTITY_COLORS = {
    "PER": "orange",
    "ORG": "blue",
    "LOC": "green",
    "EVENT": "violet",
}

# characters that would otherwise be read as markdown syntax; brackets matter
# most, since they would terminate the :color-background[...] directive early
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_\[\]()~$])")


def has_entities(ner_tags: list[tuple[str, str]]) -> bool:
    return any(tag != "O" for _, tag in ner_tags)


def format_entities(ner_tags: list[tuple[str, str]]) -> str:
    """Turn BIO tags into markdown with each entity chunk highlighted and badged."""
    pieces: list[str] = []
    i = 0
    while i < len(ner_tags):
        word, tag = ner_tags[i]
        i += 1
        if tag == "O":
            pieces.append(_escape(word))
            continue

        entity_type = tag.split("-", 1)[1]
        chunk = [word]
        while i < len(ner_tags) and ner_tags[i][1] == f"I-{entity_type}":
            chunk.append(ner_tags[i][0])
            i += 1

        color = ENTITY_COLORS.get(entity_type, "gray")
        text = _escape(" ".join(chunk))
        pieces.append(f":{color}-background[{text}] :{color}-badge[{entity_type}]")
    return " ".join(pieces)


def format_legend() -> str:
    return " ".join(f":{color}-badge[{name}]" for name, color in ENTITY_COLORS.items())


def _escape(text: str) -> str:
    return _MARKDOWN_SPECIALS.sub(r"\\\1", text)

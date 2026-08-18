"""Convert spaCy entities to display-friendly BIO tags for SVM predictions."""

NER_LABEL_MAP = {"PERSON": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC", "EVENT": "EVENT"}


def format_bio_entities(text: str) -> str:
    """Return the tweet with detected spaCy entities annotated as BIO tags."""
    import spacy  # Load lazily so normal model imports stay lightweight.
    doc = spacy.load("en_core_web_trf")(text)
    tags = ["O"] * len(doc)
    for entity in doc.ents:
        entity_type = NER_LABEL_MAP.get(entity.label_)
        if entity_type:
            for index in range(entity.start, entity.end):
                tags[index] = f"{'B' if index == entity.start else 'I'}-{entity_type}"
    return " ".join(f"{token.text} [{tag}]" if tag != "O" else token.text for token, tag in zip(doc, tags))

import re
import emoji

def demojize_emoji(text: str, language: str = "en") -> str:
    """Convert emoji into readable text names using the specified language."""
    return emoji.demojize(text, language=language)

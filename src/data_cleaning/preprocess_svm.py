from .utils import demojize_emoji, remove_links


def clean_for_svm(text: str, language: str = "en") -> str:
    text = remove_links(text)
    text = demojize_emoji(text, language=language)
    return text

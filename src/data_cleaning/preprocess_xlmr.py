from .utils import remove_links


def clean_for_xlmr(text: str) -> str:
    return remove_links(text)

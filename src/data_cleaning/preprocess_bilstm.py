from .utils import remove_links


def clean_for_bilstm(text: str) -> str:
    return remove_links(text)

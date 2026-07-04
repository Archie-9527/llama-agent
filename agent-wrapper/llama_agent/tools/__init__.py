"""Tools available to the local agent."""

from langchain_core.tools import tool


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of English text.

    Args:
        text: The text to count words in.

    Returns:
        The number of words.
    """
    return len(text.split())

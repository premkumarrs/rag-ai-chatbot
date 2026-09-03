"""Lightweight deterministic query normalization."""

from __future__ import annotations

import re
import unicodedata

# Informal / low-literacy substitutions applied only to whole words.
_INFORMAL_MAP = {
    "wat": "what",
    "wht": "what",
    "whats": "what's",
    "pls": "please",
    "plz": "please",
    "thru": "through",
    "abt": "about",
    "becuz": "because",
    "bcoz": "because",
    "cuz": "because",
    "hw": "how",
    "teh": "the",
    "adn": "and",
    "nd": "and",
    "n": "and",
    "ur": "your",
    "u": "you",
    "r": "are",
    "y": "why",
    "im": "i'm",
    "dont": "don't",
    "doesnt": "doesn't",
    "cant": "can't",
    "wont": "won't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "havent": "haven't",
    "hasnt": "hasn't",
    "shouldnt": "shouldn't",
    "couldnt": "couldn't",
    "wouldnt": "wouldn't",
}

# Tokens that look like technical identifiers must never be rewritten.
_TECH_TOKEN = re.compile(
    r"^(?:"
    r"[A-Z]{2,}[0-9]+[A-Z0-9\-]*|"  # ADXL345, ESP32, FD001
    r"[A-Z]+-[A-Z0-9]+|"            # C-MAPSS
    r"[A-Za-z]*\d+[A-Za-z0-9\-]*|"  # DS18B20, qwen3
    r"[A-Z]{3,}"                    # RUL, MQTT (all-caps acronyms)
    r")$"
)

_WORD = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[^\s]")


def _is_technical_token(token: str) -> bool:
    if _TECH_TOKEN.match(token):
        return True
    # Mixed alphanumeric identifiers such as ADXL345 already covered;
    # also protect tokens containing digits.
    return any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token)


def extract_technical_terms(text: str) -> list[str]:
    """Extract likely technical identifiers from a query or chunk."""
    terms: list[str] = []
    for match in re.finditer(r"[A-Za-z0-9][A-Za-z0-9\-_/]*", text):
        token = match.group(0)
        if _is_technical_token(token):
            terms.append(token)
    return terms


def normalize_query(query: str) -> str:
    """
    Normalize a user question without calling an LLM.

    Safe operations only: whitespace, punctuation spacing, informal wording,
    and light contractions. Technical identifiers are preserved exactly.
    """
    if not query:
        return ""

    text = unicodedata.normalize("NFKC", query)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize spaced punctuation: "What ?" -> "What?"
    text = re.sub(r"\s+([?!,.;:])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)}\]])", r"\1", text)

    tokens = _WORD.findall(text)
    normalized: list[str] = []
    for token in tokens:
        if _is_technical_token(token):
            normalized.append(token)
            continue

        lower = token.lower()
        if lower in _INFORMAL_MAP:
            replacement = _INFORMAL_MAP[lower]
            # Preserve original capitalization for sentence starts when useful.
            if token[:1].isupper() and replacement:
                replacement = replacement[0].upper() + replacement[1:]
            normalized.append(replacement)
        else:
            normalized.append(token)

    # Re-join: attach punctuation without leading spaces.
    result_parts: list[str] = []
    for token in normalized:
        if token in {",", ".", "!", "?", ";", ":", ")", "]", "}"}:
            if result_parts:
                result_parts[-1] = result_parts[-1] + token
            else:
                result_parts.append(token)
        elif token in {"(", "[", "{"}:
            result_parts.append(token)
        else:
            if result_parts and result_parts[-1] in {"(", "[", "{"}:
                result_parts[-1] = result_parts[-1] + token
            else:
                result_parts.append(token)

    result = " ".join(result_parts)
    result = re.sub(r"\s+", " ", result).strip()
    return result

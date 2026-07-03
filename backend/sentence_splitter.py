import re
from typing import List


ABBREVIATIONS = [
    # Malayalam titles and common abbreviations
    "ഡോ.", "ഡോ .",
    "അഡ്വ.", "അഡ്വ .",
    "പ്രൊഫ്.", "പ്രൊഫ് .",
    "ശ്രീ.", "ശ്രീ .",
    "ശ്രീമതി.", "ശ്രീമതി .",
    "കു.", "കു .",
    "ലഫ്.", "കേണൽ.", "ക്യാപ്റ്റൻ.",

    # Malayalam initials / name abbreviations
    "പി.കെ.", "കെ.ടി.", "എം.കെ.", "വി.കെ.", "എന്‍.", "എൻ.", "പി.", "കെ.", "ടി.", "എം.", "വി.",

    # English abbreviations that may appear in Malayalam text
    "Dr.", "Adv.", "Prof.", "Mr.", "Mrs.", "Ms.",
    "Lt.", "Col.", "Capt.", "Gen.", "Sr.", "Jr.",
    "No.", "Dept.", "Govt.", "Ltd.", "Pvt.", "Inc.",
    "etc.", "e.g.", "i.e.",
]


def _protect_abbreviations(text: str) -> str:
    protected = text

    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

    # Protect repeated initials like പി.കെ., കെ.എസ്., A.K., P.K.
    protected = re.sub(
        r"((?:[A-Za-zഅ-ഹ]\s*\.\s*){2,})",
        lambda match: match.group(0).replace(".", "<DOT>"),
        protected,
    )

    return protected


def _restore_abbreviations(text: str) -> str:
    return text.replace("<DOT>", ".")


def clean_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = re.sub(r"^[\-–—•*]+\s*", "", sentence)
    sentence = re.sub(r"^\d+[.)]\s*", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


def segment_malayalam_text(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    raw_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    normalized = re.sub(r"\s+", " ", text.strip())

    protected = _protect_abbreviations(normalized)

    parts = re.split(r"(?<=[.!?।])\s+", protected)

    sentences = []
    for part in parts:
        restored = _restore_abbreviations(part)
        cleaned = clean_sentence(restored)
        if len(cleaned) > 2:
            sentences.append(cleaned)

    # Fallback for line-separated input with weak punctuation.
    if len(sentences) <= 1 and len(raw_lines) > 1:
        sentences = []
        for line in raw_lines:
            restored = _restore_abbreviations(_protect_abbreviations(line))
            cleaned = clean_sentence(restored)
            if len(cleaned) > 2:
                sentences.append(cleaned)

    return sentences
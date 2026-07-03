import re
from typing import List


# -----------------------------------------------------------------------------
# Explicit abbreviations
# -----------------------------------------------------------------------------

TITLE_ABBREVIATIONS = [
    # Malayalam titles / designations
    "ഡോ.",
    "ഡോ .",
    "അഡ്വ.",
    "അഡ്വ .",
    "പ്രൊഫ്.",
    "പ്രൊഫ് .",
    "ശ്രീ.",
    "ശ്രീ .",
    "ശ്രീമതി.",
    "ശ്രീമതി .",
    "കു.",
    "കു .",
    "ലഫ്.",
    "കേണൽ.",
    "ക്യാപ്റ്റൻ.",
    "ജന.",
    "ബ്രിഗ്.",
    "മേജർ.",

    # English titles / designations
    "Dr.",
    "Adv.",
    "Prof.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Lt.",
    "Col.",
    "Capt.",
    "Gen.",
    "Brig.",
    "Maj.",
    "Sr.",
    "Jr.",
    "No.",
    "Dept.",
    "Govt.",
    "Ltd.",
    "Pvt.",
    "Inc.",
    "etc.",
]


COMMON_MULTI_INITIAL_ABBREVIATIONS = [
    # Malayalam common multi-initial names
    "പി.കെ.",
    "കെ.ടി.",
    "എം.കെ.",
    "വി.കെ.",
    "എം.ടി.",
    "ടി.കെ.",
    "എൻ.കെ.",
    "എന്‍.കെ.",
    "എസ്.കെ.",
    "കെ.എസ്.",
    "പി.എസ്.",
    "എം.എസ്.",
    "എൻ.എസ്.",
    "എന്‍.എസ്.",
    "ടി.എം.",
    "വി.എസ്.",
    "കെ.എം.",
    "എം.ജി.",
    "സി.പി.",
    "കെ.പി.",
    "ആർ.കെ.",
    "ആര്‍.കെ.",
    "ആർ.എസ്.",
    "ആര്‍.എസ്.",
    "എം.ആർ.",
    "എം.ആര്‍.",
    "കെ.ആർ.",
    "കെ.ആര്‍.",
    "ടി.ആർ.",
    "ടി.ആര്‍.",
    "പി.ആർ.",
    "പി.ആര്‍.",
    "എസ്.ആർ.",
    "എസ്.ആര്‍.",
    "എ.ബി.",
    "ബി.സി.",
    "ഡി.കെ.",
    "ജി.കെ.",
    "ജെ.കെ.",
    "എച്ച്.കെ.",
    "എൽ.ഡി.",
    "എല്‍.ഡി.",
    "എൻ.ആർ.",
    "എന്‍.ആര്‍.",

    # English common initial forms
    "A.K.",
    "P.K.",
    "K.T.",
    "M.K.",
    "V.K.",
    "R.K.",
    "R.S.",
    "M.R.",
    "K.R.",
    "T.R.",
    "P.R.",
    "S.R.",

    # Latin abbreviations
    "e.g.",
    "i.e.",
]


# Malayalam letters used as standalone name initials in real articles.
# These are NOT globally replaced. They are protected only when they appear
# as separate tokens such as "ആർ. ശങ്കർ", not inside words like "ചൂണ്ടിക്കാട്ടി."
MALAYALAM_SINGLE_INITIALS = [
    "എ",
    "ബി",
    "സി",
    "ഡി",
    "ഇ",
    "എഫ്",
    "ജി",
    "എച്ച്",
    "ഐ",
    "ജെ",
    "കെ",
    "എൽ",
    "എല്‍",
    "എം",
    "എൻ",
    "എന്‍",
    "ഒ",
    "ഓ",
    "പി",
    "ക്യു",
    "ആർ",
    "ആര്‍",
    "എസ്",
    "ടി",
    "യു",
    "വി",
    "ഡബ്ല്യു",
    "എക്സ്",
    "വൈ",
    "സെഡ്",
]


ENGLISH_SINGLE_INITIAL_PATTERN = r"[A-Za-z]"


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _safe_dot(text: str) -> str:
    return text.replace(".", "<DOT>")


def _restore_abbreviations(text: str) -> str:
    return text.replace("<DOT>", ".")


def _protect_exact_abbreviations(text: str, abbreviations: List[str]) -> str:
    protected = text

    for abbr in sorted(abbreviations, key=len, reverse=True):
        protected = protected.replace(abbr, _safe_dot(abbr))

    return protected


def _protect_malayalam_single_initials(text: str) -> str:
    """
    Protect standalone Malayalam initials.

    Examples protected:
        ആർ. ശങ്കർ
        പ്രൊഫ്. ആർ. ശങ്കർ
        പി. കെ. ശശി
        കെ. സുരേഷ്

    Examples NOT protected:
        ചൂണ്ടിക്കാട്ടി.
        നടത്തി.
        അറിയിച്ചു.
    """
    protected = text

    initial_group = "|".join(re.escape(initial) for initial in sorted(MALAYALAM_SINGLE_INITIALS, key=len, reverse=True))

    # A valid initial must:
    # - not be preceded by another Malayalam character or word character
    # - be followed by a dot
    # - after the dot, be followed by whitespace and then a likely name/initial,
    #   or be at end of a multi-initial chain.
    #
    # This prevents matching the "ടി." inside "ചൂണ്ടിക്കാട്ടി."
    pattern = rf"(?<![\w\u0D00-\u0D7F])({initial_group})\.(?=\s+(?:[\u0D00-\u0D7F]|[A-Za-z])|$)"

    return re.sub(
        pattern,
        lambda match: match.group(1) + "<DOT>",
        protected,
    )


def _protect_malayalam_repeated_initials(text: str) -> str:
    """
    Protect repeated Malayalam initials written with spaces:
        പി. കെ. ശശി
        കെ. എസ്. ചിത്ര
        ആർ. കെ. നായർ

    This runs after single-initial protection, so dots may already be <DOT>.
    """
    protected = text

    initial_group = "|".join(re.escape(initial) for initial in sorted(MALAYALAM_SINGLE_INITIALS, key=len, reverse=True))

    pattern = rf"(?<![\w\u0D00-\u0D7F])((?:{initial_group})<DOT>\s*){{2,}}"

    return re.sub(
        pattern,
        lambda match: match.group(0).replace(".", "<DOT>"),
        protected,
    )


def _protect_english_initials(text: str) -> str:
    """
    Protect English initials:
        A.K. Antony
        P. K. Menon
        R. Shankar
    """
    protected = text

    # Repeated initials like A.K. or P. K.
    protected = re.sub(
        rf"\b((?:{ENGLISH_SINGLE_INITIAL_PATTERN}\s*\.\s*){{2,}})",
        lambda match: _safe_dot(match.group(0)),
        protected,
    )

    # Single English initial before a name, like R. Shankar
    protected = re.sub(
        rf"\b({ENGLISH_SINGLE_INITIAL_PATTERN})\.(?=\s+[A-Z])",
        lambda match: match.group(1) + "<DOT>",
        protected,
    )

    return protected


def _protect_abbreviations(text: str) -> str:
    """
    Protect abbreviation dots before sentence splitting.

    Order matters:
    1. Protect known multi-part and title abbreviations.
    2. Protect standalone Malayalam initials dynamically.
    3. Protect English initials.
    """
    protected = text

    protected = _protect_exact_abbreviations(protected, COMMON_MULTI_INITIAL_ABBREVIATIONS)
    protected = _protect_exact_abbreviations(protected, TITLE_ABBREVIATIONS)

    protected = _protect_malayalam_single_initials(protected)
    protected = _protect_malayalam_repeated_initials(protected)
    protected = _protect_english_initials(protected)

    return protected


def clean_sentence(sentence: str) -> str:
    """
    Remove bullet/numbering noise without changing real sentence content.
    """
    sentence = sentence.strip()
    sentence = re.sub(r"^[\-–—•*]+\s*", "", sentence)
    sentence = re.sub(r"^\d+[.)]\s*", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


def _split_protected_text(protected: str) -> List[str]:
    """
    Split protected text into sentences.

    Sentence boundary:
    Split after . ! ? । when followed by whitespace or end of text.

    Abbreviation dots have already been converted to <DOT>, so this does not
    split inside ഡോ., അഡ്വ., ആർ., പി.കെ., Lt., Dr., etc.
    """
    sentences: List[str] = []
    buffer: List[str] = []

    i = 0

    while i < len(protected):
        char = protected[i]
        buffer.append(char)

        if char in ".!?।":
            next_char = protected[i + 1] if i + 1 < len(protected) else ""

            if not next_char or next_char.isspace():
                sentence = "".join(buffer).strip()
                if sentence:
                    sentences.append(sentence)

                buffer = []

                while i + 1 < len(protected) and protected[i + 1].isspace():
                    i += 1

        i += 1

    remaining = "".join(buffer).strip()
    if remaining:
        sentences.append(remaining)

    return sentences


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def segment_malayalam_text(text: str) -> List[str]:
    """
    Abbreviation-safe Malayalam sentence splitter.

    Handles:
    - Malayalam full stops and punctuation
    - Malayalam titles: ഡോ., അഡ്വ., പ്രൊഫ്., ലഫ്.
    - Malayalam initials: ആർ., പി., കെ., എം., etc.
    - Multi-initial names: പി.കെ., കെ.ടി., ആർ.കെ., etc.
    - English initials and abbreviations: Dr., Adv., Lt., A.K., R. Shankar
    - Line-separated fallback input
    """
    if not isinstance(text, str) or not text.strip():
        return []

    raw_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    normalized = re.sub(r"\s+", " ", text.strip())

    protected = _protect_abbreviations(normalized)
    parts = _split_protected_text(protected)

    sentences: List[str] = []

    for part in parts:
        restored = _restore_abbreviations(part)
        cleaned = clean_sentence(restored)

        if len(cleaned) > 2:
            sentences.append(cleaned)

    # Fallback for text that is mostly line-separated and has weak punctuation.
    if len(sentences) <= 1 and len(raw_lines) > 1:
        sentences = []

        for line in raw_lines:
            protected_line = _protect_abbreviations(line)
            line_parts = _split_protected_text(protected_line)

            for part in line_parts:
                restored = _restore_abbreviations(part)
                cleaned = clean_sentence(restored)

                if len(cleaned) > 2:
                    sentences.append(cleaned)

    return sentences


# -----------------------------------------------------------------------------
# Local test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "കൊച്ചി: സംസ്ഥാനത്തിലെ പൊതുഗതാഗത സേവനങ്ങളിൽ ഡിജിറ്റൽ ടിക്കറ്റിംഗ് സംവിധാനം വ്യാപിപ്പിക്കുന്നതിനുള്ള പുതിയ പദ്ധതി ഇന്ന് പ്രഖ്യാപിച്ചു. "
        "ഡോ. മീര നായർ പദ്ധതിയുടെ സാങ്കേതിക രൂപരേഖ യോഗത്തിൽ അവതരിപ്പിച്ചു. "
        "അഡ്വ. സുനിൽ ജോസഫ് യാത്രക്കാരുടെ ഡാറ്റാ സ്വകാര്യത ഉറപ്പാക്കേണ്ടതിന്റെ ആവശ്യകത ചൂണ്ടിക്കാട്ടി. "
        "പ്രൊഫ്. ആർ. ശങ്കർ ആർട്ടിഫിഷ്യൽ ഇന്റലിജൻസ് ഉപയോഗിച്ചുള്ള റൂട്ടു വിശകലനത്തെക്കുറിച്ച് വിശദീകരിച്ചു. "
        "പി.കെ. അനിൽ, കെ.ടി. രമേഷ്, എം.കെ. ജോൺ എന്നിവർ വിവിധ ജില്ലകളിലെ പരീക്ഷണാടിസ്ഥാനത്തിലുള്ള നടപ്പാക്കൽ വിവരങ്ങൾ പങ്കുവച്ചു. "
        "ലഫ്. ജനറൽ അരവിന്ദ് മേനോൻ സൈബർ സുരക്ഷാ നിരീക്ഷണ സംവിധാനം ശക്തമാക്കണമെന്ന് നിർദേശിച്ചു. "
        "ആദ്യഘട്ടത്തിൽ തിരുവനന്തപുരം, എറണാകുളം, കോഴിക്കോട് ജില്ലകളിലാണ് പദ്ധതി നടപ്പാക്കുക. "
        "യാത്രക്കാർക്ക് മൊബൈൽ ആപ്പ്, സ്മാർട്ട് കാർഡ്, ക്യൂആർ കോഡ് എന്നിവ വഴി ടിക്കറ്റ് എടുക്കാൻ കഴിയും. "
        "സ്വകാര്യ വിവരങ്ങൾ എൻക്രിപ്റ്റ് ചെയ്ത സെർവറുകളിൽ മാത്രം സൂക്ഷിക്കുമെന്ന് സാങ്കേതിക സംഘം വ്യക്തമാക്കി. "
        "പൈലറ്റ് പദ്ധതി വിജയകരമായാൽ ആറുമാസത്തിനകം മറ്റു ജില്ലകളിലേക്കും സേവനം വ്യാപിപ്പിക്കും."
    )

    for index, sentence in enumerate(segment_malayalam_text(sample), start=1):
        print(index, sentence)
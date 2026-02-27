from __future__ import annotations


_FLORES3_TO_ISO2 = {
    "eng": "en",
    "deu": "de",
    "fra": "fr",
    "rus": "ru",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "nld": "nl",
    "ces": "cs",
    "ukr": "uk",
    "jpn": "ja",
    "zho": "zh",
    "kor": "ko",
    "hin": "hi",
    "isl": "is",
    "swe": "sv",
    "pol": "pl",
    "tur": "tr",
    "ron": "ro",
    "arb": "ar",
    "ben": "bn",
    "vie": "vi",
    "ind": "id",
    "fil": "tl",
    "est": "et",
    "fin": "fi",
    "hun": "hu",
    "slk": "sk",
    "slv": "sl",
    "hrv": "hr",
    "srp": "sr",
    "cat": "ca",
    "dan": "da",
    "ell": "el",
    "heb": "he",
    "fas": "fa",
    "tha": "th",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "guj": "gu",
    "pan": "pa",
    "lit": "lt",
    "lav": "lv",
    "nor": "no",
    "swh": "sw",
    "urd": "ur",
    "zul": "zu",
}

_FLORES3_TO_NAME = {
    "eng": "English",
    "deu": "German",
    "fra": "French",
    "rus": "Russian",
    "spa": "Spanish",
    "ita": "Italian",
    "por": "Portuguese",
    "nld": "Dutch",
    "ces": "Czech",
    "ukr": "Ukrainian",
    "jpn": "Japanese",
    "zho": "Chinese",
    "kor": "Korean",
    "hin": "Hindi",
    "isl": "Icelandic",
    "swe": "Swedish",
    "pol": "Polish",
    "tur": "Turkish",
    "ron": "Romanian",
    "arb": "Arabic",
    "ben": "Bengali",
    "vie": "Vietnamese",
    "ind": "Indonesian",
    "fil": "Filipino",
    "est": "Estonian",
    "fin": "Finnish",
    "hun": "Hungarian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "hrv": "Croatian",
    "srp": "Serbian",
    "cat": "Catalan",
    "dan": "Danish",
    "ell": "Greek",
    "heb": "Hebrew",
    "fas": "Persian",
    "tha": "Thai",
    "tam": "Tamil",
    "tel": "Telugu",
    "kan": "Kannada",
    "mal": "Malayalam",
    "mar": "Marathi",
    "guj": "Gujarati",
    "pan": "Punjabi",
    "lit": "Lithuanian",
    "lav": "Latvian",
    "nor": "Norwegian",
    "swh": "Swahili",
    "urd": "Urdu",
    "zul": "Zulu",
}


def flores_to_iso2(lang_code: str) -> str:
    prefix = lang_code.split("_", maxsplit=1)[0]
    mapped = _FLORES3_TO_ISO2.get(prefix)
    if mapped is None:
        raise KeyError(
            f"No FLORES->ISO2 mapping for '{lang_code}'. "
            "Add it in active_mt_distill.datasets.language_codes._FLORES3_TO_ISO2."
        )
    return mapped


def language_name(lang_code: str) -> str:
    prefix = lang_code.split("_", maxsplit=1)[0]
    return _FLORES3_TO_NAME.get(prefix, lang_code)


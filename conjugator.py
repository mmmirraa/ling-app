# This code is a regular portuguese verb generator
# This will be utilized when the spaCy model is unable to provide information
# related to the given verb


# Master table for regular endings
# Limitation: So far this is only for verbs in the past-tense
REGULAR_ENDINGS = {
    "ar": {
        "PSTSimple": {
            "1SG": "ei",
            "2SG-Inf": "aste",
            "2SG-For": "ou",
            "3SG": "ou",
            "1PL": "amos",
            "2PL-Inf": "astes",
            "2PL-For": "aram",
            "3PL": "aram"
        },
        "PSTImperfect": {
            "1SG": "ava",
            "2SG-Inf": "avas",
            "2SG-For": "ava",
            "3SG": "ava",
            "1PL": "ávamos",
            "2PL-Inf": "áveis",
            "2PL-For": "avam",
            "3PL": "avam"
        },
            "PRSTInd": {
            "1SG": "o",
            "2SG-Inf": "as",
            "2SG-For": "a",
            "3SG": "a",
            "1PL": "amos",
            "2PL-Inf": "ais",
            "2PL-For": "am",
            "3PL": "am"
        },
            "PRSTSub": {
            "1SG": "e",
            "2SG-Inf": "es",
            "2SG-For": "e",
            "3SG": "e",
            "1PL": "emos",
            "2PL-Inf": "eis",
            "2PL-For": "em",
            "3PL": "em"
        }
        
    },

    "er": {
        "PSTSimple": {
            "1SG": "i",
            "2SG-Inf": "este",
            "2SG-For": "eu",
            "3SG": "eu",
            "1PL": "emos",
            "2PL-Inf": "estes",
            "2PL-For": "eram",
            "3PL": "eram"
        },
        "PSTImperfect": {
            "1SG": "ia",
            "2SG-Inf": "ias",
            "2SG-For": "ia",
            "3SG": "ia",
            "1PL": "íamos",
            "2PL-Inf": "íeis",
            "2PL-For": "iam",
            "3PL": "iam"
        },
            "PRSTInd": {
            "1SG": "o",
            "2SG-Inf": "es",
            "2SG-For": "e",
            "3SG": "e",
            "1PL": "emos",
            "2PL-Inf": "eis",
            "2PL-For": "em",
            "3PL": "em"
        },
            "PRSTSub": {
            "1SG": "a",
            "2SG-Inf": "as",
            "2SG-For": "a",
            "3SG": "a",
            "1PL": "amos",
            "2PL-Inf": "ais",
            "2PL-For": "am",
            "3PL": "am"
        }
    },

    "ir": {
        "PSTSimple": {
            "1SG": "i",
            "2SG-Inf": "iste",
            "2SG-For": "iu",
            "3SG": "iu",
            "1PL": "imos",
            "2PL-Inf": "istes",
            "2PL-For": "iram",
            "3PL": "iram"
        },
        "PSTImperfect": {
            "1SG": "ia",
            "2SG-Inf": "ias",
            "2SG-For": "ia",
            "3SG": "ia",
            "1PL": "íamos",
            "2PL-Inf": "íeis",
            "2PL-For": "iam",
            "3PL": "iam"
        },
            "PRSTInd": {
            "1SG": "o",
            "2SG-Inf": "es",
            "2SG-For": "e",
            "3SG": "e",
            "1PL": "imos",
            "2PL-Inf": "is",
            "2PL-For": "em",
            "3PL": "em"
        },
            "PRSTSub": {
            "1SG": "a",
            "2SG-Inf": "as",
            "2SG-For": "a",
            "3SG": "a",
            "1PL": "amos",
            "2PL-Inf": "ais",
            "2PL-For": "am",
            "3PL": "am"
        }
    }
}


# -----------------------------------------
# Generate full conjugation table
# -----------------------------------------
def generate_regular_conjugations(lemma: str):
    """Return dict of all regular forms for a verb ending in -ar/-er/-ir."""
    ending = lemma[-2:]
    if ending not in REGULAR_ENDINGS:
        return {"Infinitive": lemma}

    base = lemma[:-2]
    conj = {"Infinitive": lemma}

    for tense_name, persons in REGULAR_ENDINGS[ending].items():
        for person_code, suffix in persons.items():
            key = f"{person_code}-{tense_name}"
            conj[key] = base + suffix

    return conj


# Human-Friendly label mappings
PERSON_LABELS = {
    "1SG": "First Person Singular",
    "2SG-Inf": "Second Person Singular Informal",
    "2SG-For": "Second Person Singular Formal",
    "3SG": "Third Person Singular",
    "1PL": "First Person Plural",
    "2PL-Inf": "Second Person Plural Informal",
    "2PL-For": "Second Person Plural Formal",
    "3PL": "Third Person Plural",
}

TENSE_LABELS = {
    "PSTSimple": "Past-Tense Simple",
    "PSTImperfect": "Past-Tense Imperfect",
    "PRSTInd": "Present-Tense Indicative",
    "PRSTSub": "Present-Tense Subjunctive"
}

def pretty_label(code: str):
    if code == "Infinitive":
        return "Infinitive"
    person, tense = code.split("-")
    return f"{PERSON_LABELS.get(person, person)} {TENSE_LABELS.get(tense, tense)}"


# fallback for verbs identified by spaCy but morphology not given
def match_regular_conjugation(token: str, lemma: str):
    """
    Attempt to match a verb to its regular conjugation paradigm.
    Returns human-readable label or None.
    """
    conj = generate_regular_conjugations(lemma)

    for code, form in conj.items():
        if form == token:
            return pretty_label(code)

    return None

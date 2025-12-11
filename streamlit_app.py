# streamlit_app.py
import os
import re
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional

# --- spaCy loading (cached) ---
@st.cache_resource
def load_spacy_model(model_name: str = "pt_core_news_sm"):
    try:
        import spacy
        nlp = spacy.load(model_name)
        return nlp
    except Exception:
        # try to download model then load
        import spacy
        from spacy.cli import download as spacy_download
        spacy_download(model_name)
        nlp = spacy.load(model_name)
        return nlp

nlp = load_spacy_model()

# --- Human-readable mapping dictionaries ---
person_map = {"1": "First-Person", "2": "Second-Person", "3": "Third-Person"}
number_map = {"Sing": "Singular", "Plur": "Plural"}
tense_map = {
    "Pres": "Present-Tense",
    "Past": "Past-Tense",
    "Fut": "Future-Tense"
}
mood_map = {
    "Ind": "Indicative",
    "Imp": "Imperative",
    "Sub": "Subjunctive",
    "Cnd": "Conditional"
}
verbform_map = {
    "Fin": "Finite",
    "Inf": "Infinitive",
    "Part": "Participle",
    "Ger": "Gerund"
}

def analyze_sentence_spacy(sentence: str):
    doc = nlp(sentence)
    results = []

    for token in doc:
        if token.pos_ in ["VERB", "AUX"]:
            morph = token.morph
            person = person_map.get(morph.get("Person")[0], "") if morph.get("Person") else ""
            number = number_map.get(morph.get("Number")[0], "") if morph.get("Number") else ""
            tense = tense_map.get(morph.get("Tense")[0], "") if morph.get("Tense") else ""
            mood = mood_map.get(morph.get("Mood")[0], "") if morph.get("Mood") else ""
            verbform = verbform_map.get(morph.get("VerbForm")[0], "") if morph.get("VerbForm") else ""

            human_readable = " ".join(filter(None, [person, number, tense, mood, verbform]))

            results.append({
                "Verb": token.text,
                "Lemma": token.lemma_,
                "Conjugation": human_readable,
                "Person": person,
                "Number": number,
                "Tense": tense,
                "Mood": mood
            })
    return results

# --- Streamlit UI ---
sentence = st.text_input(
    "Enter a Portuguese sentence:",
    placeholder="A garota sabe como ele encontrou o anel dela."
)

if st.button("Analyze"):
    results = analyze_sentence_spacy(sentence)

    if not results:
        st.info("No verbs found in the sentence.")
    else:
        for r in results:
            st.markdown("---")
            st.markdown(f"### {r['Verb']} (Lemma: *{r['Lemma']}*)")
            st.write(f"**Conjugation:** {r['Conjugation'] if r['Conjugation'] else '(none)'}")

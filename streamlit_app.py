import streamlit as st
import spacy
from conjugator import match_regular_conjugation

# --- Load spaCy model ---
@st.cache_resource
def load_spacy_model():
    return spacy.load("pt_core_news_sm")

nlp = load_spacy_model()

# --- readable mapping dictionaries for spaCy ---
person_map = {"1": "First-Person", "2": "Second-Person", "3": "Third-Person"}
number_map = {"Sing": "Singular", "Plur": "Plural"}
tense_map = {"Pres": "Present-Tense", "Past": "Past-Tense", "Fut": "Future-Tense"}
mood_map = {"Ind": "Indicative", "Imp": "Imperative", "Sub": "Subjunctive", "Cnd": "Conditional"}
verbform_map = {"Fin": "Finite", "Inf": "Infinitive", "Part": "Participle", "Ger": "Gerund"}


# --- Analyze sentence ---
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

            # spaCy-generated label
            spacy_label = " ".join(filter(None, [person, number, tense, mood, verbform]))

            # fallback: regular conjugator if spaCy fails
            if not spacy_label.strip():
                fallback = match_regular_conjugation(token.text.lower(), token.lemma_)
            else:
                fallback = None

            readable = spacy_label if spacy_label else (fallback or "(unknown)")

            results.append({
                "Verb": token.text,
                "Lemma": token.lemma_,
                "Conjugation": readable
            })

    return results


# --- UI ---
st.title("Portuguese Verb Analyzer")

sentence = st.text_input("Enter a Portuguese sentence: (Hint: Try 'A garota sabe como ele encontrou o anel dela.')")

if st.button("Analyze"):
    results = analyze_sentence_spacy(sentence)

    if not results:
        st.info("No verbs found.")
    else:
        for r in results:
            st.markdown("---")
            st.markdown(f"### {r['Verb']} (Lemma: *{r['Lemma']}*)")
            st.write(f"**Conjugation:** {r['Conjugation']}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: #666;">
        <strong>Portuguese Verb Analyzer</strong><br>
        LING 430 — Final Project<br>
        Created by Maria Hernandez Carpio, Samaya Castro, and Mirfat Maani
        Built using <strong>spaCy</strong> and the <em>pt_core_news_sm</em> language model.
    </div>
    """,
    unsafe_allow_html=True
)

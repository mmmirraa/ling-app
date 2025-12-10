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

# --- Paths ---
ROOT_DIR = os.path.dirname(__file__) or "."
DATADIR = os.path.join(ROOT_DIR, "data")

# --- Utilities: Read transcripts and clean dialogue (cached) ---
@st.cache_data
def read_bfamdl_files(base_path="bfamdl", prefix="bfamdl", start=1, end=14) -> str:
    """
    Reads bfamdl transcript files from data/bfamdl/bfamdl01.cha ... bfamdl14.cha
    Returns combined text (string). If files not found, returns empty string.
    """
    combined_text = ""
    for i in range(start, end + 1):
        file_name = f"{base_path}/{prefix}{i:02d}.cha"
        full_path = os.path.join(DATADIR, file_name)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                combined_text += f.read() + "\n"
    return combined_text

@st.cache_data
def dialogue_cleaner_from_text(text: str) -> str:
    """
    Extracts lines beginning with *PAR and cleans them.
    Input can be a big string (all transcripts).
    """
    if not text:
        return ""
    lines = text.splitlines()
    dialogue = [line for line in lines if line.startswith("*PAR")]
    cleaned_lines = []
    for line in dialogue:
        cleaned = re.sub(r"^(\*PAR\d:)", "", line)
        cleaned = re.sub(r"\x15\d+_\d+\x15", "", cleaned)
        cleaned = re.sub(r"\n", "", cleaned)
        cleaned = cleaned.strip().lower()
        cleaned_lines.append(cleaned)
    return " ".join(cleaned_lines)

# --- Verb endings (regular) ---
def get_ar_endings(verb: str) -> Dict[str, str]:
    short = verb[:-2]
    return {
        "Infinitive": verb,
        "1SG-PSTSimple": short + "ei",
        "2SG-InfPSTSimple": short + "aste",
        "2SG-ForPSTSimple": short + "ou",
        "3SG-PSTSimple": short + "ou",
        "1PL-PSTSimple": short + "amos",
        "2PL-InfPSTSimple": short + "astes",
        "2PL-ForPSTSimple": short + "aram",
        "3PL-PSTSimple": short + "aram",
        "1SG-PSTImperfect": short + "ava",
        "2SG-InfPSTImperfect": short + "avas",
        "2SG-ForPSTImperfect": short + "ava",
        "3SG-PSTImperfect": short + "ava",
        "1PL-PSTImperfect": short + "ávamos",
        "2PL-InfPSTImperfect": short + "áveis",
        "2PL-ForPSTImperfect": short + "avam",
        "3PL-PSTImperfect": short + "avam",
    }

def get_er_endings(verb: str) -> Dict[str, str]:
    short = verb[:-2]
    return {
        "Infinitive": verb,
        "1SG-PSTSimple": short + "i",
        "2SG-InfPSTSimple": short + "este",
        "2SG-ForPSTSimple": short + "eu",
        "3SG-PSTSimple": short + "eu",
        "1PL-PSTSimple": short + "emos",
        "2PL-InfPSTSimple": short + "estes",
        "2PL-ForPSTSimple": short + "eram",
        "3PL-PSTSimple": short + "eram",
        "1SG-PSTImperfect": short + "ia",
        "2SG-InfPSTImperfect": short + "ias",
        "2SG-ForPSTImperfect": short + "ia",
        "3SG-PSTImperfect": short + "ia",
        "1PL-PSTImperfect": short + "íamos",
        "2PL-InfPSTImperfect": short + "íeis",
        "2PL-ForPSTImperfect": short + "iam",
        "3PL-PSTImperfect": short + "iam",
    }

def get_ir_endings(verb: str) -> Dict[str, str]:
    short = verb[:-2]
    return {
        "Infinitive": verb,
        "1SG-PSTSimple": short + "i",
        "2SG-InfPSTSimple": short + "iste",
        "2SG-ForPSTSimple": short + "iu",
        "3SG-PSTSimple": short + "iu",
        "1PL-PSTSimple": short + "imos",
        "2PL-InfPSTSimple": short + "istes",
        "2PL-ForPSTSimple": short + "iram",
        "3PL-PSTSimple": short + "iram",
        "1SG-PSTImperfect": short + "ia",
        "2SG-InfPSTImperfect": short + "ias",
        "2SG-ForPSTImperfect": short + "ia",
        "3SG-PSTImperfect": short + "ia",
        "1PL-PSTImperfect": short + "íamos",
        "2PL-InfPSTImperfect": short + "íeis",
        "2PL-ForPSTImperfect": short + "iam",
        "3PL-PSTImperfect": short + "iam",
    }

# --- Build verb_totals (cached) ---
@st.cache_data
def build_verb_totals(top_n: int = 58) -> pd.DataFrame:
    # Load verbs.csv if present
    verbs_csv = os.path.join(DATADIR, "verbs.csv")
    if os.path.exists(verbs_csv):
        verb_freq = pd.read_csv(verbs_csv)
        top_verbs = verb_freq["Lemma"].head(top_n).to_list()
    else:
        # fallback: a small set if verbs.csv missing
        top_verbs = [
            "falar", "comer", "abrir", "ser", "estar", "ter", "ir", "fazer", "dar",
            "dizer", "querer", "saber", "ver", "vir", "pôr"
        ]

    # build regular endings dataframe
    columns = [
        "Infinitive",
        "1SG-PSTSimple",
        "2SG-InfPSTSimple",
        "2SG-ForPSTSimple",
        "3SG-PSTSimple",
        "1PL-PSTSimple",
        "2PL-InfPSTSimple",
        "2PL-ForPSTSimple",
        "3PL-PSTSimple",
        "1SG-PSTImperfect",
        "2SG-InfPSTImperfect",
        "2SG-ForPSTImperfect",
        "3SG-PSTImperfect",
        "1PL-PSTImperfect",
        "2PL-InfPSTImperfect",
        "2PL-ForPSTImperfect",
        "3PL-PSTImperfect",
    ]
    verb_endings = pd.DataFrame(columns=columns)

    irregular_list = [
        "ser", "estar", "ter", "poder", "ir", "fazer", "dar", "dizer",
        "querer", "saber", "ver", "vir", "pôr"
    ]

    for v in top_verbs:
        if v in irregular_list:
            # add row with only infinitive
            verb_endings.loc[len(verb_endings)] = {"Infinitive": v}
        else:
            if v.endswith("ar"):
                verb_endings.loc[len(verb_endings)] = get_ar_endings(v)
            elif v.endswith("er"):
                verb_endings.loc[len(verb_endings)] = get_er_endings(v)
            elif v.endswith("ir"):
                verb_endings.loc[len(verb_endings)] = get_ir_endings(v)
            else:
                verb_endings.loc[len(verb_endings)] = {"Infinitive": v}

    # Create irregular DataFrame from list (so concat later is easy)
    irr_df = pd.DataFrame({"Infinitive": irregular_list})

    # Combine: keep regular rows (rows with conjugations) + our irregulars
    conj_cols = verb_endings.columns[1:]
    regular_verbs = verb_endings[verb_endings[conj_cols].notna().all(axis=1)]
    verb_totals = pd.concat([regular_verbs, irr_df], ignore_index=True, sort=False)
    return verb_totals

verb_totals = build_verb_totals()

# --- Build df_verbs from corpus (cached) ---
def classify_verb_local(lemma: str, irregulars: Optional[List[str]] = None) -> str:
    if irregulars is None:
        irregulars = ["ser", "estar", "ter", "poder", "ir", "fazer", "dar", "dizer",
                      "querer", "saber", "ver", "vir", "pôr"]
    if lemma in irregulars:
        return "irregular"
    if lemma.endswith(("ar", "er", "ir")):
        return "regular"
    return "unknown"

def get_past_tense_local(text: str) -> pd.DataFrame:
    doc = nlp(text)
    results = []
    for token in doc:
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            lemma = token.lemma_
            classification = classify_verb_local(lemma)
            results.append({
                "token": token.text,
                "lemma": lemma,
                "classification": classification,
                "pos": token.pos_
            })
    return pd.DataFrame(results)

def find_conjugation_local(token: str, lemma: str, verb_totals_df: pd.DataFrame):
    rows = verb_totals_df[verb_totals_df["Infinitive"] == lemma]
    results = []
    for idx, row in rows.iterrows():
        cols = [col for col in verb_totals_df.columns if row.get(col) == token]
        if cols:
            results.append({"row_index": idx, "columns": cols})
    return results if results else None

def annotate_conjugations_local(df_verbs_df: pd.DataFrame, verb_totals_df: pd.DataFrame) -> pd.DataFrame:
    all_conjugations = []
    for _, row in df_verbs_df.iterrows():
        token = row["token"]
        lemma = row["lemma"]
        conjugations = find_conjugation_local(token, lemma, verb_totals_df)
        all_conjugations.append(conjugations)
    df_verbs_df["conjugations"] = all_conjugations
    return df_verbs_df

@st.cache_data
def build_df_verbs_from_corpus():
    raw = read_bfamdl_files()
    cleaned = dialogue_cleaner_from_text(raw)
    if not cleaned:
        return pd.DataFrame(columns=["token", "lemma", "classification", "pos", "conjugations"])
    df = get_past_tense_local(cleaned)
    df = annotate_conjugations_local(df, verb_totals)
    return df

# Build df_verbs lazily (first UI action triggers building/caching)
df_verbs = None

# --- SPA UI ---
st.set_page_config(page_title="Portuguese Verb Analyzer", layout="centered")
st.title("Portuguese Verb Analyzer")
st.write("Type a Portuguese sentence and get verb analysis (lemma, tense/person/number).")

with st.sidebar:
    st.write("Data and settings")
    st.write(f"Data directory: `{DATADIR}`")
    if st.button("(Re)build cached corpus data"):
        # force rebuild caches by calling the cached functions explicitly
        _ = build_verb_totals()
        _ = read_bfamdl_files()  # reads files and caches
        _ = dialogue_cleaner_from_text(read_bfamdl_files())
        _ = build_df_verbs_from_corpus()
        st.success("Caches refreshed (if files were present).")
def analyze_sentence_spacy(sentence):
    doc = nlp(sentence)
    results = []

    for token in doc:
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            morph = token.morph
            results.append({
                "token": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tense": morph.get("Tense"),
                "person": morph.get("Person"),
                "number": morph.get("Number"),
                "mood": morph.get("Mood"),
                "verbform": morph.get("VerbForm")
            })

    return results

sentence = st.text_input("Enter a Portuguese sentence:")

if st.button("Analyze"):
    results = analyze_sentence_spacy(sentence)

    if not results:
        st.info("No verbs found.")
    else:
        for r in results:
            st.markdown("---")
            st.markdown(f"### {r['token']} (lemma: *{r['lemma']}*)")
            
            st.write("**spaCy Morphology:**")
            st.write(f"- Tense: {', '.join(r['tense']) if r['tense'] else '(none)'}")
            st.write(f"- Person: {', '.join(r['person']) if r['person'] else '(none)'}")
            st.write(f"- Number: {', '.join(r['number']) if r['number'] else '(none)'}")
            st.write(f"- Mood: {', '.join(r['mood']) if r['mood'] else '(none)'}")
            st.write(f"- VerbForm: {', '.join(r['verbform']) if r['verbform'] else '(none)'}")


    if not results:
        st.info("No verbs found in the sentence (or sentence empty).")
    else:
        for r in results:
            st.markdown("---")
            st.markdown(f"**{r['token']}** — lemma: *{r['lemma']}*")
            st.write(f"- Classification: `{r['classification']}`")
            if r["conjugation_annotation"]:
                st.write("- Conjugation (dictionary matches):")
                for entry in r["conjugation_annotation"]:
                    for c in entry["columns"]:
                        st.write(f"  - {c}")
            else:
                st.write("- Conjugation (dictionary matches): (none)")

            st.write("- spaCy morphological features:")
            for k, v in r["morphology"].items():
                st.write(f"  - {k}: {', '.join(v) if v else '(none)'}")

# Show helpful debug info at bottom
st.markdown("---")
st.write("Verb dictionary sample (first 10 rows):")
st.dataframe(verb_totals.head(10))
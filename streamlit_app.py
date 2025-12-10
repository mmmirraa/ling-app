import streamlit as st
import spacy

from turninversionling430project import interactive_piece, df_verbs

# ----------------------------------------------------------
# LOAD SPACY MODEL
# ----------------------------------------------------------
nlp = spacy.load("pt_core_news_sm")

# ----------------------------------------------------------
# PAGE SETUP
# ----------------------------------------------------------
st.set_page_config(page_title="Portuguese Verb Analyzer", layout="centered")

st.title("Portuguese Verb Analyzer")
st.write("Enter a sentence in Portuguese and the system will analyze its verbs using your past-tense dictionary + spaCy morphology.")

# ----------------------------------------------------------
# USER INPUT
# ----------------------------------------------------------
sentence = st.text_input("Digite uma frase:", placeholder="Ex: Nós falamos ontem.")

# ----------------------------------------------------------
# PROCESS INPUT
# ----------------------------------------------------------
if sentence.strip() != "":
    st.subheader("Results")

    results = interactive_piece(df_verbs, sentence)

    if not results:
        st.write("No verbs found in this sentence.")
    else:
        # Display structured output
        for r in results:
            st.markdown("---")
            st.markdown(f"### **{r['token']}**")
            st.write(f"**Lemma:** {r['lemma']}")
            st.write(f"**Classification:** {r['classification']}")

            # Verb conjugation (from your manually-built dictionary)
            if r["conjugation_annotation"] is None:
                st.write("**Conjugation (dictionary):** Not found")
            else:
                st.write("**Conjugation (dictionary):**")
                for c in r["conjugation_annotation"]:
                    st.write(f"- {c}")

            # spaCy morphological info
            st.write("**spaCy Morphology:**")
            for key, val in r["spaCy_morphology"].items():
                if val:
                    st.write(f"- {key.capitalize()}: {', '.join(val)}")
                else:
                    st.write(f"- {key.capitalize()}: (none)")

else:
    st.info("Type a sentence above to see verb analysis.")
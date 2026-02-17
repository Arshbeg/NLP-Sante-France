import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page conf
st.set_page_config(
    page_title="DrBERT Clinical AI",
    page_icon="🩺",
    layout="wide"
)

# load and cache
@st.cache_resource
def load_model():
    model_path = "./final_french_ner_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_pipeline

nlp = load_model()

# hard code cim-10 code (logic), if we have the database like eds in hospitals like CHU we can automate it and no need to hardcode it
@st.cache_resource
def build_cim10_ranker():
    cim10_data = {
        "Code": ["I10", "E11.9", "R06.0", "R07.4", "I48.9", "R51", "G03.9"],
        "Description": [
            "Hypertension artérielle essentielle",
            "Diabète sucré de type 2 sans complication",
            "Dyspnée",
            "Douleur thoracique, sans précision",
            "Fibrillation auriculaire non spécifiée",
            "Céphalée",
            "Méningite, non spécifiée"
        ]
    }
    df = pd.DataFrame(cim10_data)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(df['Description'])
    return df, vectorizer, tfidf_matrix

df_cim10, vectorizer, tfidf_matrix = build_cim10_ranker()

def get_cim10_code(term):
    vec = vectorizer.transform([term])
    sims = cosine_similarity(vec, tfidf_matrix)
    best_idx = sims.argmax()
    score = sims[0, best_idx]
    if score > 0.25:
        return df_cim10.iloc[best_idx]['Code'], df_cim10.iloc[best_idx]['Description'], score
    return "N/A", "No Match", 0.0

# rgpd scrubber to hide the identtities/personal details(AI + Regex)
def scrub_text(text, entities):
    scrubbed = text

    name_patterns = [
        r"(s'appelle\s+)([A-Z][a-z]+)", 
        r"(nommé\s+)([A-Z][a-z]+)",
        r"(M\.\s+)([A-Z][a-z]+)", 
        r"(Mme\s+)([A-Z][a-z]+)"
    ]
    for pattern in name_patterns:
        scrubbed = re.sub(pattern, r"\1[NOM]", scrubbed)
    sorted_ents = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    for ent in sorted_ents:
        word = ent['word'].strip()
        
        if ent['entity_group'] == 'LABEL_4': # LIVB (Demographics)
            scrubbed = scrubbed.replace(word, "[PATIENT]")
            
        elif ent['entity_group'] == 'LABEL_8': # GEOG (Locations)
            scrubbed = scrubbed.replace(word, "[HÔPITAL]")
            
    # Clean up any weird double spaces created by the replacements
    scrubbed = re.sub(r'\s+', ' ', scrubbed).strip()
    
    return scrubbed

# UI
st.title("🇫🇷 DrBERT: Clinical Entity Extraction & Coding")
st.markdown("""
**Model:** Fine-Tuned DrBERT (7GB) | **Task:** NER + CIM-10 Coding + RGPD Scrubbing  
*Enter a French clinical note below to extract diseases, drugs, and procedures.*
""")

# Sidebar
st.sidebar.header("Options")
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.50)

# Input Area
default_text = "Un patient de 68 ans avec des antécédents d'hypertension artérielle est admis aux urgences de l'Hôpital Pitié-Salpêtrière pour une dyspnée sévère et des douleurs thoraciques. Prescription de 1g de Paracétamol."
text_input = st.text_area("Clinical Note", default_text, height=150)

if st.button("Analyze Clinical Note 🚀"):
    with st.spinner("Running AI Inference..."):
        # Run Inference
        entities = nlp(text_input)
        
        # Filter by threshold
        valid_entities = [e for e in entities if e['score'] >= threshold]
    
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Extracted Entities")
            # Highlighted Text Logic could go here, but we'll list them first
            for ent in valid_entities:
                label = ent['entity_group']
                word = ent['word']
                
                # Color Coding
                color = "grey"
                if label == "LABEL_1": color = "#ff4b4b" # DISO (Red)
                if label == "LABEL_5": color = "#09ab3b" # CHEM (Green)
                if label == "LABEL_2": color = "#ffbd45" # PROC (Orange)
                
                # CIM-10 Lookup for Diseases
                cim_info = ""
                if label == "LABEL_1": # Only lookup diseases
                    code, desc, conf = get_cim10_code(word)
                    if code != "N/A":
                        cim_info = f"➡️ **CIM-10: {code}**"

                st.markdown(f"""
                <div style="padding:10px; border-radius:5px; background-color:f0f2f6; margin-bottom:5px; border-left: 5px solid {color}">
                    <span style="font-weight:bold; font-size:1.1em">{word}</span> 
                    <span style="font-size:0.9em; color:#555">({label})</span>
                    <br>{cim_info}
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.subheader("🔒 RGPD Compliance")
            safe_text = scrub_text(text_input, valid_entities)
            st.text_area("Anonymized Output", safe_text, height=150, disabled=True)
            st.success("✅ Patient Data & Locations Scrubbed")

st.markdown("---")
st.caption("Built with DrBERT & Streamlit for the French Healthcare System.")
##  NLP Sante France (Clinical-Insight-Pipeline)

**Automated French Medical NER & Normalization System**

A professional NLP pipeline for extracting clinical entities from unstructured French medical text, mapping them to international standards **(UMLS/CIM-10)**, and ensuring **RGPD (GDPR)** compliance through automated de-identification.

---

### Project Overview
This project is an attempt to help doctors save time on administration coding the French health sector. This project demonstrates an end-to-end solution to:

* **Extract:** Identify critical medical entities (Disorders, Procedures, Drugs) using a fine-tuned DrBERT model.

* **Normalize:** Map colloquial French clinical terms to standardized codes (CIM-10/UMLS).

* **Protect:** Automatically mask patient identifiers to meet CNIL and RGPD requirements.

### Tech Stack
* **Language:** Python 3.11+

* **Models:** DrBERT(Sovereign French Biomedical LLM)

* **Dataset:** QUAERO French Medical Corpus (Gold Standard)

Libraries: transformers, spacy, pandas, seqeval (for clinical metrics)

### Key Features

### Performance

### Demo

### How to use
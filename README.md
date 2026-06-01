# Medical-term-explainer
Final project for UTSA NLP class

# Medical Term Simplification System (RAG Pipeline)

This repository contains a **Retrieval-Augmented Generation (RAG)** pipeline designed to bridge the health literacy gap by translating professional, college-level medical jargon into clear, accessible language for pediatric audiences (6th-grade target). 

By grounding an LLM in verified clinical research from the **UMLS (Unified Medical Language System)** and **PubMed**, the system ensures high semantic fidelity while dramatically lowering reading complexity.

---

##  Key Features

* **Intelligent Query Parsing:** Uses an LLM-based pre-processor to clean user prompts and extract standardized medical keywords.
* **Hybrid Retrieval Architecture:** Combines **Dense Retrieval** (PubMedBERT embeddings via a FAISS index) for deep semantic matching with **Sparse Retrieval** (BM25) for exact terminology alignment.
* **Syllable-Density Prompting:** Implements structural linguistic constraints rather than artificial sentence limits, forcing the generator to swap multi-syllable jargon for simple "everyday" words.
* **Comprehensive Evaluation Suite:** Automated testing tracking readability metrics (Flesch-Kincaid Grade Level) and semantic overlap (ROUGE-L) against human-curated **MedlinePlus** gold standards.

---

## Evaluation & Performance Results

Evaluated against professional baselines using a controlled "Oracle/Gold Retrieval" framework to isolate text simplification capabilities:

| Method / Pipeline Phase | Flesch-Kincaid Grade Level <br>*(Lower = Better)* | ROUGE-L Score <br>*(Semantic Overlap)* | Target Audience Accessibility |
| :--- | :---: | :---: | :--- |
| **Baseline (Raw Technical Text)** | 12.63 | 0.0492 | College Sophomore |
| **GPT-5 (Zero-Shot)** | 10.71 | **0.1454** | High School Freshman |
| **GPT-5 + RAG Pipeline (Ours)** | **6.27** | 0.1237 | **6th Grader (Target Met)** |

### Key Takeaways
* **The Readability Win:** The RAG pipeline stripped out heavy multi-syllable clinical phrasing, successfully dropping the reading comprehension level by **over 6 grade levels**.
* **Semantic Fidelity:** Despite using vastly simpler vocabulary, the RAG system maintained competitive semantic alignment (`0.1237` ROUGE-L) with expert human simplifications, avoiding hallucinations.

---

## 🛠️ Tech Stack & Architecture

* **LLM Core:** Azure OpenAI (GPT-5)
* **Vector Database & Search:** FAISS (Facebook AI Similarity Search) + BM25
* **Embeddings:** PubMedBERT (Domain-specific clinical embeddings)
* **Scientific Alignment:** SciPy (`spatial.distance`) for medical diagnostic code matching
* **Metrics & Testing:** `textstat` (FKGL calculation), `rouge-score`

---

from data import MedTable
from embeddings_processing import dense_retrieve
from qa_agent.generated_answer import generated_answer, extract_medical_keywords

# 1. Load data ONCE globally
MED_TABLE = MedTable.load()

def run_pipeline(query, history=[]):
    keywords = extract_medical_keywords(query)
    
    all_extracted_terms = []
    all_source_texts = []
    
    for kw in keywords:
        results = dense_retrieve(kw, k=1)
        top_cui, score = results[0]
        
        if score >= 0.75:
            entity = MED_TABLE.entities.get(top_cui)
            if entity and entity.mentions:

                best_paper = None
                for mention in entity.mentions[:10]:
                    temp_paper = MED_TABLE.papers[mention.pmid]
                    # Check if paper text contains words from original query
                    if any(word in temp_paper.text.lower() for word in query.lower().split()):
                        best_paper = temp_paper
                        break
                
                # Fallback to first mention if no perfect match
                if not best_paper:
                    best_paper = MED_TABLE.papers[entity.mentions[0].pmid]

                all_source_texts.append(best_paper.text)
                
                for m in best_paper.mentions:
                    ent = MED_TABLE.entities.get(m.cui)
                    if ent and ent.definitions:
                        all_extracted_terms.append({
                            "term": m.synonym,
                            "def": ent.definitions[0].definition
                        })

    # 5. Handle "External Knowledge" path
    source_text = "\n\n---\n\n".join(set(all_source_texts)) if all_source_texts else "EXTERNAL_KNOWLEDGE"
    
    return generated_answer(all_extracted_terms, source_text, query, history=history)

if __name__ == "__main__":
    test_query = "heart attack and blood clots"
    print(f"Running pipeline for: {test_query}")
    answer = run_pipeline(test_query)
    print(f"\n--- AI Answer ---\n{answer}")
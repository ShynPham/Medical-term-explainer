import warnings
import streamlit as st
from data import MedTable
from embeddings_processing import dense_retrieve
from qa_agent.generated_answer import generated_answer, extract_medical_keywords

# 1. Setup & Configuration
warnings.filterwarnings("ignore", message=".*Accessing `__path__`.*")

st.set_page_config(page_title="Medical Term Explainer", page_icon="🩺", layout="centered")
st.title("Medical Term Explainer")

@st.cache_resource
def load_and_map_data():
    """Loads data and builds a map of ONLY the papers we actually have."""
    table = MedTable.load()
    
    # Make a map of cui to contain papers we actually have in our local dataset
    local_map = {}
    for pmid, paper in table.papers.items():
        for mention in paper.mentions:
            if mention.cui not in local_map:
                local_map[mention.cui] = []
            if pmid not in local_map[mention.cui]:
                local_map[mention.cui].append(pmid)
    
    return table, local_map

# Load everything once
med_table, local_mention_map = load_and_map_data()

# Start the Streamlit app
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_context" not in st.session_state:
    st.session_state.last_context = {"terms": [], "paper": "", "ids": "N/A"}

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat UI
if query := st.chat_input("EX: Antisigma, Lidocaine, etc..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("One moment..."):
            all_extracted_terms = []
            all_source_texts = []
            all_source_ids = []
            
            # Check for follow-up 
            follow_up_keywords = ["yes", "no", "more", "go ahead", "continue", "sure", "yeah"]
            is_follow_up = any(word in query.lower() for word in follow_up_keywords)

            if is_follow_up and st.session_state.last_context["terms"]:
                extracted_terms = st.session_state.last_context["terms"]
                source_text = st.session_state.last_context["paper"]
                source_ids = st.session_state.last_context["ids"]
            else:
                keywords = extract_medical_keywords(query) 
                
                for kw in keywords:
                    # look up more papers for this term, but only from the local map to avoid "vector noise"
                    results = dense_retrieve(kw, k=20)
                    
                    found_pmid = None
                    for cui, score in results:
                        if score < 0.60: continue
                        
                        if cui in local_mention_map:
                            available_pmids = local_mention_map[cui]
                            
                            # Use the Re-ranker ONLY on papers we know exist
                            candidates = []
                            for p_id in available_pmids[:5]: 
                                paper_obj = med_table.papers[p_id]
                                candidates.append({"pmid": p_id, "text": paper_obj.text[:500]})
                            
                            # Ask the AI extra step to pick from the REAL candidates
                            from qa_agent.generated_answer import client, DEPLOYMENT
                            ranking_prompt = f"Query: {query}\nCandidates: {candidates}\nReturn ONLY the best PMID or 'NONE'."
                            
                            rank_res = client.chat.completions.create(
                                model=DEPLOYMENT,
                                messages=[{"role": "user", "content": ranking_prompt}],
                                max_tokens=10, temperature=0
                            )
                            
                            ans = rank_res.choices[0].message.content.strip()
                            best_pmid = "".join(filter(str.isdigit, ans))
                            
                            # Fallback, if AI is confused or it doesn't exist
                            found_pmid = best_pmid if best_pmid in med_table.papers else available_pmids[0]
                            break 
                    
                    if found_pmid:
                        paper = med_table.papers[found_pmid]
                        all_source_texts.append(paper.text)
                        all_source_ids.append(str(found_pmid))
                        # Add definitions for the Word Bank
                        for m in paper.mentions:
                            ent = med_table.entities.get(m.cui)
                            if ent and ent.definitions:
                                all_extracted_terms.append({"term": m.synonym, "def": ent.definitions[0].definition})

                # Combine results
                source_text = "\n\n---\n\n".join(list(dict.fromkeys(all_source_texts))) if all_source_texts else "EXTERNAL_KNOWLEDGE"
                source_ids = ", ".join(all_source_ids) if all_source_ids else "N/A"
                extracted_terms = all_extracted_terms
                st.session_state.last_context = {"terms": extracted_terms, "paper": source_text, "ids": source_ids}

            # DEBUG 
            with st.sidebar:
                st.write("### Debug Info")
                st.write(f"Keywords: {keywords}")
                st.write(f"Papers Found: {all_source_ids}")
                st.write(f"Threshold score: {score:.2f}")

            answer = generated_answer(extracted_terms, source_text, query, history=st.session_state.messages)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            if source_text != "EXTERNAL_KNOWLEDGE" and all_source_ids:
                with st.expander("View Scientific Source(s)"):
                    for pmid in all_source_ids:
                        # Show the markdown link to PubMed and a snippet of the paper text
                        st.markdown(f"🔗 **PMID: [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)**")
                    st.divider()
                    st.write(source_text[:500])
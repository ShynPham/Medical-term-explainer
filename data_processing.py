import spacy
from scispacy.abbreviation import AbbreviationDetector
from collections import defaultdict
import pickle
from data import Entity, MedTable, Paper, EntityDefinition, EntityMention
import re

tuis = set()
with open("tuis.txt", "r", encoding="utf-8") as f:
    for line in f:
        fields = line.split(",")
        tuis.add((fields[0], fields[1]))
med_table = MedTable(tuis=frozenset(tuis))

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("abbreviation_detector")
nlp.add_pipe("sentencizer", before="parser", config={"punct_chars": ["\n"]})

rea_1 = re.compile(r"\[((Source|[A-Z_]+):[^\]]+)?\]") # Annotation patterns
rea_2 = re.compile(r"(\(\"?http[^\)]+\))|(\[\"?http[^\]]+\])") # URL patterns
rea_3 = re.compile(r"<\/?[a-z0-9\-]+ ?([a-z0-9\-]*=\"[^\"]*\" ?)*>") # HTML tag patterns
print("Parsing UMLS definitions...")
# --- Parse MRDEF ---
with open("MRDEF.RRF", "r", encoding="utf-8") as f:
    for line in f:
        fields = line.strip().rstrip("|").split("|")
        cui = fields[0]
        sab = fields[4]
        assert sab in med_table.sabs
        defn = fields[5]
        cui_entity = med_table.entities[cui]
        cui_defs = cui_entity.definitions
        print(f"Processing CUI [{cui}], definition #{len(cui_defs)}", end="\r", flush=True)
        if not [d for d in cui_defs if d.sab == sab]: # Only keep one definition per SAB for each CUI
            if not cui_defs:
                med_table.def_cui_count += 1
            # Remove added annotations like [Source:...], [UMLS_CUI:...], (http://...), HTML tags
            defn = rea_1.sub("", defn)
            defn = rea_2.sub("", defn)
            defn = rea_3.sub("", defn)
            defn = defn.strip()

            sentences = []
            doc_sents = list(nlp(defn).sents)
            for sent in doc_sents:
                sentences.append((sent.start_char, sent.end_char))
            cui_def = EntityDefinition(sab, defn, sentences)
            cui_defs.append(cui_def)
print("\nFinished parsing UMLS entity definitions.")
# --- Parse MedMentions ---

print("\nParsing entity mentions...")
with open("corpus_pubtator.txt", "r", encoding="utf-8") as f:
    pmid = None
    title = None
    abstract = None
    for line in f:
        line = line.rstrip("\n")

        if not line:
            continue
        elif "|t|" in line:
            pmid, _, title = line.partition("|t|")
            pmid = int(pmid)
            title += "\n"
        elif "|a|" in line:
            assert isinstance(title, str)
            _, _, abstract = line.partition("|a|")
            title_len = len(title)
            text = title + abstract
            print(f"Processing PMID [{pmid}], Paper: {title[:20] if len(title) > 20 else title[:-1]}...", end="\r", flush=True)
            doc_sents = list(nlp(text).sents)
            sentences = []
            for sent in doc_sents:
                sentences.append((sent.start_char, sent.end_char))
            assert isinstance(pmid, int)
            med_table.papers[pmid] = Paper(text, title_len, sentences)
        elif "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 6:
                assert int(parts[0]) == pmid
                assert isinstance(pmid, int)
                start, end = int(parts[1]), int(parts[2])
                synonym = parts[3]
                print(f"Processing PMID [{pmid}], Mention \"{synonym}\"", end="\r", flush=True)
                tui_types = frozenset(parts[4].split(","))
                cui = parts[5]
                mention = EntityMention(pmid, (start, end), synonym, tui_types, cui)
                med_table.papers[pmid].mentions.append(mention)
                med_table.mention_count += 1
                entity_mentions = med_table.entities[cui].mentions
                if not entity_mentions:
                    med_table.mention_cui_count += 1
                entity_mentions.append(mention)

cuis_with_both = [cui for cui, ent in med_table.entities.items() if ent.definitions and ent.mentions]
med_table.both_cui_count = len(cuis_with_both)

print(f"Total CUIs in MRDEF: {med_table.def_cui_count}")
print(f"Total entity mentions in MedMentions: {med_table.mention_cui_count}")
print(f"Total CUIs in MedMentions: {med_table.mention_cui_count}")
print(f"CUIs with both definition + mention: {med_table.both_cui_count}")

count = 1

for x in cuis_with_both[:5]:  # print a few examples
    print(f"CUI: {x}")
    entity = med_table.entities[x]
    print("Definitions:")
    for defn in entity.definitions:
        print(f"  - [{defn.sab}] {defn.definition}")
    print("Mentions:")
    for mention in entity.mentions[:2]:  # show a couple mentions
        print(f"  - Keyword: {mention.synonym}")
        print(f"    Context: {med_table.papers[mention.pmid].get_context(mention.range[0], mention.range[1])}")  # show sentence it appears in
        print(f"    Sem Types: {mention.tuis}")
    print()

med_table.save()
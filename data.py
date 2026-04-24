import pickle
import numpy as np
import numpy.typing as npt
import re
from dataclasses import dataclass, field
from collections import defaultdict
from typing import ClassVar

def empty_list():
    return []
@dataclass(slots=True)
class Entity:
    definitions: list["EntityDefinition"] = field(default_factory=empty_list)
    mentions: list["EntityMention"] = field(default_factory=empty_list)
def default_entities():
    return defaultdict(Entity)
def empty_dict():
    return {}
@dataclass(slots=True)
class MedTable:
    FILE_NAME: ClassVar[str] = "CUI.pkl"
    tuis: frozenset[tuple[str, str]]
    papers: dict[int, "Paper"] = field(default_factory=empty_dict)# Maps PMID
    entities: defaultdict[str,"Entity"] = field(default_factory=default_entities)
    sabs: frozenset[str] = frozenset({'MCM', 'NCI', 'CHV', 'HPO', 'SPN', 'PDQ', 'MSH', 'HL7V3.0', 'AOT', 'MEDLINEPLUS', 'AIR', 'FMA', 'UWDA', 'CSP', 'LNC', 'GO'})
    def_cui_count: int = field(default=0)
    mention_cui_count: int = field(default=0)
    both_cui_count: int = field(default=0)
    mention_count: int = field(default=0)
    def save(self):
        with open(self.__class__.FILE_NAME, "wb") as f:
            pickle.dump(self, f)
    @classmethod
    def load(cls) -> "MedTable":
        contents = None
        with open(cls.FILE_NAME, "rb") as f:
            contents = pickle.load(f)
        return contents
    def encoding_data(self) -> tuple[npt.NDArray, list[str]]:
        # --- Flatten: One document per CUI, concatenating definitions
        # Keep a parallel list of CUIs for mapping index -> CUI
        cuis = []
        embedding_documents = []
        for cui, ent in self.entities.items():
            print(f"Readying CUI [{cui}] definitions for embedding", end="\r", flush=True)
            cuis.append(cui)
            defs = ent.definitions
            embedding_documents.append(" ".join(en_def.definition for en_def in defs))

        cuis = np.array(cuis)
        return cuis, embedding_documents


@dataclass(slots=True)
class Paper:
    text: str
    _title_len: int
    sentences: list[tuple[int, int]] = field(default_factory=empty_list)
    mentions: list["EntityMention"] = field(default_factory=empty_list)
    @property
    def title(self) -> str:
        return self.text[0:self._title_len-1]
    @property
    def abstract(self) -> str:
        return self.text[self._title_len:]
    def get_context(self, start: int, end: int):
        i = 0
        while self.sentences[i][1] < start:
            i += 1
        j = i
        while self.sentences[j][1] < end:
            j += 1
        return self.text[i:j]


@dataclass(slots=True)
class EntityDefinition:
    sab: str
    definition: str
    sentences: list[tuple[int, int]]

@dataclass(slots=True)
class EntityMention:
    pmid: int
    range: tuple[int, int]
    synonym: str
    tuis: frozenset[str]
    cui: str



# --- Load corpus ---
#with open("mrdef.pkl", "rb") as f:
#    mrdef = pickle.load(f)

# --- Flatten: One document per CUI, concatenating definitions
# Keep a parallel list of CUIs for mapping index -> CUI
#cuis = []
#embedding_documents = []
#for cui, defs in mrdef.items():
#    cuis.append(cui)
#    embedding_documents.append(" ".join(defn for _, defn in defs))

#cuis = np.array(cuis)

#def cui(idx: int):
#    return cuis[idx]
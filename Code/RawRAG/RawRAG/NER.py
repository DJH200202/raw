from typing import List, Dict
import spacy
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class NER:
    def __init__(self, spacy_model: str = "en_core_web_trf"):
        try:
            self.nlp = spacy.load(spacy_model)
            logging.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logging.info(f"SpaCy model {spacy_model} not found. Downloading...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)

    def extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )
        return entities

    def extract_entities_from_corpus(self, corpus: List[str]) -> List[List[Dict]]:
        entities = []
        for text in corpus:
            entities.append(self.extract_entities(text))
        return entities

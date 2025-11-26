# symptom_extractor.py

import re
import json
import difflib
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

# --------------------------------------------------------
#  symptom vocabulary same file used for the  model
# --------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SYMPTOM_COL_PATH = BASE_DIR.parent / "data" / "symptom_columns.json"

with open(SYMPTOM_COL_PATH, "r") as f:
    SYMPTOM_VOCAB = json.load(f)

# fast sentence-embedding model
MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------------
# Normalization fct to use later
# --------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase + strip punctuation + collapse spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------------
# small medical synonym map
# --------------------------------------------------------
CUSTOM_SYNONYMS = {
    
    "vomiting": ["vomit", "throwing up", "puking", "barfing"],
    "nausea": ["nauseous", "queasy", "upset stomach"],
    "abdominal_pain": ["stomach pain", "stomachache", "belly pain", "tummy pain"],
    "diarrhoea": ["diarrhea", "loose stools", "watery stool", "runny stool"],
    "constipation": ["hard stool", "difficulty pooping", "haven't pooped"],
    "heartburn": ["burning in chest after eating", "acid reflux", "acid in throat"],

    # General
    "fatigue": ["tired", "exhausted", "worn out", "no energy"],
    "fever": ["high temperature", "running a fever", "burning up", "hot body"],
    "chills": ["shivering", "cold sweats"],
    "sweating": ["sweaty", "cold sweat", "sweating a lot"],
    "headache": ["head pain", "pounding head", "migraine", "throbbing head"],
    "phonophobia": ["Discomfort from loud sounds","sensitivity to sounds"],
    "photophobia": [
    "sensitivity to light",
    "sensitive to light",
    "light hurts",
    "bright light hurts",
    "light pain",
    "sunlight hurts",
    "light makes it worse",
    "light bothers my eyes"],


    # Respiratory
    "cough": ["coughing", "dry cough", "wet cough"],
    "shortness_of_breath": [
        "short of breath", "breathless", "can't breathe",
        "difficulty breathing", "hard to breathe"
    ],
    "wheezing": ["whistling sound when breathing", "wheezy"],

    # Cardiac / chest
    "chest_pain": [
        "chest pain", "pain in my chest", "pressure on my chest",
        "tightness in my chest", "crushing chest pain"
    ],
    "chest_tightness": ["tight chest", "pressure chest"],
    "palpitations": ["heart racing", "pounding heart", "heart beating fast"],

    # Neuro
    "dizziness": ["lightheaded", "vertigo", "feeling faint", "about to pass out"],
}


# --------------------------------------------------------
# Literal and fuzzymatching
# --------------------------------------------------------
def literal_and_fuzzy_match(text_norm: str, symptom_vocab):
    """
    Literal and fuzzy match on a text.
    Returns a set of symptom names.
    """
    detected = set()

    for symptom in symptom_vocab:
        base = symptom.replace("_", " ")  # ex: 'chest_pain' to  'chest pain'

        # 1) Exact phrase in text
        if base in text_norm:
            detected.add(symptom)
            continue

        # 2) phrase vs whole text
        if difflib.SequenceMatcher(None, base, text_norm).ratio() > 0.75:
            detected.add(symptom)
            continue

        # 3)  match word-by-word
        for word in text_norm.split():
            if difflib.SequenceMatcher(None, word, base).ratio() > 0.85:
                detected.add(symptom)
                break

    return detected


# --------------------------------------------------------
# Custom synonyms
# --------------------------------------------------------
def custom_synonym_match(text_norm: str, symptom_vocab):
    """
     to map natural phrases to  canonical symptom names
    Only matches phrases that actually appear in text_norm
    """
    detected = set()

    for symptom in symptom_vocab:
        if symptom not in CUSTOM_SYNONYMS:
            continue

        for phrase in CUSTOM_SYNONYMS[symptom]:
            if phrase in text_norm:
                detected.add(symptom)
                break

    return detected


# --------------------------------------------------------
#  similarity
# --------------------------------------------------------
def semantic_match(user_text: str, symptom_vocab, threshold: float = 0.65):
    """
    Semantic similarity between user text and each symptom phrase
    high treshold to avoid random matches
    """
    if not user_text.strip():
        return set()

    # Encode text and symptom descriptions
    text_emb = MODEL.encode([user_text])[0]
    sym_strings = [s.replace("_", " ") for s in symptom_vocab]
    sym_emb = MODEL.encode(sym_strings)

    sims = util.cos_sim(text_emb, sym_emb).cpu().numpy()[0]

    matched = set()
    for i, score in enumerate(sims):
        if score >= threshold:
            matched.add(symptom_vocab[i])

    return matched


# --------------------------------------------------------
# maybe later for improved model not currently used its to detect emergency patterns 
# --------------------------------------------------------
def detect_emergency_flags(user_text: str):
    """
     simple rule-based redflag detector
    can call it in app.py
    to override predictions for real emergency patterns
    """
    t = normalize(user_text)
    flags = {}

    # Cardiac red flag example
    chest_terms = ["chest pain", "pressure on my chest", "crushing chest", "tightness in my chest"]
    arm_terms = ["left arm pain", "pain in my left arm"]
    sob_terms = ["shortness of breath", "short of breath", "can't breathe", "difficulty breathing"]
    sweat_terms = ["cold sweat", "cold sweats", "sweating a lot", "sweaty"]

    if any(p in t for p in chest_terms) and any(s in t for s in sob_terms):
        if any(a in t for a in arm_terms) or any(s in t for s in sweat_terms):
            flags["cardiac_red_flag"] = True

    return flags


# --------------------------------------------------------
# finla funct
# --------------------------------------------------------
def extract_symptoms(user_text: str):
    """
    Main hybrid extractor used by Flask.

    Strategy:
    - Normalize text
    - Get:
         custom synonym matches
         literal+fuzzy matches
         semantic matches 
    - Combine with >voting>:
         literal & synonym hits are strong (weight 2)
         semantic are weak (weight 1)
    - A symptom is kept if total score >= 2:
          requires at least one strong signal
           or combination (e.g. synonym + semantic).
    """
    if not user_text.strip():
        return []

    text_norm = normalize(user_text)

    synonym_hits = custom_synonym_match(text_norm, SYMPTOM_VOCAB)
    literal_hits = literal_and_fuzzy_match(text_norm, SYMPTOM_VOCAB)
    semantic_hits = semantic_match(user_text, SYMPTOM_VOCAB, threshold=0.65)

    # Voting system
    scores = {}

    def add_votes(sym_set, weight):
        for s in sym_set:
            scores[s] = scores.get(s, 0) + weight

    # Strong evidence
    add_votes(literal_hits, 2)
    add_votes(synonym_hits, 2)
    # Weak evidence
    add_votes(semantic_hits, 1)

    # Keep only symptoms with en9ug evidence
    final = [
        s for s, score in scores.items()
        if score >= 2
    ]

    # Prioritize symptioms that had literal/synonym maybe for latrer not used 
    if len(final) > 20:
        strong = [s for s in final if s in literal_hits or s in synonym_hits]
        weak = [s for s in final if s not in strong]
        final = strong[:20] if len(strong) >= 5 else (strong + weak)[:20]

    return final


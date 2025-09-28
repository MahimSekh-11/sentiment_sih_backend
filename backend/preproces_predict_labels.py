# import argparse
import re
import html
import string
from deep_translator import GoogleTranslator
from typing import List, Optional
import numpy as np
import pandas as pd
import emoji
from nltk.tokenize import word_tokenize
from textblob import TextBlob  
from gensim.models import Word2Vec, KeyedVectors
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0 

# ----------------------------
#  CONFIG / SMALL HELPERS
# ----------------------------
GENZ_DICT = {
    "brb": "be right back",
    "lol": "laughing out loud",
    "idk": "i do not know",
    "imo": "in my opinion",
    "btw": "by the way",
    "omg": "oh my god",
    "u": "you",
    "ur": "your",
    "pls": "please",
    "thx": "thanks",
}


def remove_html(text: str) -> str:
    # unescape any HTML entities and remove HTML tags
    text = html.unescape(text)
    # remove tags
    text = re.sub(r'<[^>]+>', ' ', text)
    return text
 # ensure consistent language detection

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def translate_to_english(text: str) -> str:
    if not text or text.strip() == "":
        return ""
    
    lang = detect_language(text)
    
    if lang == "en" or lang == "unknown":
        return text
    
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated
    except Exception as e:
        print(f"Translation failed: {e}")  # optional logging
        return text  # fallback: return original text

def remove_punctuation(text: str) -> str:
    # keep emojis (handled separately), remove punctuation
    return text.translate(str.maketrans('', '', string.punctuation))


def lower(text: str) -> str:
    return text.lower()


def expand_genz(text: str) -> str:
    tokens = text.split()
    expanded = [GENZ_DICT.get(t.lower(), t) for t in tokens]
    return " ".join(expanded)


def correct_spelling(text: str) -> str:
    # NOTE: TextBlob spelling correction is slow. Use only if necessary.
    try:
        tb = TextBlob(text)
        return str(tb.correct())
    except Exception:
        return text


def handle_emojis(text: str, remove=True) -> str:
    if remove:
        return emoji.replace_emoji(text, replace='')  # remove all emojis
    else:
        # convert to text description (emoji package may provide)
        return emoji.demojize(text, delimiters=(" ", " "))

def tokenize(text: str) -> List[str]:
    return word_tokenize(text)


# ----------------------------
#  Pipeline & training utilities
# ----------------------------
def preprocess_text(
    text: str,
    do_lower=True,
    remove_html_tags=True,
    remove_punct=True,
    expand_slang=True,
    do_spellcheck=False,
    remove_emoji=True,
    translate_to_english=True,
    do_tokenize=True
):
    if pd.isna(text):
        return ""
    s = str(text)
    if remove_html_tags:
        s = remove_html(s)
    if do_lower:
        s = lower(s)
    if translate_to_english:
        s = translate_to_english(s)
    if expand_slang:
        s = expand_genz(s)
    if remove_emoji:
        s = handle_emojis(s, remove=True)
    if remove_punct:
        s = remove_punctuation(s)
    if do_spellcheck:
        s = correct_spelling(s)
    tokens = tokenize(s) if do_tokenize else s.split()
    return " ".join(tokens)
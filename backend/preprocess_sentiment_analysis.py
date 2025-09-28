import pandas as pd
import re
from deep_translator import GoogleTranslator
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

import nltk
nltk.download("stopwords")
nltk.download("wordnet")


# Initialize NLP tools
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def simple_clean(text: str) -> str:
    # remove URLs, punctuation, digits, etc.
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def translate_to_english(text: str) -> str:
    if not text or text.strip() == "":
        return ""
    lang = detect_language(text)
    if lang == "en" or lang == "unknown":
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text  # fallback

def remove_stopwords_and_lemmatize(text: str) -> str:
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def preprocess_text_field(series: pd.Series) -> pd.Series:
    out = []
    for raw in series.astype(str).fillna(""):
        cleaned = simple_clean(raw)
        translated = translate_to_english(cleaned)
        cleaned2 = simple_clean(translated)
        final = remove_stopwords_and_lemmatize(cleaned2)
        out.append(final)
    return pd.Series(out)

def find_text_column(df: pd.DataFrame) -> str:
    preferred = ["clean_comment", "comment", "body", "text", "message"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].dtype == object:
            return col
    raise ValueError("No text column found in the dataframe")

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_col = find_text_column(df)
    df["original_" + text_col] = df[text_col]
    df["clean_comment"] = preprocess_text_field(df[text_col])

    # add extra metadata
    df["word_count"] = df["clean_comment"].apply(lambda x: len(str(x).split()))
    df["num_chars"] = df["clean_comment"].apply(lambda x: len(str(x)))

    df = df[df["clean_comment"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)
    return df

def wordcloud_to_base64(text: str) -> str:
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    buf = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

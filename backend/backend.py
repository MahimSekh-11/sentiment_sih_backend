from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import io, os, re, base64, json
from typing import Optional
from pydantic import BaseModel
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv
import seaborn as sns
import numpy as np
import requests
import uvicorn

# ---------------------- API ----------------------
app = FastAPI(title="Sentiment Analysis API")


# Load .env file from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
# API Key for our app
API_KEY = os.getenv("API_KEY")  # Replace with secure value
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# Hugging Face API
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("❌ Please set HF_API_KEY in environment (Render env vars).")

HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
HF_API_URL = "https://api-inference.huggingface.co/models"

# Models
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
GEN_MODEL = "google/flan-t5-base"

def hf_query(model: str, payload: dict):
    url = f"{HF_API_URL}/{model}"
    response = requests.post(url, headers=HF_HEADERS, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()

# ---------------------- Helpers ----------------------
def preprocess_text(text: str):
    return text.strip()

def ngrams_from_text(text: str, n: int):
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def compute_clause_sentiment(df: pd.DataFrame, category_col: str):
    if "Clause" not in df.columns or df["Clause"].dropna().empty:
        return {}
    clause_data = {}
    for clause, group in df.groupby("Clause"):
        counts = group[category_col].value_counts(normalize=True) * 100
        clause_data[clause] = {
            "Positive": round(counts.get("Positive", 0), 2),
            "Neutral": round(counts.get("Neutral", 0), 2),
            "Negative": round(counts.get("Negative", 0), 2)
        }
    return clause_data

def generate_clause_sentiment_chart(clause_data: dict):
    if not clause_data:
        return None
    df_chart = pd.DataFrame(clause_data).T[["Positive", "Neutral", "Negative"]]
    n_clauses = len(df_chart)
    index = np.arange(n_clauses)
    bar_width = 0.25

    plt.figure(figsize=(10, max(4, n_clauses * 0.5)))
    plt.barh(index - bar_width, df_chart["Positive"], height=bar_width, color="green", label="Positive")
    plt.barh(index, df_chart["Neutral"], height=bar_width, color="gray", label="Neutral")
    plt.barh(index + bar_width, df_chart["Negative"], height=bar_width, color="red", label="Negative")
    plt.yticks(index, df_chart.index)
    plt.xlabel("Percentage (%)")
    plt.ylabel("Clause")
    plt.title("Clause Sentiment Distribution")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_wordcount_distribution(df: pd.DataFrame, category_col: str):
    if "clean_comment" not in df.columns:
        return []
    df_wc = df.copy()
    df_wc["word_count"] = df_wc["clean_comment"].str.split().str.len()
    if df_wc[category_col].dtype != str:
        df_wc[category_col] = df_wc[category_col].astype(str)
    return df_wc[[category_col, "word_count"]].rename(columns={category_col: "category"}).to_dict(orient="records")

# ---------------------- Analyze CSV ----------------------
@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    if "Comment" not in df.columns:
        raise HTTPException(status_code=400, detail="No 'Comment' column in CSV")

    df["clean_comment"] = df["Comment"].astype(str).apply(preprocess_text)
    category_col = "Label" if "Label" in df.columns else None
    if not category_col:
        raise HTTPException(status_code=400, detail="No sentiment label column found in CSV")

    all_text = " ".join(df["clean_comment"].astype(str).tolist())

    analysis = {}
    analysis["sentiment_counts"] = df[category_col].value_counts().to_dict()
    analysis["top_words"] = Counter(all_text.split()).most_common(50)
    analysis["top_bigrams"] = Counter(ngrams_from_text(all_text, 2)).most_common(25)
    analysis["top_trigrams"] = Counter(ngrams_from_text(all_text, 3)).most_common(25)
    analysis["total_comments"] = len(df)
    analysis["unique_clause"] = int(df["Clause"].nunique()) if "Clause" in df.columns else 0
    analysis["avg_word_count"] = float(df["clean_comment"].str.split().str.len().mean())
    analysis["sample_processed"] = df[["clean_comment", category_col, "Clause"]].head(10).to_dict(orient="records")
    clause_data = compute_clause_sentiment(df, category_col)
    analysis["clause_sentiment"] = clause_data
    analysis["clause_sentiment_chart_base64"] = generate_clause_sentiment_chart(clause_data)
    analysis["word_count_data"] = generate_wordcount_distribution(df, category_col)

    return JSONResponse(content=analysis)

# ---------------------- Summarize ----------------------
class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(request: SummarizeRequest):
    try:
        result = hf_query(SUMMARIZER_MODEL, {"inputs": request.text})
        summary = result[0]["summary_text"]
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------- Predict Sentiment ----------------------
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    try:
        text_clean = preprocess_text(request.text)
        output = hf_query(SENTIMENT_MODEL, {"inputs": text_clean})
        stars = output[0]
        stars_sorted = sorted(stars, key=lambda x: x["score"], reverse=True)
        top = stars_sorted[0]
        star_label = top["label"]
        confidence = round(top["score"], 4)
        star_to_sentiment = {
            "1 star": "Negative",
            "2 stars": "Negative",
            "3 stars": "Neutral",
            "4 stars": "Positive",
            "5 stars": "Positive",
        }
        sentiment = star_to_sentiment.get(star_label, "Neutral")
        return {"prediction": sentiment, "stars": star_label, "Confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

# ---------------------- Classify File ----------------------
@app.post("/classify_file")
async def classify_file(file: UploadFile = File(...), verified: bool = Depends(verify_api_key)):
    filename = file.filename
    ext = os.path.splitext(filename)[-1].lower()
    try:
        contents = await file.read()
        if ext == ".csv":
            df = pd.read_csv(io.StringIO(contents.decode('utf-8', errors='ignore')))
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": f"Unsupported file type: {ext}"}
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    if "Comment" not in df.columns:
        return {"error": "File must have a 'Comment' column"}
    if "ID" not in df.columns:
        df["ID"] = range(1, len(df) + 1)

    df["Comment"] = df["Comment"].apply(preprocess_text)
    labels, scores = [], []
    for sentence in df["Comment"]:
        try:
            output = hf_query(SENTIMENT_MODEL, {"inputs": sentence})
            stars = output[0]
            stars_sorted = sorted(stars, key=lambda x: x["score"], reverse=True)
            top = stars_sorted[0]
            star_label = top["label"]
            confidence = round(top["score"], 4)
            star_to_sentiment = {
                "1 star": "Negative",
                "2 stars": "Negative",
                "3 stars": "Neutral",
                "4 stars": "Positive",
                "5 stars": "Positive",
            }
            label = star_to_sentiment.get(star_label, "Neutral")
        except Exception:
            label, confidence = "Error", 0.0
        labels.append(label)
        scores.append(confidence)

    df["Label"] = labels
    df["Confidence"] = scores
    records = df.to_dict(orient="records")
    return JSONResponse(content={"message": "File classified successfully", "data": records})

# ---------------------- Analyze Comment ----------------------
class CommentRequest(BaseModel):
    comment: str
    clause: Optional[str] = None

@app.post("/analyze_comment")
def analyze_comment(request: CommentRequest):
    try:
        comment = request.comment
        output = hf_query(SENTIMENT_MODEL, {"inputs": comment})
        stars = output[0]
        stars_sorted = sorted(stars, key=lambda x: x["score"], reverse=True)
        top = stars_sorted[0]
        star_label = top["label"]
        confidence = round(top["score"], 3)
        star_to_sentiment = {
            "1 star": "Negative",
            "2 stars": "Negative",
            "3 stars": "Neutral",
            "4 stars": "Positive",
            "5 stars": "Positive",
        }
        sentiment = star_to_sentiment.get(star_label, "Neutral")

        # Reason
        if sentiment == "Negative":
            prompt = f"Explain clearly why the following comment is negative:\n\n{comment}\n\nExplanation:"
            reason = hf_query(GEN_MODEL, {"inputs": prompt})[0]["generated_text"]
        else:
            reason = "No major negative sentiment detected."

        # Suggestion
        prompt = f"Rewrite this comment politely. Comment: {comment}"
        if request.clause:
            prompt = f"Rewrite this comment politely so it aligns with Clause: {request.clause}. Comment: {comment}"
        suggestion = hf_query(GEN_MODEL, {"inputs": prompt})[0]["generated_text"]

        return {"sentiment": sentiment, "confidence": confidence, "reason": reason, "suggestion": suggestion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------- Root ----------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiment Analysis API is running."}

# ---------------------- Run ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8500)

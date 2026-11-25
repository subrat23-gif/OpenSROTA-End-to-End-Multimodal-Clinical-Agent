import json
import os
import random
import sys
import numpy as np
import difflib
import re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE_DIR = "/nas/vivek2/subrat/clinical_agent"
GOLD_DIR = os.path.join(BASE_DIR, "outputs/mts_gold_asr")  
PRED_DIR = os.path.join(BASE_DIR, "outputs/mts_preds_verified")  
WHISPER_DIR = os.path.join(BASE_DIR, "data/transcripts/mts_whisper")
OMI_DIR = os.path.join(BASE_DIR, "data/transcripts/mts_omi")

FIELDS = ["complaints", "symptoms", "history", "medications", "allergies", "vitals",
          "examination", "assessment", "investigations", "advice", "follow_up", "referrals"]

STOPWORDS = {"of", "the", "a", "an", "and", "or", "to", "in", "on", "with", "for", "history", "patient", "current"}

def norm_tokens(s):
    """Extracts content words for Jaccard similarity."""
    words = re.findall(r"[a-z0-9]+", str(s).lower())
    return set(w for w in words if w not in STOPWORDS)

def robust_match(a, b):
    """Matches facts using both character similarity and word overlap."""
    a, b = str(a).lower(), str(b).lower()
    if a == b: return True
    
    # 1. Character Sequence Matcher
    seq_score = difflib.SequenceMatcher(None, a, b).ratio()
    if seq_score >= 0.75: return True
    
    # 2. Jaccard Token Overlap (handles paraphrasing/word order)
    ta, tb = norm_tokens(a), norm_tokens(b)
    if ta and tb:
        jaccard = len(ta & tb) / len(ta | tb)
        if jaccard >= 0.65: return True
        
    return False

def fuzzy_in_text(fact_name, text):
    if not fact_name or not text: return 0.0
    if str(fact_name).lower() in str(text).lower(): return 1.0
    
    words = str(text).lower().split()
    fact_words = str(fact_name).lower().split()
    n = len(fact_words)
    if n == 0: return 0.0
    
    best_score = 0.0
    for i in range(max(1, len(words) - n + 1)):
        window = " ".join(words[i:i+n])
        score = difflib.SequenceMatcher(None, window, str(fact_name).lower()).ratio()
        best_score = max(best_score, score)
    return 1.0 if best_score >= 0.75 else 0.0

def ltt_calibrate(scores, errors, alpha, delta=0.05):
    idx = np.argsort(scores)
    scores = scores[idx]
    errors = errors[idx]
    n = len(scores)
    cum_errors = np.cumsum(errors)
    cum_risk = cum_errors / np.arange(1, n + 1)
    bound = cum_risk + np.sqrt(np.log(1/delta) / (2 * np.arange(1, n + 1)))
    valid_idx = np.where(bound <= alpha)[0]
    if len(valid_idx) == 0: return -1.0 
    return scores[valid_idx[-1]]

def main():
    X, y = [], []
    
    for pred_file in Path(PRED_DIR).glob("*.json"):
        doc_id = pred_file.stem
        gold_file = Path(GOLD_DIR) / f"{doc_id}.json"
        w_file = Path(WHISPER_DIR) / f"{doc_id}.json"
        o_file = Path(OMI_DIR) / f"{doc_id}.json"
        
        if not gold_file.exists(): continue
            
        pred_data = json.load(open(pred_file))
        gold_data = json.load(open(gold_file))
        w_data = json.load(open(w_file)) if w_file.exists() else []
        o_data = json.load(open(o_file)) if o_file.exists() else {}
        omi_text = o_data.get("transcript", "")
        
        # Flatten gold facts
        gold_facts = []
        for f in FIELDS:
            for fact in gold_data.get(f, []):
                gold_facts.append(fact["name"])
                
        for field in FIELDS:
            for fact in pred_data.get(field, []):
                # THE FIX: Handle strings if the verifier simplified the output
                if isinstance(fact, str):
                    name = fact
                    ev = []
                elif isinstance(fact, dict):
                    name = fact.get("name", "")
                    ev = fact.get("evidence", [])
                else:
                    continue
                    
                if not name:
                    continue
                
                # Target: Is this fact in the gold standard?
                is_err = 1
                for g in gold_facts:
                    if robust_match(name, g):
                        is_err = 0
                        break
                        
                # Feature 1: Omi found?
                omi_found = fuzzy_in_text(name, omi_text)
                
                # Feature 2: Whisper confidence
                min_prob = 0.0
                if ev and w_data:
                    quote_words = [norm_tokens(w) for w in ev[0].get("quote", "").split() if len(w) > 2]
                    probs = [w["prob"] for w in w_data if norm_tokens(w["word"]) in quote_words]
                    if probs: min_prob = min(probs)
                
                X.append([omi_found, min_prob, len(name)])
                y.append(is_err)

    X, y = np.array(X), np.array(y)
    if len(np.unique(y)) < 2:
        print("Not enough variance.")
        return
        
    split = int(len(X) * 0.5)
    clf = LogisticRegression(class_weight='balanced').fit(X[:split], y[:split])
    scores_cal = clf.predict_proba(X[:split])[:, 1]
    scores_test = clf.predict_proba(X[split:])[:, 1]
    y_test = y[split:]
    
    print(f"Total Facts Evaluated: {len(X)}")
    print(f"Base Error Rate (Test): {y_test.mean():.4f}")
    print(f"Hybrid AUC Test: {roc_auc_score(y_test, scores_test):.4f}\n")
    
    print(f"{'Alpha':<8} | {'Coverage':<10} | {'Empirical Risk':<15} | {'Guarantee Met?'}")
    print("-" * 60)

    for alpha in [0.10, 0.20, 0.30, 0.40, 0.50]:
        tau = ltt_calibrate(scores_cal, y[:split], alpha)
        if tau == -1.0:
            print(f"{alpha:<8.2f} | {'0.00%':<10} | {'N/A':<15} | No (LTT Failed)")
        else:
            accepted = scores_test <= tau
            coverage = accepted.mean()
            if coverage > 0:
                empirical_risk = y_test[accepted].mean()
                guarantee_met = "YES" if empirical_risk <= alpha else "NO"
                print(f"{alpha:<8.2f} | {coverage*100:<9.2f}% | {empirical_risk:<15.4f} | {guarantee_met}")
            else:
                print(f"{alpha:<8.2f} | {'0.00%':<10} | {'N/A':<15} | No (Rejected All)")

if __name__ == "__main__":
    main()
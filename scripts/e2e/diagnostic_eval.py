import json, os, difflib, re
from pathlib import Path

BASE_DIR = "/nas/vivek2/subrat/clinical_agent"
FIELDS = ["complaints", "symptoms", "history", "medications", "allergies", "vitals",
          "examination", "assessment", "investigations", "advice", "follow_up", "referrals"]

STOPWORDS = {"of", "the", "a", "an", "and", "or", "to", "in", "on", "with", "for", "history", "patient", "current"}

def tokens(s): return set(w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if w not in STOPWORDS)

def match(a, b):
    ta, tb = tokens(a), tokens(b)
    if ta == tb and ta: return True
    if difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio() >= 0.75: return True
    if ta and tb and len(ta & tb) / len(ta | tb) >= 0.65: return True
    return False

def eval_config(pred_dir, gold_dir):
    n_pred, n_gold, n_match = 0, 0, 0
    for gf in Path(gold_dir).glob("*.json"):
        doc_id = gf.stem
        pf = Path(pred_dir) / f"{doc_id}.json"
        if not pf.exists(): continue
        gold = json.load(open(gf))
        pred = json.load(open(pf))
        g_facts = [f["name"] for field in FIELDS for f in gold.get(field, [])]
        p_facts = [f["name"] for field in FIELDS for f in pred.get(field, [])]
        
        n_gold += len(g_facts)
        n_pred += len(p_facts)
        for p in p_facts:
            if any(match(p, g) for g in g_facts): n_match += 1
            
    precision = n_match / n_pred if n_pred else 0
    recall = n_match / n_gold if n_gold else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"n_pred": n_pred, "n_gold": n_gold, "n_match": n_match, 
            "P": round(precision, 3), "R": round(recall, 3), "F1": round(f1, 3)}

configs = [
    ("30B Teacher on Clean (Upper Bound)", "outputs/mts_gold", "outputs/mts_gold"),
    ("Cascade: 4B on ASR", "outputs/mts_preds_4b_asr", "outputs/mts_gold"),
    ("E2E: Qwen2-Audio 7B (v2)", "outputs/mts_preds_e2e_v2", "outputs/mts_gold"),
]

print(f"{'Config':<40} | {'P':<6} | {'R':<6} | {'F1':<6} | {'n_pred':<8} | {'n_gold':<8}")
print("-" * 100)
for name, pred_dir, gold_dir in configs:
    if not os.path.exists(pred_dir):
        print(f"{name:<40} | SKIP (not found)")
        continue
    r = eval_config(pred_dir, gold_dir)
    print(f"{name:<40} | {r['P']:<6} | {r['R']:<6} | {r['F1']:<6} | {r['n_pred']:<8} | {r['n_gold']:<8}")
import re
from collections import defaultdict

OBLIGATION_PATTERN = re.compile(
    r"([A-Z][a-zA-Z\s]{2,40}?)\s+(?:shall|must|agrees?\s+to|is\s+required\s+to|undertakes?\s+to|will|is\s+obligated\s+to)\s+([^.!?]{10,150})[.!?]",
    re.IGNORECASE
)
PERMISSION_PATTERN = re.compile(
    r"([A-Z][a-zA-Z\s]{2,40}?)\s+(?:may|is\s+entitled\s+to|has\s+the\s+right\s+to|reserves?\s+the\s+right\s+to|is\s+permitted\s+to)\s+([^.!?]{10,150})[.!?]",
    re.IGNORECASE
)
PROHIBITION_PATTERN = re.compile(
    r"([A-Z][a-zA-Z\s]{2,40}?)\s+(?:shall\s+not|must\s+not|may\s+not|is\s+prohibited\s+from|cannot|will\s+not)\s+([^.!?]{10,150})[.!?]",
    re.IGNORECASE
)
GENERIC_PARTIES = {
    "licensor": "Licensor", "licensee": "Licensee", "vendor": "Vendor",
    "client": "Client", "customer": "Customer", "company": "Company",
    "affiliate": "Affiliate", "provider": "Provider", "recipient": "Recipient",
    "disclosing party": "Disclosing Party", "receiving party": "Receiving Party",
    "party a": "Party A", "party b": "Party B",
}

def _normalize_party(text):
    t = re.sub(r"^the\s+", "", text.strip().lower())
    for key, canonical in GENERIC_PARTIES.items():
        if key in t:
            return canonical
    return text.strip().title()[:30]

def _is_valid_party(text):
    t = text.strip().lower()
    invalid = ["this agreement","the agreement","section","clause","exhibit","schedule",
               "appendix","herein","hereof","thereof","each","either","both","all","any",
               "no","such","said","above","following","foregoing","nothing","everything"]
    if len(t) < 2 or len(t) > 40:
        return False
    if any(t.startswith(inv) for inv in invalid):
        return False
    if re.match(r"^\d", t):
        return False
    return True

def extract_obligations(spans, clause_df):
    obligations = []
    span_to_clause = {int(row["span_id"]): row["final_clause"] for _, row in clause_df.iterrows()}
    for span_id, span_text in enumerate(spans):
        clause_type = span_to_clause.get(span_id, "Unknown")
        for pattern, verb_type in [(OBLIGATION_PATTERN,"obligation"),(PERMISSION_PATTERN,"permission"),(PROHIBITION_PATTERN,"prohibition")]:
            for match in pattern.finditer(span_text):
                subject_raw = match.group(1).strip()
                action = match.group(2).strip()[:120]
                if not _is_valid_party(subject_raw):
                    continue
                obligations.append({
                    "subject": _normalize_party(subject_raw),
                    "verb_type": verb_type,
                    "action": action,
                    "span_id": span_id,
                    "clause_type": clause_type,
                    "raw_sentence": match.group(0)[:200],
                })
    return obligations

def build_obligation_graph(obligations):
    if not obligations:
        return {"nodes":[],"edges":[],"adjacency":{},"obligation_counts":{},"balance_score":1.0,"dominant_party":None,"missing_reciprocals":[],"clause_density":{}}
    obligation_counts = defaultdict(lambda: {"obligation":0,"permission":0,"prohibition":0})
    for ob in obligations:
        obligation_counts[ob["subject"]][ob["verb_type"]] += 1
    parties = list(obligation_counts.keys())
    edges = []
    adjacency = defaultdict(lambda: defaultdict(int))
    for ob in obligations:
        subject = ob["subject"]
        action_lower = ob["action"].lower()
        target = next((p for p in parties if p != subject and p.lower() in action_lower), "General Obligation")
        edges.append({"from":subject,"to":target,"verb_type":ob["verb_type"],"action":ob["action"],"clause_type":ob["clause_type"]})
        if ob["verb_type"] == "obligation":
            adjacency[subject][target] += 1
    total_ob = [v["obligation"] for v in obligation_counts.values()]
    balance_score = round(min(total_ob)/max(max(total_ob),1),3) if len(total_ob)>=2 and sum(total_ob)>0 else 1.0
    dominant_party = max(obligation_counts.keys(), key=lambda p: obligation_counts[p]["obligation"], default=None)
    action_keywords = defaultdict(lambda: defaultdict(list))
    for ob in obligations:
        if ob["verb_type"] == "obligation":
            words = ob["action"].lower().split()[:3]
            key = " ".join(words[:2]) if len(words)>=2 else (words[0] if words else "")
            action_keywords[ob["subject"]][key].append(ob)
    missing_reciprocals = []
    seen = set()
    for pa, actions_a in action_keywords.items():
        for pb, actions_b in action_keywords.items():
            if pa == pb: continue
            for key_a in actions_a:
                if not key_a: continue
                has_rec = any(key_a[:4] in kb or kb[:4] in key_a for kb in actions_b)
                if not has_rec:
                    k = f"{pa}:{key_a[:20]}"
                    if k not in seen:
                        seen.add(k)
                        missing_reciprocals.append({"party_a":pa,"party_b":pb,"obligation":key_a,"example":actions_a[key_a][0]["raw_sentence"][:150]})
                        if len(missing_reciprocals)>=8: break
    clause_density = defaultdict(int)
    for ob in obligations:
        if ob["clause_type"] != "Unknown":
            clause_density[ob["clause_type"]] += 1
    return {
        "nodes": parties,
        "edges": edges,
        "adjacency": {k:dict(v) for k,v in adjacency.items()},
        "obligation_counts": {k:dict(v) for k,v in obligation_counts.items()},
        "balance_score": balance_score,
        "dominant_party": dominant_party,
        "missing_reciprocals": missing_reciprocals[:8],
        "clause_density": dict(sorted(clause_density.items(),key=lambda x:x[1],reverse=True)),
    }
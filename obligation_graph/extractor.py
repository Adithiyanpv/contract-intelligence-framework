import re
from collections import defaultdict

OBLIGATION_PATTERN = re.compile(
    r"([A-Z][a-zA-Z]{2,25}(?:\s+[A-Z][a-zA-Z]{2,20})?)\s+(?:shall|must|agrees?\s+to|is\s+required\s+to|undertakes?\s+to|will|is\s+obligated\s+to)\s+([^.!?]{10,150})[.!?]",
    re.IGNORECASE
)
PERMISSION_PATTERN = re.compile(
    r"([A-Z][a-zA-Z]{2,25}(?:\s+[A-Z][a-zA-Z]{2,20})?)\s+(?:may|is\s+entitled\s+to|has\s+the\s+right\s+to|reserves?\s+the\s+right\s+to|is\s+permitted\s+to)\s+([^.!?]{10,150})[.!?]",
    re.IGNORECASE
)
PROHIBITION_PATTERN = re.compile(
    r"([A-Z][a-zA-Z]{2,25}(?:\s+[A-Z][a-zA-Z]{2,20})?)\s+(?:shall\s+not|must\s+not|may\s+not|is\s+prohibited\s+from|cannot|will\s+not)\s+([^.!?]{10,150})[.!?]",
    re.IGNORECASE
)

GENERIC_PARTIES = {
    "licensor": "Licensor", "licensee": "Licensee", "vendor": "Vendor",
    "client": "Client", "customer": "Customer", "company": "Company",
    "affiliate": "Affiliate", "provider": "Provider", "recipient": "Recipient",
    "disclosing party": "Disclosing Party", "receiving party": "Receiving Party",
    "party a": "Party A", "party b": "Party B", "chase": "Chase",
    "you": "You (Affiliate)", "employer": "Employer", "employee": "Employee",
}

# Words that are definitely NOT parties
NOT_PARTIES = {
    "this", "the", "a", "an", "each", "either", "both", "all", "any", "no",
    "such", "said", "above", "following", "foregoing", "nothing", "everything",
    "section", "clause", "exhibit", "schedule", "appendix", "herein", "hereof",
    "thereof", "agreement", "contract", "party", "parties", "and", "or", "but",
    "if", "when", "where", "which", "that", "who", "whom", "whose", "what",
    "notwithstanding", "pursuant", "subject", "provided", "including", "except",
    "unless", "until", "upon", "during", "within", "without", "between", "among",
    "order", "tracking", "governing", "modification", "credit", "card", "website",
    "registration", "processing", "specific", "written", "preliminary",
}

def _normalize_party(text):
    t = re.sub(r"^the\s+", "", text.strip().lower())
    for key, canonical in GENERIC_PARTIES.items():
        if t == key or t.startswith(key + " "):
            return canonical
    return text.strip().title()[:25]

def _is_valid_party(text):
    t = text.strip().lower()
    t_clean = re.sub(r"^the\s+", "", t)
    # Must be 2-25 chars
    if len(t_clean) < 2 or len(t_clean) > 30:
        return False
    # First word must not be a non-party word
    first_word = t_clean.split()[0] if t_clean.split() else ""
    if first_word in NOT_PARTIES:
        return False
    # Must not start with a digit
    if re.match(r"^\d", t_clean):
        return False
    # Must not contain special chars (URLs, punctuation)
    if re.search(r"[:/\(\)\[\]@#]", text):
        return False
    # Must look like a proper noun (starts with capital) or known party
    if not text.strip()[0].isupper():
        return False
    # Reject if it looks like a sentence fragment (contains verb-like words)
    fragment_words = {"processing", "tracking", "governing", "modification", "registration", "ordering"}
    if any(fw in t_clean for fw in fragment_words):
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
                normalized = _normalize_party(subject_raw)
                # Skip if normalized is still a non-party word
                if normalized.lower() in NOT_PARTIES:
                    continue
                obligations.append({
                    "subject": normalized,
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
        target = next((p for p in parties if p != subject and p.lower() in action_lower), "General")
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
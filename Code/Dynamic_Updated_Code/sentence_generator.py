# sentence_generator.py (refined)

verb_map = {
    ("Ami", "asa"):   "আমি আসি",
    ("Ami", "jaoa"):  "আমি যাই",
    ("Ami", "bola"):  "আমি বলি",

    ("tumi", "asa"):  "তুমি আসো",
    ("tumi", "jaoa"): "তুমি যাও",
    ("tumi", "bola"): "তুমি বলো",

    ("se", "asa"):    "সে আসে",
    ("se", "jaoa"):   "সে যায়",
    ("se", "bola"):   "সে বলে",
}

SUBJECTS = {"Ami", "tumi", "se"}
VERBS    = {"asa", "jaoa", "bola"}

def _dedup(seq):
    out = []
    for w in seq:
        if len(out)==0 or out[-1] != w:
            out.append(w)
    return out

def generate_sentence(pred_words):
    """
    pred_words: list of predicted labels (strings)
    Strategy:
      - remove consecutive duplicates
      - find first subject and first verb (any order)
      - if both found -> map to Bangla sentence
      - else -> return informative fallback
    """
    pred_words = _dedup(pred_words)

    subj = None
    verb = None

    # pass-1: pick first subject/verb in any order
    for w in pred_words:
        if subj is None and w in SUBJECTS:
            subj = w
        if verb is None and w in VERBS:
            verb = w
        if subj and verb:
            break

    if subj and verb:
        return verb_map.get((subj, verb), f"{subj} {verb}")

    if subj and not verb:
        return f"{subj} (verb missing)"
    if verb and not subj:
        return f"(subject missing) {verb}"

    return "(no valid sentence)"

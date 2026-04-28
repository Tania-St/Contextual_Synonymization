import re
import pandas as pd


def prepare_dataframe(texts, term):
    rows = []
    for i, t in enumerate(texts):
        if re.search(rf"\b{term}\b", t, re.IGNORECASE):
            masked = re.sub(rf"\b{term}\b", "[MASK]", t, flags=re.IGNORECASE)
            rows.append({
                "index": i,
                "text": t,
                "anon_text": masked
            })
    return pd.DataFrame(rows)
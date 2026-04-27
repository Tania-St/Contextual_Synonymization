import torch
import numpy as np
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

def embed_masked(texts, tokenizer, model, device):
    model.to(device)
    model.eval()
    vectors = []

    for t in texts:
        inputs = tokenizer(
            t,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(device)

        mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

        with torch.no_grad():
            outputs = model(**inputs)

        if len(mask_idx[1]) == 0:
            vec = torch.zeros(model.config.hidden_size).cpu().numpy()
        else:
            vec = outputs.last_hidden_state[0, mask_idx[1], :].mean(dim=0).cpu().numpy()

        vectors.append(vec)

    return np.vstack(vectors)

def embed_cls(texts, tokenizer, model, device):
    model.to(device)
    model.eval()
    vectors = []

    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        vec = outputs.last_hidden_state[:, 0, :]  # CLS
        vectors.append(vec.squeeze(0).cpu().numpy())

    return np.vstack(vectors)

def predict_mask(tokenizer, model, text, device, top_k=5):
    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    mask_token_id = tokenizer.mask_token_id
    mask_idx = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)

    if len(mask_idx[1]) == 0:
        return []

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    mask_positions = mask_idx[1]
    logits_mask = logits[0, mask_positions, :]

    probs = F.softmax(logits_mask, dim=-1)

    top_tokens = torch.topk(probs, top_k, dim=-1).indices[0]

    words = tokenizer.convert_ids_to_tokens(top_tokens)

    return words



def embed_tfidf(texts):
    vectorizer = TfidfVectorizer(max_features=768)
    X = vectorizer.fit_transform(texts)
    return X.toarray()
import torch
import numpy as np

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
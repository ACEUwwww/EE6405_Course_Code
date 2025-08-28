from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_word_embedding(sentence, target_word):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)

    tokens = tokenizer.tokenize(sentence)
    target_ids = [i for i, tok in enumerate(tokens) if target_word in tok]

    word_embedding = embeddings[target_ids].mean(dim=0)
    return word_embedding

sent1 = "I went to the bank to deposit money."
sent2 = "The fisherman sat by the bank of the river."

emb1 = get_word_embedding(sent1, "bank")
emb2 = get_word_embedding(sent2, "bank")

similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))

print("Cosine similarity:", similarity.item())

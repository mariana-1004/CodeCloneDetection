import torch
from transformers import RobertaTokenizer, RobertaModel
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('HF_TOKEN')

def load_codebert(model_name, device):
    tokenizer = RobertaTokenizer.from_pretrained(model_name, token=token)
    codebert = RobertaModel.from_pretrained(model_name, token=token, use_safetensors=True).to(device)

    for param in codebert.parameters():
        param.requires_grad = False

    codebert.eval()

    return tokenizer, codebert


def generate_embeddings(
    func1_list,
    func2_list,
    labels,
    tokenizer,
    codebert,
    device,
    max_length=128,
    batch_size=32
):
    codebert.eval()

    all_combined_embeddings = []
    all_labels = []

    for i in range(0, len(labels), batch_size):
        batch_func1 = func1_list[i:i + batch_size]
        batch_func2 = func2_list[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        func1_tokens = tokenizer(
            batch_func1,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        func2_tokens = tokenizer(
            batch_func2,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        func1_input_ids = func1_tokens["input_ids"].to(device)
        func1_attention_mask = func1_tokens["attention_mask"].to(device)

        func2_input_ids = func2_tokens["input_ids"].to(device)
        func2_attention_mask = func2_tokens["attention_mask"].to(device)

        with torch.no_grad():
            func1_outputs = codebert(
                input_ids=func1_input_ids,
                attention_mask=func1_attention_mask
            )

            func2_outputs = codebert(
                input_ids=func2_input_ids,
                attention_mask=func2_attention_mask
            )

            func1_embedding = func1_outputs.last_hidden_state[:, 0, :]
            func2_embedding = func2_outputs.last_hidden_state[:, 0, :]

            difference = torch.abs(func1_embedding - func2_embedding)
            product = func1_embedding * func2_embedding

            combined = torch.cat(
                [func1_embedding, func2_embedding, difference, product],
                dim=1
            )

        all_combined_embeddings.append(combined.cpu())
        all_labels.extend(batch_labels)

        if i % (batch_size * 10) == 0:
            print(f"Generated embeddings: {i}/{len(labels)}")

    all_combined_embeddings = torch.cat(all_combined_embeddings, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.float)

    return all_combined_embeddings, all_labels


def save_embeddings(embeddings, labels, embeddings_path, labels_path):
    torch.save(embeddings, embeddings_path)
    torch.save(labels, labels_path)


def load_embeddings(embeddings_path, labels_path):
    embeddings = torch.load(embeddings_path)
    labels = torch.load(labels_path)

    return embeddings, labels

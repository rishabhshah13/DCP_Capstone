from transformers import BertForSequenceClassification, BertTokenizerFast
import pandas as pd
import torch
from tqdm import tqdm


def des_classifier(df, model_path="DCPduke/Des-classification-model", batch_size=8):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    def predict_batch(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

        model.eval()

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = outputs.logits.softmax(dim=1)

        pred_label_idxs = probs.argmax(dim=1)

        pred_labels = [model.config.id2label[idx.item()] for idx in pred_label_idxs]

        return pred_labels

    texts = df['Company Li Description'].fillna('null').tolist()

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    predictions = []
    for batch in tqdm(batches, desc="Processing batches", leave=False):
        batch_predictions = predict_batch(batch)
        predictions.extend(batch_predictions)

    df['Company Des Relevant Score'] = predictions

    return df

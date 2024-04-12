from transformers import BertForSequenceClassification, BertTokenizerFast
import pandas as pd
import torch
from tqdm import tqdm

def des_classifier(df, model_path="DCPduke/Des-classification-model", batch_size=8):
    """
    Classifies company descriptions using a pre-trained BERT model.

    Args:
    df (DataFrame): Input dataframe containing company data.
    model_path (str): Path to the pre-trained BERT model. Default is "DCPduke/Des-classification-model".
    batch_size (int): Batch size for processing. Default is 8.

    Returns:
    DataFrame: DataFrame with added 'Company Des Relevant Score' column containing classification scores.
    """

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using Device: {device}")

    # Load the model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    def predict_batch(texts):
        """
        Predicts relevant scores for a batch of texts using the loaded BERT model.

        Args:
        texts (list): List of company descriptions.

        Returns:
        list: Predicted relevant scores for the batch.
        """
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

    # Split texts into batches for processing
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    predictions = []
    # Process batches and collect predictions
    for batch in tqdm(batches, desc="Processing batches", leave=False):
        batch_predictions = predict_batch(batch)
        predictions.extend(batch_predictions)

    # Add relevant scores to the dataframe
    df['Company Des Relevant Score'] = predictions

    return df

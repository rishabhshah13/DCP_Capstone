
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
import pandas as pd
from torch import cuda

drive.mount('/content/drive')

def des_classifier(df, model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer= BertTokenizerFast.from_pretrained(model_path)
    device = 'cuda' if cuda.is_available() else 'cpu'
    nlp= pipeline("text-classification", model=model, tokenizer=tokenizer)

    def predict(text):
      model = BertForSequenceClassification.from_pretrained(model_path)
      tokenizer= BertTokenizerFast.from_pretrained(model_path)

      inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")

      # Move the tensors to the same device as the model
      inputs = inputs.to("cuda")

      # Ensure the model is in evaluation mode and on the correct device
      model.eval()
      model.to("cuda")

      # Get model output (logits)
      outputs = model(**inputs)

      probs = outputs[0].softmax(1)

      # Get the index of the class with the highest probability
      # argmax() finds the index of the maximum value in the tensor along a specified dimension.
      # By default, if no dimension is specified, it returns the index of the maximum value in the flattened tensor.
      pred_label_idx = probs.argmax()

      # map the predicted class index to the actual class label
      # Since pred_label_idx is a tensor containing a single value (the predicted class index),
      # the .item() method is used to extract the value as a scalar
      pred_label = model.config.id2label[pred_label_idx.item()]

      return pred_label

    df['Company Li Description with null'] = df['Company Li Description'].copy()
    df['Company Li Description with null'] = df['Company Li Description with null'].fillna('null')
    df['Company Des Relevant Score'] = df['Company Li Description with null'].apply(predict)
    return df


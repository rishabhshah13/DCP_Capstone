from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
import torch

def create_label_mappings(df, label_column_name):
    """
    Creates label mappings for multi-class classification.
    
    Args:
    df (DataFrame): Input dataframe containing label data.
    label_column_name (str): Name of the column containing labels.
    
    Returns:
    DataFrame: DataFrame with numerical labels.
    dict: Mapping of numerical labels to original labels.
    dict: Mapping of original labels to numerical labels.
    int: Number of unique labels.
    """
    labels = df[label_column_name].unique().tolist()
    num_labels = len(labels)
    id2label = {id: label for id, label in enumerate(labels)}
    label2id = {label: id for id, label in enumerate(labels)}

    # Add a new column to the DataFrame with numerical labels
    df["labels"] = df[label_column_name].map(lambda x: label2id[x.strip()])

    return df, id2label, label2id, num_labels

def initialize_tokenizer_and_model(model_name, num_labels, id2label, label2id, device):
    """
    Initializes tokenizer and model for training or inference.
    
    Args:
    model_name (str): Name or path of the pre-trained model.
    num_labels (int): Number of unique labels.
    id2label (dict): Mapping of numerical labels to original labels.
    label2id (dict): Mapping of original labels to numerical labels.
    device (str): Device to use for training or inference ('cuda' or 'cpu').
    
    Returns:
    BertTokenizerFast: Initialized tokenizer.
    BertForSequenceClassification: Initialized model.
    """
    # Initialize the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_name, max_length=512)

    # Initialize the model with the given specifications
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Move the model to the specified device
    model.to(device)

    return tokenizer, model

def split_data(df, description_column, label_column, random_state=0):
    """
    Splits the dataset into training, validation, and test sets.
    
    Args:
    df (DataFrame): Input dataframe containing data.
    description_column (str): Name of the column containing text descriptions.
    label_column (str): Name of the column containing labels.
    random_state (int): Random state for reproducibility.
    
    Returns:
    list: Training text descriptions.
    list: Validation text descriptions.
    list: Test text descriptions.
    list: Training labels.
    list: Validation labels.
    list: Test labels.
    """
    # Extracting text descriptions and labels into lists
    data_texts = df[description_column].to_list()
    data_labels = df[label_column].to_list()

    # Splitting the dataset into training+validation and test datasets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data_texts, data_labels, test_size=0.2, random_state=random_state
    )

    # Splitting the training dataset into training and validation datasets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.01, random_state=random_state
    )

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def tokenize_datasets(tokenizer, train_texts, val_texts, test_texts):
    """
    Tokenizes datasets using the provided tokenizer.
    
    Args:
    tokenizer (BertTokenizerFast): Tokenizer object.
    train_texts (list): List of training text descriptions.
    val_texts (list): List of validation text descriptions.
    test_texts (list): List of test text descriptions.
    
    Returns:
    dict: Encodings for training texts.
    dict: Encodings for validation texts.
    dict: Encodings for test texts.
    """
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    return train_encodings, val_encodings, test_encodings

class DataLoader(Dataset):
    """
    Custom PyTorch dataset for loading tokenized data.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieves tokenized data for a given index.
        """
        # Retrieve tokenized data for the given index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add the label for the given index to the item dictionary
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.labels)

def compute_metrics(pred):
    """
    Computes evaluation metrics (accuracy, precision, recall, F1) for a given prediction.
    """
    # Extract true labels from the input object
    labels = pred.label_ids

    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)

    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    acc = accuracy_score(labels, preds)

    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

def setup_and_train(model, train_dataloader, val_dataloader, compute_metrics, training_args=None):
    """
    Sets up training and starts the training process.
    """
    # Set default training arguments 
    training_args = {
            'output_dir': './TTC4900Model',
            'do_train': True,
            'do_eval': True,
            'num_train_epochs': 8,
            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 32,
            'warmup_steps': 100,
            'weight_decay': 0.01,
            'logging_strategy': 'steps',
            'logging_dir': './multi-class-logs',
            'logging_steps': 50,
            'evaluation_strategy': "steps",
            'eval_steps': 50,
            'save_strategy': "steps",
            'fp16': False,
            'load_best_model_at_end': True
        }

    # Initialize TrainingArguments
    training_arguments = TrainingArguments(**training_args)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader,
        compute_metrics=compute_metrics
    )

    # Start training
    trainer.train()

def save_model(model_path):
    """
    Saves the trained model and tokenizer.
    """
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer= BertTokenizerFast.from_pretrained(model_path)
    return model, tokenizer

def predict(text):
    """
    Performs inference on a single text sample.
    """
    # Tokenize the input text and move tensors to the GPU if available
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")

    # Move the tensors to the same device as the model
    inputs = inputs.to("cuda")

    # Ensure the model is in evaluation mode and on the correct device
    model.eval()
    model.to("cuda")

    # Get model output (logits)
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    pred_label_idx = probs.argmax()
    pred_label = model.config.id2label[pred_label_idx.item()]

    return pred_label

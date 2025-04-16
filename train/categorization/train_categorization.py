import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import joblib

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
dataset_path = os.path.join(script_dir, "data", "dataset.csv")
model_save_dir = os.path.join(script_dir, "models")

# Create directories
os.makedirs(model_save_dir, exist_ok=True)

# Load dataset with label encoding
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    
    # Ensure the 'text' column exists and is of type string
    if 'text' not in df.columns:
        raise ValueError("Dataset must contain a 'text' column")
    
    # Convert all text entries to strings and handle NaN/None
    df['text'] = df['text'].fillna('').astype(str)  # Replace NaN with empty strings
    texts = df['text'].tolist()
    
    # Convert labels to integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'])
    
    return texts, labels, label_encoder

# Load data
texts, labels, label_encoder = load_dataset(dataset_path)

# Save label encoder
joblib.dump(label_encoder, os.path.join(model_save_dir, "label_encoder.pkl"))

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_)
)

# Dataset class
class SurveyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = SurveyDataset(train_texts, train_labels, tokenizer)
val_dataset = SurveyDataset(val_texts, val_labels, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir=model_save_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir=os.path.join(script_dir, "logs"),
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train and save
trainer.train()
model.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)
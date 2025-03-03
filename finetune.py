import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch

# Load the dataset
file_path = "leo_at_risk_answers.csv"
data = pd.read_csv(file_path)

# Ensure 'text' is string and 'label' is integer
data["text"] = data["text"].astype(str)
data["label"] = data["label"].astype(int)

# Remap labels to start from 0
label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
data["label"] = data["label"].map(label_mapping)

# Split the data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["text"].tolist(), data["label"].tolist(), test_size=0.2, random_state=42
)

# Explicitly ensure all splits have consistent types
train_texts = [str(text) for text in train_texts]
test_texts = [str(text) for text in test_texts]
train_labels = [int(label) for label in train_labels]
test_labels = [int(label) for label in test_labels]

# Create Hugging Face Datasets
train_data = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_data = Dataset.from_dict({"text": test_texts, "label": test_labels})

# Load tokenizer and preprocess the data
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Set format for PyTorch tensors
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load pre-trained model for sequence classification
num_labels = len(set(data["label"]))
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10_000,
    save_total_limit=2,
)

# Define Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate()
print("Evaluation results:", results)

# Save the fine-tuned model and tokenizer
model.save_pretrained("./distilbert-finetuned-at-risk")
tokenizer.save_pretrained("./distilbert-finetuned-at-risk")


# Function to classify new text
def classify_text(text):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class


# Example usage
#new_text = (
#    "I was unable to use my hotspot while traveling abroad. It was very frustrating."
#)
#predicted_label = classify_text(new_text)
#print(f"Predicted label: {predicted_label}")

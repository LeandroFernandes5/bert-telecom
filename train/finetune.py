import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, Features, ClassLabel, Value
from transformers import (
    DistilBertTokenizerFast, # Using Fast tokenizer for potential speed benefits
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
import os
import glob
import chardet
import argparse
import numpy as np

# --- Configuration ---
# Define your categories. The order determines the integer mapping (0, 1, 2, ...)
CATEGORIES = ["Competitor", "Data", "Network", "NotRoaming", "Positive", "Value", "Others"]
LABEL2ID = {label: i for i, label in enumerate(CATEGORIES)}
ID2LABEL = {i: label for i, label in enumerate(CATEGORIES)}
NUM_LABELS = len(CATEGORIES)

DATA_FOLDER = "datasets"  # Folder containing your CSV files
TEXT_COLUMN = "text"      # Name of the column with survey responses in your CSVs
LABEL_COLUMN = "label"    # Name of the column with topic labels (strings) in your CSVs

MODEL_CHECKPOINT = "distilbert-base-uncased"
OUTPUT_MODEL_DIR = "./distilbert-finetuned-survey-topics"
LOGGING_DIR = "./logs_survey_topics"
RESULTS_DIR = "./results_survey_topics"

# --- Utility Functions ---
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(20000) # Read a larger chunk for better detection
    result = chardet.detect(rawdata)
    return result['encoding'] if result['encoding'] else 'utf-8' # Fallback to utf-8

def load_and_preprocess_data(folder_path, text_col, label_col, first_csv_encoding):
    """Loads data from CSVs, applies label mapping, and checks for issues."""
    dataframes = []
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the folder: {folder_path}")

    print(f"Found {len(csv_files)} CSV files. Attempting to load with encoding: {first_csv_encoding}")

    for i, file_path in enumerate(csv_files):
        try:
            # For more robustness, you could detect encoding for each file here,
            # but for simplicity, we're using the one from the first file.
            df = pd.read_csv(file_path, encoding=first_csv_encoding)
            if text_col not in df.columns or label_col not in df.columns:
                print(f"Warning: File {os.path.basename(file_path)} is missing '{text_col}' or '{label_col}' column. Skipping.")
                continue
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)} with encoding {first_csv_encoding}: {e}. Trying utf-8...")
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                if text_col not in df.columns or label_col not in df.columns:
                    print(f"Warning: File {os.path.basename(file_path)} (utf-8) is missing '{text_col}' or '{label_col}' column. Skipping.")
                    continue
                dataframes.append(df)
            except Exception as e_utf8:
                print(f"Error reading {os.path.basename(file_path)} with utf-8 as well: {e_utf8}. Skipping this file.")
                continue
        
    if not dataframes:
        raise ValueError("No data could be loaded. Check CSV files and column names.")

    data = pd.concat(dataframes, ignore_index=True)
    print(f"Successfully loaded and concatenated data. Total rows: {len(data)}")

    # Ensure text is string and prepare label column
    data[text_col] = data[text_col].astype(str)
    data["label_id"] = data[label_col].map(LABEL2ID) # Map string labels to integer IDs

    # Validate labels
    if data["label_id"].isnull().any():
        unknown_labels = data[data["label_id"].isnull()][label_col].unique()
        print(f"Warning: Found unknown labels not in CATEGORIES: {unknown_labels}. These rows will be dropped.")
        data.dropna(subset=["label_id"], inplace=True)

    data["label_id"] = data["label_id"].astype(int) # Final label column for the model
    
    print("Label distribution after mapping:")
    print(data["label_id"].value_counts().sort_index().rename(index=ID2LABEL))

    return data[[text_col, "label_id"]].rename(columns={text_col: "text", "label_id": "label"})

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- Main Fine-tuning Pipeline ---
def run_finetuning_pipeline():
    print("Starting fine-tuning pipeline...")

    # 1. Detect encoding for loading data
    csv_files_for_ft = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    if not csv_files_for_ft:
        raise FileNotFoundError(f"No CSV files found in {DATA_FOLDER} for fine-tuning.")
    
    # Use encoding of the first file for all.
    # For more robustness, could allow user to specify or detect per file.
    initial_encoding = detect_encoding(csv_files_for_ft[0])
    print(f"Using detected encoding '{initial_encoding}' for loading CSV files for fine-tuning.")

    # 2. Load and preprocess data
    try:
        processed_data = load_and_preprocess_data(DATA_FOLDER, TEXT_COLUMN, LABEL_COLUMN, initial_encoding)
    except Exception as e:
        print(f"Error during data loading and preprocessing: {e}")
        return

    if processed_data.empty:
        print("No data available after preprocessing. Exiting.")
        return
        
    # 3. Split data (stratified)
    train_df, val_df = train_test_split(
        processed_data,
        test_size=0.2, # 20% for validation
        random_state=42,
        stratify=processed_data["label"] # Crucial for imbalanced datasets
    )
    print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")

    # 4. Create Hugging Face Datasets
    # Define features including ClassLabel for proper label handling
    features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=NUM_LABELS, names=CATEGORIES),
        'input_ids': Value(dtype='int32', id=None), # Will be populated by map
        'attention_mask': Value(dtype='int8', id=None) # Will be populated by map
    })

    train_dataset_hf = Dataset.from_pandas(train_df, features=features, preserve_index=False)
    val_dataset_hf = Dataset.from_pandas(val_df, features=features, preserve_index=False)
    
    # 5. Load tokenizer and tokenize data
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    
    train_dataset_tokenized = train_dataset_hf.map(
        lambda examples: tokenize_function(examples, tokenizer), batched=True
    )
    val_dataset_tokenized = val_dataset_hf.map(
        lambda examples: tokenize_function(examples, tokenizer), batched=True
    )

    # Remove original text column after tokenization, set format
    train_dataset_tokenized = train_dataset_tokenized.remove_columns(["text"])
    val_dataset_tokenized = val_dataset_tokenized.remove_columns(["text"])
    train_dataset_tokenized.set_format("torch")
    val_dataset_tokenized.set_format("torch")

    # 6. Load pre-trained model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL, # For better inference later
        label2id=LABEL2ID
    )
    
    # If using GPU, ensure model is on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # 7. Define Training Arguments
    # Note on class weights:
    # To implement class weights, you would typically subclass Trainer and override the compute_loss method.
    # This is where you'd calculate weights (e.g., inversely proportional to class frequency)
    # and apply them to the CrossEntropyLoss. For now, we're proceeding without explicit class weights.
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        evaluation_strategy="epoch",    # Evaluate at the end of each epoch
        save_strategy="epoch",          # Save model at the end of each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # Adjust based on your GPU memory
        per_device_eval_batch_size=8,   # Adjust based on your GPU memory
        num_train_epochs=3,             # Start with 3, can tune later
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        logging_steps=50,               # Log more frequently with smaller datasets
        load_best_model_at_end=True,    # Load the best model (based on metric_for_best_model) at the end
        metric_for_best_model="f1",     # Use F1 score to select the best model
        save_total_limit=2,             # Only keep the best and the latest checkpoints
        report_to="tensorboard"         # Or "wandb", "none"
    )

    # 8. Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=val_dataset_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn # Pass the custom metrics function
    )

    # 9. Train the model
    print("Starting model training...")
    trainer.train()
    print("Training complete.")

    # 10. Evaluate the model (on the validation set, as it's the best model found during training)
    print("Evaluating the best model on the validation set...")
    eval_results = trainer.evaluate() # This will use val_dataset_tokenized
    print("Evaluation results:", eval_results)

    # 11. Save the fine-tuned model and tokenizer
    print(f"Saving the best model and tokenizer to {OUTPUT_MODEL_DIR}...")
    trainer.save_model(OUTPUT_MODEL_DIR) # Saves best model due to load_best_model_at_end=True
    # tokenizer.save_pretrained(OUTPUT_MODEL_DIR) # Trainer already saves it
    print("Model and tokenizer saved.")

    # (Optional) Example of how to classify new text with the fine-tuned model
    # print_sample_predictions(OUTPUT_MODEL_DIR, tokenizer)


# (Optional) Function to demonstrate prediction
def print_sample_predictions(model_dir, tokenizer_ref):
    print("\n--- Sample Predictions ---")
    # Load the fine-tuned model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = tokenizer_ref # or DistilBertTokenizerFast.from_pretrained(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set to evaluation mode

    sample_texts = [
        "Their prices are way too high compared to other providers.", # Competitor/Value
        "My internet speed is terribly slow, I can't stream anything.", # Data/Network
        "Excellent customer service, resolved my issue quickly!", # Positive
        "I had no signal when I traveled to Germany.", # NotRoaming
        "The app is very confusing to use." # Others (or could be Value if it's about product usability)
    ]

    for text in sample_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predicted_label = model.config.id2label[predicted_class_id]
        print(f"Text: \"{text}\"")
        print(f"Predicted Label: {predicted_label} (ID: {predicted_class_id})\n")


# --- Script Execution Control ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for survey topic classification or detect CSV encoding.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--detect-encoding-only",
        action="store_true",
        help="If set, only detect encoding of CSV files and exit."
    )
    group.add_argument(
        "--run-finetune",
        action="store_true",
        help="If set, run the full fine-tuning process."
    )
    args = parser.parse_args()

    if args.detect_encoding_only    :
        print(f"Detecting encoding for all CSV files in '{DATA_FOLDER}' folder:")
        all_csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
        if not all_csv_files:
            print(f"No CSV files found in {DATA_FOLDER}.")
        else:
            for file_p in all_csv_files:
                try:
                    enc = detect_encoding(file_p)
                    print(f"  - Detected encoding for {os.path.basename(file_p)}: {enc}")
                except Exception as e_enc:
                    print(f"  - Error detecting encoding for {os.path.basename(file_p)}: {e_enc}")
        print("Encoding detection process complete. Exiting.")
    
    elif args.run_finetune:
        run_finetuning_pipeline()
        # After fine-tuning, you can optionally run sample predictions:
        # To get the tokenizer used during training, you'd ideally pass it or reload it.
        # For simplicity, let's assume the tokenizer saved with the model is sufficient.
        # A better way would be to return the tokenizer from run_finetuning_pipeline
        # or load it explicitly.
        # print_sample_predictions(OUTPUT_MODEL_DIR, DistilBertTokenizerFast.from_pretrained(OUTPUT_MODEL_DIR))

    print("Script finished.")

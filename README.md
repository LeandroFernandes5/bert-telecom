# Telecom Roaming Experience Sentiment Analysis

This project is designed to analyze the sentiment of customer feedback regarding their international roaming experience. It uses a fine-tuned BERT model to classify customer responses into three categories: **Negative (0)**, **Neutral (1)**, and **Positive (2)**. The project consists of two main components:

1. **Fine-tuning a BERT model** using telecom-specific survey data.
2. **A FastAPI web server** that provides a sentiment analysis API for classifying customer responses.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Fine-tuning the BERT Model](#fine-tuning-the-bert-model)
   - [Running the FastAPI Server](#running-the-fastapi-server)
   - [Making API Requests](#making-api-requests)
4. [Project Structure](#project-structure)
5. [Dependencies](#dependencies)
6. [License](#license)
7. [Contact](#contact)

---

## Project Overview

The goal of this project is to automate the sentiment analysis of customer feedback related to international roaming experiences. The system uses a BERT model fine-tuned on telecom-specific survey data to classify responses into three sentiment categories:

- **0: Negative**
- **1: Neutral**
- **2: Positive**

The project consists of two main components:

1. **`finetune.py`**: A Python script to fine-tune the BERT model using labeled telecom survey data.
2. **`class.py`**: A FastAPI web server that exposes an API for sentiment analysis.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/telecom-sentiment-analysis.git
   cd telecom-sentiment-analysis
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained BERT Model**:
   - Download a pre-trained BERT model (e.g., `bert-base-uncased`) and place it in the `models/` directory.
   - Alternatively, you can use the `transformers` library to download the model automatically during the fine-tuning process.

---

## Usage

### Fine-tuning the BERT Model

1. **Prepare Data**:
   - Place your labeled telecom survey data in a CSV file (e.g., `data/survey_data.csv`).
   - The CSV file should have at least two columns: `Answer` (customer response) and `Label` (sentiment label: 0, 1, or 2).

2. **Run the Fine-tuning Script**:
   ```bash
   python finetune.py --data_path data/survey_data.csv --model_path models/bert-base-uncased --output_dir models/finetuned-bert
   ```

   - This script will fine-tune the BERT model and save the fine-tuned model to the `models/finetuned-bert` directory.

### Running the FastAPI Server

1. **Start the Server**:
   ```bash
   python class.py
   ```

   - The server will start at `http://127.0.0.1:8000`.

2. **Test the Server**:
   - Open your browser and go to `http://127.0.0.1:8000/docs` to access the Swagger UI for the API.

### Making API Requests

You can send a JSON POST request to the `/predict` endpoint to get the sentiment analysis score for a customer response.

#### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "Unique_ID": "12345",
  "Answer": "The roaming experience was excellent, and I had no issues with connectivity."
}'
```

#### Example Response

```json
{
  "Unique_ID": "12345",
  "Sentiment": 2,
  "Sentiment_Label": "Positive"
}
```

---

## Project Structure

```
telecom-sentiment-analysis/
├── data/                   # Directory for survey data
│   └── survey_data.csv     # Labeled telecom survey data
├── models/                 # Directory for pre-trained and fine-tuned models
│   ├── bert-base-uncased/  # Pre-trained BERT model
│   └── finetuned-bert/     # Fine-tuned BERT model
├── finetune.py             # Script for fine-tuning the BERT model
├── class.py                # FastAPI server for sentiment analysis
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

---

## Dependencies

The project relies on the following Python libraries:

- `transformers` (for BERT model)
- `torch` (PyTorch for deep learning)
- `pandas` (for data processing)
- `scikit-learn` (for evaluation metrics)
- `fastapi` (for the web server)
- `uvicorn` (for running the FastAPI server)

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please contact:

- **Your Name**
- **Email**: your.email@example.com
- **GitHub**: [your-username](https://github.com/your-username)

---


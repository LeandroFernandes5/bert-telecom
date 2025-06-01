from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import logging
import sys
from pathlib import Path # Added pathlib

app = FastAPI(title='api')

# Determine the absolute path to the directory where this script is located
script_dir = Path(__file__).resolve().parent

# Load the fine-tuned model and tokenizer for general sentiment
model_path = script_dir / "distilbert-telecom-finetuned"
try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    print(f"Successfully loaded general sentiment model from {model_path}")
except OSError as e:
    print(f"CRITICAL: Failed to load general sentiment model from {model_path}. Error: {e}", file=sys.stderr)
    sys.exit(1)

# Load the fine-tuned model and tokenizer for at-risk prediction
at_risk_model_path = script_dir / "distilbert-finetuned-at-risk" # Path to the at-risk model
try:
    at_risk_tokenizer = DistilBertTokenizer.from_pretrained(at_risk_model_path)
    at_risk_model = DistilBertForSequenceClassification.from_pretrained(at_risk_model_path)
    print(f"Successfully loaded at-risk prediction model from {at_risk_model_path}")
except OSError as e:
    print(f"CRITICAL: Failed to load at-risk prediction model from {at_risk_model_path}. Error: {e}", file=sys.stderr)
    sys.exit(1)


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('./classify.log', mode='a')
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
file_handler.setFormatter(file_formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(file_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Pydantic models
class UserRequest(BaseModel):
  unique_id: int 
  answer: str = Query(..., min_length=1, max_length=4096, description="Text to classify")

class ClassificationResponse(BaseModel):
    sentiment: int
    unique_id: int

# Pydantic models for the new at-risk endpoint
class AtRiskRequest(BaseModel):
  unique_id: int 
  sentiment: str = Query(..., min_length=1, max_length=4096, description="Survey response text to classify for at-risk status")

class AtRiskResponse(BaseModel):
    predicted_status: int # This will be the class predicted by the at-risk model
    unique_id: int

@app.get('/health')
async def health_check():
    return {"status": "healthy"}

# Function to classify text
@app.post('/classify', response_model=ClassificationResponse, 
          summary="Classify Text Sentiment", 
          description="Classify the sentiment of the input text using a fine-tuned DistilBERT model.")
async def classify_text(user_request: UserRequest):
    """
        Classify the sentiment of the input text using a fine-tuned DistilBERT model.
    
        Args:
            user_request (UserRequest): Contains the unique_id and answer text.
    
        Returns:
            ClassificationResponse: Contains the predicted sentiment and unique_id.
    """
    if tokenizer is None or model is None:
        logger.error(f"General sentiment model not loaded. Cannot process request {user_request.unique_id}.")
        raise HTTPException(status_code=503, detail="General sentiment model is not available.")

    try:
        answer = user_request.answer
        unique_id = user_request.unique_id

        # Tokenize the input text
        inputs = tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted class (label)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        logger.info(f'Processing {unique_id} = sentiment {predicted_class}')

        return ClassificationResponse(sentiment=predicted_class, unique_id=unique_id)
    
    except Exception as e:
        logger.error(f"Error processing request {unique_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint for at-risk prediction
@app.post('/predict_at_risk', response_model=AtRiskResponse,
          summary="Predict At-Risk Status",
          description="Predict if a survey response indicates an at-risk sentiment using a specialized model.")
async def predict_at_risk_status(at_risk_request: AtRiskRequest):
    """
        Predict if the sentiment of a survey response indicates an at-risk status.
    
        Args:
            at_risk_request (AtRiskRequest): Contains the unique_id and sentiment text.
    
        Returns:
            AtRiskResponse: Contains the predicted at-risk status and unique_id.
    """
    if at_risk_tokenizer is None or at_risk_model is None:
        logger.error(f"At-risk model not loaded. Cannot process request {at_risk_request.unique_id}.")
        raise HTTPException(status_code=503, detail="At-risk prediction model is not available.")

    try:
        text_to_classify = at_risk_request.sentiment
        unique_id = at_risk_request.unique_id

        # Tokenize the input text using the at-risk tokenizer
        inputs = at_risk_tokenizer(text_to_classify, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Perform inference with the at-risk model
        with torch.no_grad():
            outputs = at_risk_model(**inputs)
        
        # Get predicted class (label)
        logits = outputs.logits
        predicted_status = torch.argmax(logits, dim=1).item()

        logger.info(f'Processing at-risk prediction for {unique_id} = predicted status {predicted_status}')

        return AtRiskResponse(predicted_status=predicted_status, unique_id=unique_id)
    
    except Exception as e:
        logger.error(f"Error processing at-risk request {unique_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI server
if __name__ == "__main__":
    logger.info('API is starting up')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

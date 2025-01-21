from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import sys

app = FastAPI(title='api')

# Load the fine-tuned model and tokenizer
#model_path = "./distilbert-finetuned"  # Replace with the path where your model is saved
model_path = "./distilbert-telecom-finetuned"

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"}
    )


# Pydantic models
class UserRequest(BaseModel):
  unique_id: int 
  answer: str = Query(..., min_length=1, max_length=4096, description="Text to classify")

class ClassificationResponse(BaseModel):
    sentiment: int
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


# Run the FastAPI server
if __name__ == "__main__":
    logger.info('API is starting up')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

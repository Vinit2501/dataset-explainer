from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI()

# Configure the Gemini API key (You will add this in Vercel's dashboard)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class DataPayload(BaseModel):
    csv_snippet: str

@app.post("/api/analyze")
def analyze_dataset(payload: DataPayload):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are an expert Data Science assistant. Analyze the following raw CSV snippet.
        1. Classify the likely data types of the columns.
        2. Suggest one specific machine learning model (e.g., Logistic Regression, Random Forest) that would be well-suited for predicting a target variable in this dataset.
        3. Keep the response under 4 sentences. 
        
        Data:
        {payload.csv_snippet}
        """
        
        response = model.generate_content(prompt)
        return {"analysis": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

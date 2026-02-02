import os
import io
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from tavily import TavilyClient
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "success", "message": "Render is working!"}

# 1. Initialize API Clients
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

app = FastAPI()
genai_client = genai.Client(api_key=GEMINI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

class SnapFlipResponse(BaseModel):
    brand: str
    item_name: str
    suggested_price: str
    condition: str
    description: str
    seo_title: str

@app.post("/process", response_model=SnapFlipResponse)
async def process_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        # --- PHASE 1: Vision AI ---
        prompt = "Analyze this item. Return JSON with: brand, item_name, condition (1-10), seo_title (80 chars), and 3-bullet description."
        
        vision_response = genai_client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        item_info = json.loads(vision_response.text)
        
        # --- PHASE 2: Pricing ---
        search_query = f"recent sold price for {item_info.get('brand')} {item_info.get('item_name')}"
        search_results = tavily_client.search(query=search_query, max_results=3)
        
        price_prompt = f"Based on these results: {search_results}, what is the average SOLD price? Return ONLY the number."
        price_response = genai_client.models.generate_content(model="gemini-2.0-flash", contents=price_prompt)
        
        return SnapFlipResponse(
            brand=item_info.get("brand", "Unknown"),
            item_name=item_info.get("item_name", "Unknown"),
            suggested_price=price_response.text.strip(),
            condition=str(item_info.get("condition", "N/A")),
            description=item_info.get("description", ""),
            seo_title=item_info.get("seo_title", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

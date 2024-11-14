from fastapi import FastAPI
from transformers import MarianMTModel, MarianTokenizer
from pydantic import BaseModel

app = FastAPI()

# Charger le modèle à partir de Hugging Face au démarrage
model_name = "Helsinki-NLP/opus-mt-en-fr"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

class TextToTranslate(BaseModel):
    text: str

@app.post("/translate")
async def translate_text(request: TextToTranslate):
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    return {"translation": translation}

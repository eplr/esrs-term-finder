import os
import boto3
from fastapi import FastAPI
from transformers import MarianMTModel, MarianTokenizer
from pydantic import BaseModel

app = FastAPI()

# Configuration Bucketeer
BUCKET_NAME = os.getenv("BUCKETEER_BUCKET_NAME")
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("BUCKETEER_AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("BUCKETEER_AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("BUCKETEER_AWS_REGION")
)

# Chemin du modèle
model_dir = "./model"
model_name = "fine_tuned_model"

# Télécharger le modèle depuis Bucketeer si non présent
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

for file_name in ["pytorch_model.bin", "config.json", "tokenizer_config.json", "vocab.json", "special_tokens_map.json"]:
    s3.download_file(BUCKET_NAME, f"{model_name}/{file_name}", os.path.join(model_dir, file_name))

# Charger le modèle
model = MarianMTModel.from_pretrained(model_dir)
tokenizer = MarianTokenizer.from_pretrained(model_dir)

class TextToTranslate(BaseModel):
    text: str

@app.post("/translate")
async def translate_text(request: TextToTranslate):
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    return {"translation": translation}

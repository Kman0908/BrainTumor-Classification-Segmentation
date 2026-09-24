from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from backend.app.inference import InferenceEngine

app = FastAPI(
    title = 'Brain Tumor Classification and Segmentation API',
    version = '1.0.0'
)

engine = InferenceEngine()

@app.get('/')
def root():
    return {'message': 'API running'}

@app.get('/health')
def health():
    return {'status': 'healthy'}

@app.post('/classify')
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(BytesIO(contents)).convert('RGB')
        result = engine.classify(image)

        return result
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))

@app.post('/segment')
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(BytesIO(contents)).convert('RGB')
        mask = engine.segment(image)

        buffer = BytesIO()
        mask.save(buffer, format = 'PNG')
        buffer.seek(0)

        return StreamingResponse(buffer, media_type = 'image/png')
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
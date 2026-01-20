from fastapi import FastAPI, Header, HTTPException

app = FastAPI()

API_KEY = "Hostel_Fight_Detection"

@app.post("/detect-face")
async def detect_face(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return {"status": "ok"}

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import google.generativeai as genai
from ml_predict_gemini import predict_iris_gen

app = FastAPI()
cors_origins = ["http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"message": "Resource not found."})

@app.get("/")
async def root():
    with open("index.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    sepal_l = data.get("sepal_l")
    sepal_w = data.get("sepal_w")
    petal_l = data.get("petal_l")
    petal_w = data.get("petal_w")
    genai.configure(api_key='')
    model = genai.GenerativeModel("gemini-1.5-flash")
    result = predict_iris_gen(sepal_l,sepal_w,petal_l,petal_w,model)
    return {"species": result}
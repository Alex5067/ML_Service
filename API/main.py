import logging
import os
import uvicorn
import fastapi
import fastapi.middleware.cors as cors
from fastapi import Request
from typing import Any
from fastapi.responses import HTMLResponse
from model import Model


# Create FastAPI instance
app = fastapi.FastAPI()
logging.basicConfig(filename="log.log", filemode="w",level=logging.INFO)

model = Model()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

# Add CORS middleware
app.add_middleware(cors.CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

# Root
@app.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )
    return HTMLResponse(content=body)

@app.post("/classify", status_code=200)
def get_class(text: str | None = None):
    try:
        result = model.predict(str(text))
        logging.info(f"For text: {text}\n"
                     f"Predicted: {result}")
        return result
    except Exception as e:
        logging.info(e)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

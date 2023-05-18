import uvicorn
from fastapi import FastAPI
from routers import classifier_router, speech_response_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:4200"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )
app.include_router(classifier_router.router)
app.include_router(speech_response_router.router)

if __name__ == "__main__":
    config = uvicorn.Config("app:app", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

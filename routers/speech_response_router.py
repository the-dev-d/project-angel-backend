from fastapi import APIRouter
from services.speech import SpeechResponseEngine
from pydantic import BaseModel
from typing import List

router = APIRouter(
    prefix="/bot"
)


class BotRequestModel(BaseModel):
    query: str

class BotWordTokenModel(BaseModel):
    tokens: List[str]


responseEngine = SpeechResponseEngine()


@router.post("/")
def sendResponse(request: BotRequestModel):
    is_fallback, text, b_audio = responseEngine.get_response(query=request.query)
    return {"isFallback": is_fallback, "responseText": text, "binaryAudio": b_audio}

@router.post("/sentence")
def sendResponse(request: BotWordTokenModel):
    status, text, b_audio = responseEngine.make_sentence(request.tokens)
    return {"status": status, "responseText": text, "binaryAudio": b_audio}

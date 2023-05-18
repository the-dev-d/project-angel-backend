from fastapi import APIRouter, UploadFile
from services.classifier import Classifier
from pydantic import BaseModel
from typing import List

router = APIRouter(
    prefix="/classifier"
)
classifier_model = Classifier()


class ClassificationRequestModel(BaseModel):
    data_url: str


class ClassificationAddRequestModel(BaseModel):
    label: str
    dataset: List[str]


@router.post("/")
async def classify(req: ClassificationRequestModel):
    status, data, message = await classifier_model.predict_label(req.data_url)
    return {"data": data, "status": status, "message": message}


@router.get("/")
async def classify():
    status = await classifier_model.on_training;
    return {"status": status}


@router.post("/add")
async def add_label(req: ClassificationAddRequestModel):
    status = Classifier.add_to_model(req.label, req.dataset)
    return {"status": status}


@router.get("/retrain")
async def add_label():
    status = classifier_model.retrain_model()
    return {"status": status}

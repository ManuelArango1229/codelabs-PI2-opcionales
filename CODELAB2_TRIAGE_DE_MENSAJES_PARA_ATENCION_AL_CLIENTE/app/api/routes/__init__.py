from fastapi import APIRouter

from . import triage

router = APIRouter()

router.include_router(triage.router)
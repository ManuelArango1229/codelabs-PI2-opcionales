from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.security import verify_api_key
from app.services.triage.handle_message_triage import handle_message_triage

router = APIRouter()

@router.post("/message", dependencies= [Depends(verify_api_key)])
async def triage_message(request: Request):
    try:
        body = await request.json()
    except Exception as err:
        raise HTTPException(status_code=400, detail="El cuerpo de la solicitud no es un JSON válido") from err
    
    if not body:
        raise HTTPException(status_code=400, detail="El cuerpo de la solicitud no puede estar vacío")

    return await handle_message_triage(body)
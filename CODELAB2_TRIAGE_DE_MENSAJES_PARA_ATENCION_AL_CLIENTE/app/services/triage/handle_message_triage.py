from fastapi.responses import JSONResponse
import joblib

async def handle_message_triage(body):
    try:
        loaded = joblib.load("pipeline_triage.joblib")
        
        texto = body.get("mensaje")
        if not texto:
            return JSONResponse(
                content={"error": True, "message": "Falta el campo 'mensaje'"},
                status_code=400
            )
            
        prediction = loaded.predict([texto])[0]


        return JSONResponse(content={"message": "OK", "prediction": prediction})
        
    except Exception as e:
        return JSONResponse(content={"error": True, "message": f"Error interno: {str(e)}"})
from fastapi import FastAPI

app = FastAPI(title="HealthAI - Service IA")

@app.get("/")
def read_root():
    return {"status": "IA Service Online", "version": "1.0"}

# Route d'exemple pour l'analyse de repas demandée [cite: 84, 86]
@app.post("/analyze-meal")
async def analyze_meal():
    return {"message": "Analyse en cours..."}
from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def get_models():
    return {"message": "Models endpoint - implementation pending"}

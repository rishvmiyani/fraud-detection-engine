from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

# Create FastAPI app for frontend
frontend_app = FastAPI(title="Fraud Detection Frontend")

# Mount static files
frontend_app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Templates
templates = Jinja2Templates(directory="frontend/templates")


@frontend_app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@frontend_app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


if __name__ == "__main__":
    print("🌐 Starting Fraud Detection Frontend Server...")
    print("📊 Dashboard: http://localhost:3001")
    print("📖 API Docs: http://localhost:8000/docs")
    uvicorn.run(frontend_app, host="0.0.0.0", port=3001)

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(
    title="DoomRL Tactical AI Console API",
    description="Backend API serving static metrics and streaming gameplay footage."
)

# Define scenario metadata models
class ScenarioMetadata(BaseModel):
    id: str
    title: str
    difficulty: int
    episodes: int
    final_mean_reward: float
    description: str

# 1. Mount static folders for Plotly charts and videos
app.mount("/charts", StaticFiles(directory="frontend/charts"), name="charts")
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

# 2. Startup Verification Check
@app.on_event("startup")
def verify_system_assets():
    print("=" * 60)
    print(" DoomRL FastAPI Server Startup diagnostics")
    print("=" * 60)
    
    # Check model weights
    checkpoints = {
        "Basic Scenario": "models/basic/dqn_basic_best.pth",
        "Defend Center": "models/defend_the_center/dqn_defend_best.pth",
        "Deadly Corridor": "models/deadly_corridor/dqn_corridor_best.pth"
    }
    
    for name, path in checkpoints.items():
        exists = os.path.exists(path)
        status = "[OK] Found" if exists else "[WARNING] Missing"
        print(f"  {name:20} -> {path:50} {status}")
        
    # Check metrics logs
    metrics = {
        "Basic metrics": "logs/json_exports/basic_metrics.json",
        "Defend Center metrics": "logs/json_exports/defend_the_center_metrics.json",
        "Deadly Corridor metrics": "logs/json_exports/deadly_corridor_metrics.json"
    }
    
    for name, path in metrics.items():
        exists = os.path.exists(path)
        status = "[OK] Found" if exists else "[WARNING] Missing"
        print(f"  {name:20} -> {path:50} {status}")
        
    print("=" * 60)

# ── API ENDPOINTS ──────────────────────────────────────────────────────

# Endpoint to return trained scenarios list with metadata
@app.get("/api/scenarios", response_model=List[ScenarioMetadata])
def get_scenarios():
    return [
        ScenarioMetadata(
            id="basic",
            title="Basic",
            difficulty=1,
            episodes=1000,
            final_mean_reward=-11.1,
            description="Single enemy, static placement. AI learns to turn and shoot."
        ),
        ScenarioMetadata(
            id="defend_the_center",
            title="Defend the Center",
            difficulty=3,
            episodes=2000,
            final_mean_reward=6.9,
            description="Enemies spawn perimeter-wise. Agent rotates 360 degrees to defend center."
        ),
        ScenarioMetadata(
            id="deadly_corridor",
            title="Deadly Corridor",
            difficulty=5,
            episodes=3000,
            final_mean_reward=-127.4,
            description="Hallway navigation with flanking fire. Exposes composite movement and GPS rewards."
        )
    ]

# Endpoint to return metrics JSON logs for Plotly integration
@app.get("/api/stats/{scenario}")
def get_scenario_stats(scenario: str):
    # Sanitize inputs
    allowed_scenarios = ["basic", "defend_the_center", "deadly_corridor"]
    if scenario not in allowed_scenarios:
        raise HTTPException(status_code=404, detail="Scenario not found.")
        
    metrics_file = f"logs/json_exports/{scenario}_metrics.json"
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail=f"Metrics file for {scenario} is missing on disk.")
        
    import json
    with open(metrics_file, "r") as f:
        return json.load(f)

# Endpoint to serve dynamic gameplay videos
@app.get("/api/video/{scenario}/{episode}")
def serve_video(scenario: str, episode: str):
    # Map requested names to actual video files
    video_map = {
        ("basic", "trained"): "videos/basic_trained.mp4",
        ("basic", "random"): "videos/basic_random.mp4",
        ("defend_the_center", "trained"): "videos/defend_trained.mp4",
        ("defend_the_center", "random"): "videos/defend_random.mp4",
        ("deadly_corridor", "trained"): "videos/corridor_trained.mp4",
        ("deadly_corridor", "random"): "videos/corridor_random.mp4"
    }
    
    key = (scenario, episode)
    if key not in video_map:
        raise HTTPException(status_code=404, detail="Requested video mapping not found.")
        
    video_path = video_map[key]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file {video_path} not found on disk.")
        
    return FileResponse(video_path, media_type="video/mp4")

# Mount frontend folder at root (/) as the final catch-all fallback
# This must remain at the bottom so it doesn't mask /api or other static mounts.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

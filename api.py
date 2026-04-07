import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from models import EmailTriageAction
from server.email_triage_environment import EmailTriageEnvironment

app = FastAPI(title="Email Triage Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailTriageEnvironment()
METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")

class EvaluationRequest(BaseModel):
    email_text: str
    ground_truth_category: str
    ground_truth_action: str
    ground_truth_resolution: str
    groq_api_key: str = ""

class MetricsResponse(BaseModel):
    total_episodes: int
    total_score: float
    history: list

@app.get("/metrics")
def get_metrics():
    if not os.path.exists(METRICS_FILE):
        return {"total_episodes": 0, "total_score": 0.0, "history": []}
    with open(METRICS_FILE, "r") as f:
        return json.load(f)

@app.post("/evaluate")
def evaluate_email(req: EvaluationRequest):
    token = req.groq_api_key if req.groq_api_key else os.environ.get("GROQ_API_KEY", "")
    if not token or not token.startswith("gsk_"):
        return {"error": "Missing or invalid GROQ_API_KEY. Please provide it in the UI or export it as an environment variable."}
        
    client = OpenAI(api_key=token, base_url="https://api.groq.com/openai/v1")
    
    # Reset Environment and override evaluating parameters
    env.reset()
    env._current_email = {
        "text": req.email_text,
        "category": req.ground_truth_category.lower(),
        "ideal_action": req.ground_truth_action.lower(),
        "ideal_resolution": req.ground_truth_resolution.lower()
    }
    
    # Brain Agent Prompt
    system_prompt = (
        "You are an expert AI Email Triage agent. Read the email and output exactly a valid JSON map with three keys:\n"
        "1. 'category' (Options: 'spam', 'urgent', 'standard')\n"
        "2. 'action_type' (Options: 'delete', 'reply', 'forward', 'categorize')\n"
        "3. 'resolution' (Options: 'sales', 'support', 'it', 'none')\n"
        "If forward, specify the department in resolution. Output ONLY raw JSON, no markdown blocks, no words."
    )
    user_prompt = f"Email: {req.email_text}"
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=64
        )
        content = response.choices[0].message.content.strip()
        # Parse logic
        start = content.find('{')
        end = content.rfind('}') + 1
        guess_json = json.loads(content[start:end])
        
        guess_category = guess_json.get("category", "standard").lower()
        guess_action = guess_json.get("action_type", "categorize").lower()
        guess_res = guess_json.get("resolution", "none").lower()
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        guess_category, guess_action, guess_res = "standard", "categorize", "none"
        
    action = EmailTriageAction(category=guess_category, action_type=guess_action, resolution=guess_res)
    obs = env.step(action)
    
    # Track metrics
    metrics_data = {"total_episodes": 0, "total_score": 0.0, "history": []}
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            metrics_data = json.load(f)
            
    metrics_data["total_episodes"] += 1
    metrics_data["total_score"] += float(obs.reward)
    metrics_data["history"].append({
        "email_text": req.email_text,
        "reward": float(obs.reward)
    })
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics_data, f)
    
    return {
        "guess_category": guess_category,
        "guess_action": guess_action,
        "guess_resolution": guess_res,
        "reward": obs.reward,
        "feedback": obs.feedback
    }

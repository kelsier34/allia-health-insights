from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncpraw
import os
import asyncio
import json
from transformers import pipeline

app = FastAPI()

# Enable CORS to allow the frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; update to your Replit URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Reddit client with environment variables
reddit = asyncpraw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="allia-health-insights/0.1",
)

# Set up emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
)

# Emotion hierarchy for sub-emotions
EMOTION_HIERARCHY = {
    "sadness": ["hopelessness", "guilt", "shame"],
    "fear": ["anxiety", "panic", "worry"],
    "anger": ["frustration", "irritation"],
    "disgust": ["revulsion"],
    "joy": ["happiness", "relief"],
    "surprise": ["shock", "amazement"],
    "neutral": ["fatigue", "dissociation"]
}

# Generate a summary based on emotions
def generate_summary(text, primary_emotion, sub_emotions):
    return f"The user appears to be experiencing {primary_emotion.lower()} with elements of {', '.join(sub_emotions).lower()}, possibly indicating underlying mental health challenges."

# Analyze subreddit posts
async def analyze_subreddit(subreddit):
    results = []
    async with reddit:
        subreddit_obj = await reddit.subreddit(subreddit)
        async for post in subreddit_obj.new(limit=20):
            if not post.selftext:
                continue
            text = post.selftext[:512]  # Limit text length
            emotion_scores = emotion_classifier(text)[0]
            primary_emotion = max(emotion_scores, key=lambda x: x['score'])['label']
            primary_confidence = max(emotion_scores, key=lambda x: x['score'])['score'] * 100
            sub_emotions = EMOTION_HIERARCHY.get(primary_emotion, [])
            summary = generate_summary(text, primary_emotion, sub_emotions)
            results.append({
                "text": text,
                "primary_emotion": primary_emotion.capitalize(),
                "sub_emotions": [e.capitalize() for e in sub_emotions],
                "confidence": round(primary_confidence, 2),
                "summary": summary
            })
    return json.dumps(results)

# Define the API endpoint
@app.get("/analyze/{subreddit}")
async def get_analysis(subreddit: str):
    return {"data": await analyze_subreddit(subreddit)}

# Run the app on Replit's port
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

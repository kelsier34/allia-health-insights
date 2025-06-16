from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncpraw
import torch
from transformers import pipeline
import asyncio
import os
from typing import List, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Reddit client
reddit = asyncpraw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="allia-health-insights/0.1",
)

# Initialize emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

# Emotion mapping for sub-emotions
EMOTION_HIERARCHY = {
    "sadness": ["hopelessness", "guilt", "shame"],
    "fear": ["anxiety", "panic", "worry"],
    "anger": ["frustration", "irritation"],
    "disgust": ["revulsion"],
    "joy": ["happiness", "relief"],
    "surprise": ["shock", "amazement"],
    "neutral": ["fatigue", "dissociation"]
}

async def generate_summary(text: str, primary_emotion: str, sub_emotions: List[str]) -> str:
    return f"The user appears to be experiencing {primary_emotion.lower()} with elements of {', '.join(sub_emotions).lower()}, possibly indicating underlying mental health challenges."

@app.get("/analyze/{subreddit}")
async def analyze_subreddit(subreddit: str) -> List[Dict]:
    results = []
    async with reddit:
        subreddit_obj = await reddit.subreddit(subreddit)
        async for post in subreddit_obj.new(limit=20):
            if not post.selftext:
                continue
            text = post.selftext[:512]  # Truncate for performance
            # Run emotion classification
            emotion_scores = emotion_classifier(text)[0]
            # Get primary emotion
            primary_emotion = max(emotion_scores, key=lambda x: x['score'])['label']
            primary_confidence = max(emotion_scores, key=lambda x: x['score'])['score'] * 100
            # Get sub-emotions
            sub_emotions = EMOTION_HIERARCHY.get(primary_emotion, [])
            summary = await generate_summary(text, primary_emotion, sub_emotions)
            results.append({
                "text": text,
                "primary_emotion": primary_emotion.capitalize(),
                "sub_emotions": [e.capitalize() for e in sub_emotions],
                "confidence": round(primary_confidence, 2),
                "summary": summary
            })
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

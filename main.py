import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bson import ObjectId

# Local database utilities
from database import db, create_document, get_documents

# Schemas for reference
from schemas import Topic as TopicSchema, Conversation as ConversationSchema, Suggestion as SuggestionSchema

# Optional OpenAI client (LLM classification)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_LLM_DEFAULT = os.getenv("USE_LLM", "true").lower() in {"1", "true", "yes", "on"}

try:
    if OPENAI_API_KEY:
        from openai import OpenAI  # type: ignore
        _openai_client = OpenAI()
    else:
        _openai_client = None
except Exception:
    _openai_client = None

app = FastAPI(title="Adaptive Topic Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Utility helpers
# -----------------------
STOPWORDS = set(
    "a an the is are was were be been being i you he she it we they them our your my of for in on at by to from with without and or but so if then as about into over under after before this that these those have has had do did doing can could will would should may might not just more most less least also too very really get got make makes made need needs needed see seen think thought say said ask asked tell told want wanted help helped issue issues problem problems order orders shipping delivery refund return account password payment card charge subscription cancel cancellation bug feature request support agent wait time delay delayed late broken damaged missing lost".split()
)

def oid_str(value: Any) -> str:
    return str(value) if isinstance(value, (ObjectId, str)) else str(value)


def list_topics() -> List[Dict[str, Any]]:
    items = db["topic"].find({}).sort("name", 1)
    return [{"id": oid_str(x.get("_id")), "name": x.get("name"), "parent": x.get("parent"), "description": x.get("description")} for x in items]


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    words = [w.strip(".,!?()[]{}:;\"'`).-/\\").lower() for w in text.split()]
    freq: Dict[str, int] = {}
    for w in words:
        if not w or w in STOPWORDS or any(ch.isdigit() for ch in w):
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_n]]


def simple_classify(transcript: str) -> Dict[str, Optional[str]]:
    """Heuristic classifier:
    - Score topics/subtopics if their names appear in text or keywords are close
    - Prefer subtopic match; fall back to topic match
    Returns dict with topic, subtopic, score
    """
    text = transcript.lower()
    kws = extract_keywords(transcript, top_n=8)
    topics = list(db["topic"].find({}))

    best = {"topic": None, "subtopic": None, "score": 0.0}
    for t in topics:
        name = (t.get("name") or "").lower()
        parent = (t.get("parent") or None)
        score = 0
        # direct name mention
        if name and name in text:
            score += 3
        # token overlap
        parts = [p for p in name.split() if p]
        score += sum(1 for p in parts if p in kws)
        # boost if shipping/order/payment common domains
        domain_boost = 1 if any(d in name for d in ["shipping", "order", "payment", "refund", "account", "delivery", "billing", "login"]) else 0
        score += domain_boost

        if score > best["score"]:
            if parent:
                best = {"topic": parent, "subtopic": t.get("name"), "score": float(score)}
            else:
                best = {"topic": t.get("name"), "subtopic": None, "score": float(score)}

    return best


def llm_classify(transcript: str) -> Dict[str, Any]:
    """Use an LLM to classify transcript to an existing topic/subtopic.
    Falls back to heuristic if LLM unavailable or errors.
    Returns: {topic: Optional[str], subtopic: Optional[str], confidence: float, score: float, suggested_name: Optional[str], reason: Optional[str], used_llm: bool}
    """
    # If client not available, fallback
    if _openai_client is None:
        h = simple_classify(transcript)
        return {"topic": h.get("topic"), "subtopic": h.get("subtopic"), "confidence": 0.0, "score": float(h.get("score", 0)), "suggested_name": None, "reason": "LLM unavailable; used heuristic.", "used_llm": False}

    # Build knowledge of topics tree
    topics = list(db["topic"].find({}).sort("name", 1))
    topic_tree: Dict[str, List[str]] = {}
    for t in topics:
        name = t.get("name")
        parent = t.get("parent")
        if parent:
            topic_tree.setdefault(parent, []).append(name)
        else:
            topic_tree.setdefault(name, [])
    # Limit to avoid overly long prompts
    MAX_PARENTS = 50
    MAX_CHILDREN = 25
    trimmed = {}
    for i, (k, v) in enumerate(topic_tree.items()):
        if i >= MAX_PARENTS:
            break
        trimmed[k] = v[:MAX_CHILDREN]

    system_prompt = (
        "You are a precise classifier for customer support conversations. "
        "Choose the best matching topic and optional subtopic from the provided catalog. "
        "If no adequate match exists, return nulls and propose a concise suggested_name (max 3 words). "
        "Respond ONLY with JSON that matches the schema."
    )

    schema_example = {
        "topic": "<string or null>",
        "subtopic": "<string or null>",
        "confidence": 0.0,
        "suggested_name": "<string or null>",
        "reason": "<brief rationale>"
    }

    user_content = {
        "catalog": trimmed,
        "transcript": transcript,
        "instructions": "Return highest-likelihood topic/subtopic if a clear match (confidence 0-1). If not clear, return both as null and propose suggested_name."
    }

    try:
        completion = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_content)}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        topic = data.get("topic")
        subtopic = data.get("subtopic")
        confidence = float(data.get("confidence", 0.0))
        suggested_name = data.get("suggested_name")
        reason = data.get("reason")
        # Provide a simple numeric score aligned with heuristic semantics (0-5)
        score = max(0.0, min(5.0, confidence * 5.0))
        return {
            "topic": topic,
            "subtopic": subtopic,
            "confidence": confidence,
            "score": score,
            "suggested_name": suggested_name,
            "reason": reason,
            "used_llm": True,
        }
    except Exception as e:
        h = simple_classify(transcript)
        return {"topic": h.get("topic"), "subtopic": h.get("subtopic"), "confidence": 0.0, "score": float(h.get("score", 0)), "suggested_name": None, "reason": f"LLM error: {str(e)[:100]}; used heuristic.", "used_llm": False}


# -----------------------
# Pydantic request/response models
# -----------------------
class TopicIn(BaseModel):
    name: str
    parent: Optional[str] = None
    description: Optional[str] = None

class MergeRequest(BaseModel):
    source: str = Field(..., description="Topic to merge FROM (will be merged into 'target')")
    target: str = Field(..., description="Topic to merge INTO")

class AnalyzeRequest(BaseModel):
    transcript: str
    use_llm: Optional[bool] = Field(None, description="Override to force LLM on/off. Defaults from env USE_LLM and availability.")

class SuggestionDecision(BaseModel):
    parent: Optional[str] = None


# -----------------------
# Basic routes
# -----------------------
@app.get("/")
def root():
    return {"message": "Adaptive Topic Explorer API"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
        "llm_available": bool(_openai_client is not None),
        "llm_model": OPENAI_MODEL if _openai_client is not None else None,
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = os.getenv("DATABASE_NAME") or "❌ Not Set"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
                response["connection_status"] = "Connected"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# -----------------------
# Schema metadata route (lightweight)
# -----------------------
@app.get("/schema")
def schema():
    return {
        "topic": TopicSchema.model_json_schema(),
        "conversation": ConversationSchema.model_json_schema(),
        "suggestion": SuggestionSchema.model_json_schema(),
    }


# -----------------------
# Topic management
# -----------------------
@app.get("/api/topics")
def get_topics():
    return list_topics()

@app.post("/api/topics")
def create_topic(payload: TopicIn):
    exists = db["topic"].find_one({"name": payload.name})
    if exists:
        raise HTTPException(status_code=409, detail="Topic with this name already exists")
    create_document("topic", payload.model_dump())
    return {"ok": True}

@app.post("/api/topics/merge")
def merge_topics(payload: MergeRequest):
    if payload.source == payload.target:
        raise HTTPException(status_code=400, detail="Source and target must be different")

    src = db["topic"].find_one({"name": payload.source})
    tgt = db["topic"].find_one({"name": payload.target})
    if not src or not tgt:
        raise HTTPException(status_code=404, detail="Source or target topic not found")

    # Repoint subtopics whose parent is source -> target
    db["topic"].update_many({"parent": payload.source}, {"$set": {"parent": payload.target}})
    # Update conversations assigned to source
    db["conversation"].update_many({"topic": payload.source}, {"$set": {"topic": payload.target}})
    # Mark source as merged (optional soft-flag)
    db["topic"].update_one({"_id": src["_id"]}, {"$set": {"merged_into": payload.target}})
    return {"ok": True}


# -----------------------
# Analyze conversations (LLM-first, heuristic fallback)
# -----------------------
@app.post("/api/conversations/analyze")
def analyze_conversation(req: AnalyzeRequest):
    transcript = req.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is empty")

    use_llm = req.use_llm if req.use_llm is not None else USE_LLM_DEFAULT

    if use_llm and _openai_client is not None:
        result = llm_classify(transcript)
        used_llm = result.get("used_llm", False)
    else:
        h = simple_classify(transcript)
        result = {"topic": h.get("topic"), "subtopic": h.get("subtopic"), "confidence": 0.0, "score": float(h.get("score", 0)), "suggested_name": None, "reason": "LLM disabled or unavailable; used heuristic.", "used_llm": False}
        used_llm = False

    assigned = False

    # Decide assignment
    # If we have a topic selected and either confidence >= 0.6 (LLM) or score >= 3 (heuristic-style)
    if result.get("topic") and ((used_llm and float(result.get("confidence", 0)) >= 0.6) or (not used_llm and float(result.get("score", 0)) >= 3.0)):
        assigned = True
        data = ConversationSchema(
            transcript=transcript,
            topic=result.get("topic"),
            subtopic=result.get("subtopic"),
        )
        conv_id = create_document("conversation", data)
        return {
            "assigned": True,
            "topic": result.get("topic"),
            "subtopic": result.get("subtopic"),
            "conversation_id": conv_id,
            "score": float(result.get("score", 0)),
            "confidence": float(result.get("confidence", 0)),
            "used_llm": used_llm,
        }

    # No confident match: create suggestion
    suggested_name = result.get("suggested_name")
    reason = result.get("reason") or "No strong match found."
    if not suggested_name:
        kws = extract_keywords(transcript, top_n=3)
        suggested_name = " ".join(kws[:2]).title() if kws else "General Inquiry"
        reason = f"No strong match found. Suggested from keywords: {', '.join(kws)}."

    suggestion = SuggestionSchema(name=suggested_name, reason=reason, status="pending")
    suggestion_id = create_document("suggestion", suggestion)

    # Store conversation with pointer to suggestion
    data = ConversationSchema(
        transcript=transcript,
        topic=None,
        subtopic=None,
        suggestion_id=suggestion_id,
    )
    conv_id = create_document("conversation", data)

    return {
        "assigned": False,
        "conversation_id": conv_id,
        "suggestion_id": suggestion_id,
        "suggested_name": suggested_name,
        "reason": reason,
        "score": float(result.get("score", 0)),
        "confidence": float(result.get("confidence", 0)),
        "used_llm": used_llm,
    }


@app.get("/api/conversations")
def list_conversations(limit: int = Query(20, ge=1, le=200)):
    items = db["conversation"].find({}).sort("_id", -1).limit(limit)
    out = []
    for x in items:
        out.append({
            "id": oid_str(x.get("_id")),
            "transcript": x.get("transcript"),
            "topic": x.get("topic"),
            "subtopic": x.get("subtopic"),
            "suggestion_id": oid_str(x.get("suggestion_id")) if x.get("suggestion_id") else None,
            "created_at": x.get("created_at"),
        })
    return out


# -----------------------
# Suggestions workflow
# -----------------------
@app.get("/api/suggestions")
def get_suggestions(status: Optional[str] = Query(None)):
    q: Dict[str, Any] = {}
    if status:
        q["status"] = status
    items = db["suggestion"].find(q).sort("_id", -1)
    return [{
        "id": oid_str(x.get("_id")),
        "name": x.get("name"),
        "parent": x.get("parent"),
        "reason": x.get("reason"),
        "status": x.get("status"),
    } for x in items]


@app.post("/api/suggestions/{suggestion_id}/approve")
def approve_suggestion(suggestion_id: str, payload: SuggestionDecision):
    try:
        _id = ObjectId(suggestion_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid suggestion id")

    s = db["suggestion"].find_one({"_id": _id})
    if not s:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    # Create or upsert topic
    name = s.get("name")
    parent = payload.parent or s.get("parent")
    existing = db["topic"].find_one({"name": name})
    if not existing:
        create_document("topic", TopicIn(name=name, parent=parent).model_dump())

    # Update suggestion status
    db["suggestion"].update_one({"_id": _id}, {"$set": {"status": "approved", "parent": parent}})

    # Link any conversations that referenced this suggestion to the new topic (as topic if no parent, else subtopic)
    if parent:
        db["conversation"].update_many({"suggestion_id": suggestion_id}, {"$set": {"topic": parent, "subtopic": name}})
    else:
        db["conversation"].update_many({"suggestion_id": suggestion_id}, {"$set": {"topic": name, "subtopic": None}})

    return {"ok": True}


@app.post("/api/suggestions/{suggestion_id}/reject")
def reject_suggestion(suggestion_id: str):
    try:
        _id = ObjectId(suggestion_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid suggestion id")

    res = db["suggestion"].update_one({"_id": _id}, {"$set": {"status": "rejected"}})
    if not res.matched_count:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

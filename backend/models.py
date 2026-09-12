import datetime
from typing import Optional, Any
from bson import ObjectId


def _to_str_id(val: Any) -> str:
    if val is None:
        return ""
    return str(val)


class User:
    """Wrapper class around MongoDB user document preserving attribute access."""
    def __init__(self, doc: dict):
        self.doc = doc
        self.id: str = _to_str_id(doc.get("_id", doc.get("id")))
        self.email: str = doc.get("email", "")
        self.hashed_password: str = doc.get("hashed_password", "")
        self.created_at: datetime.datetime = doc.get("created_at", datetime.datetime.utcnow())

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"


class ScanHistory:
    """Wrapper class around MongoDB scan_history document preserving attribute access."""
    def __init__(self, doc: dict):
        self.doc = doc
        self.id: str = _to_str_id(doc.get("_id", doc.get("id")))
        self.user_id: str = _to_str_id(doc.get("user_id"))
        self.identity: str = doc.get("identity", "")
        self.confidence: float = float(doc.get("confidence", 0.0))
        self.input_thumb: Optional[str] = doc.get("input_thumb")
        self.generated_thumb: Optional[str] = doc.get("generated_thumb")
        self.face_quality_score: Optional[float] = doc.get("face_quality_score")
        self.created_at: datetime.datetime = doc.get("created_at", datetime.datetime.utcnow())

    def __repr__(self) -> str:
        return f"<ScanHistory id={self.id} identity={self.identity} confidence={self.confidence}>"
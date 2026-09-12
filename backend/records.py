# backend/records.py
from typing import Optional, Any
from bson import ObjectId


def _to_str_id(val: Any) -> str:
    if val is None:
        return ""
    return str(val)


class PersonRecord:
    """Wrapper class around MongoDB person_records document."""
    def __init__(self, doc: dict):
        self.doc = doc
        self.id: str = _to_str_id(doc.get("_id", doc.get("id")))
        self.identity_label: str = doc.get("identity_label", "")
        self.full_name: Optional[str] = doc.get("full_name")
        self.record_type: Optional[str] = doc.get("record_type")
        self.status: Optional[str] = doc.get("status")
        self.notes: Optional[str] = doc.get("notes")


def lookup_record(db, identity_label: str) -> Optional[PersonRecord]:
    doc = db.person_records.find_one({"identity_label": identity_label})
    return PersonRecord(doc) if doc else None
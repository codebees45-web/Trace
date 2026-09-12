import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ConnectionFailure, ServerSelectionTimeoutError
from config import settings

logger = logging.getLogger("trace.database")

_client = None
_db = None
_db_available = False   # False when MongoDB is not reachable at startup


def init_db():
    """Initialises MongoDB client connection and sets up required indexes.

    Non-fatal: if MongoDB is unreachable the backend still boots and the
    endpoints that don't touch the database (live camera WebSocket,
    /predict/frame, /model-info, /health, …) continue to work normally.
    Endpoints that need the DB return a clean 503.
    """
    global _client, _db, _db_available
    if _db is not None:
        return _db

    kwargs = {}
    if settings.mongodb_username and settings.mongodb_password and "@" not in settings.mongodb_uri:
        kwargs["username"] = settings.mongodb_username
        kwargs["password"] = settings.mongodb_password

    logger.info("Connecting to MongoDB database: %s", settings.mongodb_db_name)
    try:
        _client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=3000,   # give up quickly if Mongo isn't there
            connectTimeoutMS=3000,
            **kwargs,
        )
        # Force an actual connection attempt so we know right away if it works.
        _client.admin.command("ping")
        _db = _client[settings.mongodb_db_name]

        try:
            _db.users.create_index("email", unique=True)
            _db.scan_history.create_index("user_id")
            _db.scan_history.create_index([("created_at", -1)])
            _db.person_records.create_index("identity_label", unique=True)
            _db.gallery_embeddings.create_index("identity", unique=True)
            _db.video_gallery_embeddings.create_index("identity", unique=True)
            logger.info("MongoDB indexes verified.")
        except Exception as e:
            logger.warning("Error verifying MongoDB indexes: %s", e)

        _db_available = True
        logger.info("MongoDB connected successfully.")
    except (ConnectionFailure, ServerSelectionTimeoutError, PyMongoError) as exc:
        logger.warning(
            "MongoDB not available (%s). "
            "Auth, scan history and demo-cache endpoints will return 503. "
            "Live camera, /predict, /predict/frame and /model-info still work.",
            exc,
        )
        _db_available = False
        # Create a dummy stub so FastAPI's Depends(get_db) still resolves
        # (it will raise 503 before touching anything on the DB).
        _db = _DummyDB()
    return _db


class _DummyDB:
    """Stands in for a real pymongo Database when MongoDB is unreachable.
    Any attribute access returns this same stub so callers can do
    db.users.find_one(…) without an AttributeError — they will get None
    or an empty list, which the endpoint code already handles.
    """
    def __getattr__(self, _):
        return self

    def __call__(self, *args, **kwargs):
        return None

    def find_one(self, *a, **k):  return None
    def find(self, *a, **k):      return iter([])
    def insert_one(self, *a, **k): raise RuntimeError("MongoDB not available")
    def update_one(self, *a, **k): raise RuntimeError("MongoDB not available")
    def delete_one(self, *a, **k): raise RuntimeError("MongoDB not available")
    def delete_many(self, *a, **k): raise RuntimeError("MongoDB not available")
    def create_index(self, *a, **k): pass
    def command(self, *a, **k): raise RuntimeError("MongoDB not available")


def is_db_available() -> bool:
    return _db_available


def get_db():
    """FastAPI dependency yielding the MongoDB Database instance."""
    db = init_db()
    yield db


def get_db_client():
    """Returns the MongoClient instance (useful for health checks)."""
    if _client is None:
        init_db()
    return _client
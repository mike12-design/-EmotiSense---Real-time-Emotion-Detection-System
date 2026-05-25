from datetime import datetime
from pathlib import Path

import pytest


pytestmark = pytest.mark.api


BACKEND_ASSETS_DIR = Path(__file__).resolve().parents[2] / "backend" / "assets"


def test_get_daily_mood_returns_placeholder_data_for_existing_user(api_client, db_session):
    from core.models import User

    user = User(username="alice", password_hash="secret", role="user")
    db_session.add(user)
    db_session.commit()

    response = api_client.get("/api/my/daily_mood", params={"username": "alice"})

    assert response.status_code == 200
    assert response.json() == {"6": "happy", "7": "neutral"}


def test_get_daily_mood_returns_empty_list_for_unknown_user(api_client):
    response = api_client.get("/api/my/daily_mood", params={"username": "nobody"})

    assert response.status_code == 200
    assert response.json() == []


def test_get_calendar_moods_merges_face_logs_and_diaries(api_client, db_session):
    from core.models import Diary, EmotionLog, User

    user = User(username="alice", password_hash="secret", role="user")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    db_session.add_all(
        [
            EmotionLog(user_id=user.id, is_stranger=False, emotion="happy", score=0.9, timestamp=datetime(2026, 5, 1, 10, 0, 0)),
            EmotionLog(user_id=user.id, is_stranger=False, emotion="happy", score=0.8, timestamp=datetime(2026, 5, 1, 12, 0, 0)),
            EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=0.6, timestamp=datetime(2026, 5, 2, 9, 0, 0)),
            Diary(user_id=user.id, title="d1", content="今天挺开心", emotion="Neutral", timestamp=datetime(2026, 5, 1, 20, 0, 0)),
            Diary(user_id=user.id, title="d2", content="今天有点低落", emotion="Sad", timestamp=datetime(2026, 5, 3, 20, 0, 0)),
        ]
    )
    db_session.commit()

    response = api_client.get("/api/my/calendar_moods", params={"username": "alice"})

    assert response.status_code == 200
    assert response.json() == {
        "2026-05-01": "neutral",
        "2026-05-02": "sad",
        "2026-05-03": "sad",
    }


def test_get_calendar_moods_returns_empty_object_for_unknown_user(api_client):
    response = api_client.get("/api/my/calendar_moods", params={"username": "nobody"})

    assert response.status_code == 200
    assert response.json() == {}


def test_upload_background_creates_user_background_file(api_client):
    BACKEND_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    created_path = BACKEND_ASSETS_DIR / "bg_alice.jpg"
    if created_path.exists():
        created_path.unlink()

    response = api_client.post(
        "/api/user/upload_background",
        data={"username": "alice"},
        files={"file": ("bg.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {"message": "success", "url": "/assets/bg_alice.jpg"}

    try:
        assert created_path.exists()
        assert created_path.read_bytes() == b"fake-image-bytes"
    finally:
        if created_path.exists():
            created_path.unlink()


def test_delete_background_removes_existing_file(api_client):
    BACKEND_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    created_path = BACKEND_ASSETS_DIR / "bg_alice.jpg"
    created_path.write_bytes(b"fake-image-bytes")

    response = api_client.delete("/api/user/upload_background", params={"username": "alice"})

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "背景已重置为默认"}
    assert not created_path.exists()


def test_delete_background_is_idempotent_when_file_missing(api_client):
    created_path = BACKEND_ASSETS_DIR / "bg_alice.jpg"
    if created_path.exists():
        created_path.unlink()

    response = api_client.delete("/api/user/upload_background", params={"username": "alice"})

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "已经是默认背景"}

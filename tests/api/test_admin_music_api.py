from pathlib import Path

import pytest


pytestmark = pytest.mark.api


ASSETS_MUSIC_DIR = Path(__file__).resolve().parents[2] / "backend" / "assets" / "music"


def test_admin_get_music_returns_global_and_user_scoped_records(api_client, db_session):
    from core.models import MusicLibrary, User

    alice = User(username="alice", password_hash="secret", role="user")
    db_session.add(alice)
    db_session.commit()
    db_session.refresh(alice)

    db_session.add_all(
        [
            MusicLibrary(title="global.mp3", filepath="assets/music/global.mp3", emotion_tag="happy", user_id=None),
            MusicLibrary(title="alice.mp3", filepath="assets/music/alice.mp3", emotion_tag="sad", user_id=alice.id),
        ]
    )
    db_session.commit()

    global_response = api_client.get("/api/admin/music", params={"target_user": "global"})
    scoped_response = api_client.get("/api/admin/music", params={"target_user": "alice"})

    assert global_response.status_code == 200
    assert scoped_response.status_code == 200
    assert [item["title"] for item in global_response.json()] == ["global.mp3"]
    assert [item["title"] for item in scoped_response.json()] == ["alice.mp3"]


def test_admin_delete_music_removes_record_and_file(api_client, db_session):
    from core.models import MusicLibrary

    ASSETS_MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    file_path = ASSETS_MUSIC_DIR / "to-delete.mp3"
    file_path.write_bytes(b"ID3")

    music = MusicLibrary(
        title="to-delete.mp3",
        filepath="assets/music/to-delete.mp3",
        emotion_tag="calming",
        user_id=None,
    )
    db_session.add(music)
    db_session.commit()
    db_session.refresh(music)

    response = api_client.delete(f"/api/admin/music/{music.id}")

    assert response.status_code == 200
    assert response.json() == {"message": "deleted"}
    assert db_session.query(MusicLibrary).filter(MusicLibrary.id == music.id).first() is None
    assert not file_path.exists()


def test_admin_upload_music_creates_global_record_and_file(api_client, db_session, monkeypatch):
    from app import api as api_module
    from core.models import MusicLibrary

    monkeypatch.setattr(api_module.time, "time", lambda: 1234567890)

    response = api_client.post(
        "/api/admin/upload_music",
        data={"emotion": "happy", "target_user": "global"},
        files={"file": ("calm.mp3", b"ID3", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "success",
        "filename": "global_happy_1234567890.mp3",
    }

    music = db_session.query(MusicLibrary).one()
    created_path = Path(__file__).resolve().parents[2] / "backend" / music.filepath

    try:
        assert music.title == "calm.mp3"
        assert music.emotion_tag == "happy"
        assert music.user_id is None
        assert music.filepath == "assets/music/global_happy_1234567890.mp3"
        assert created_path.exists()
    finally:
        if created_path.exists():
            created_path.unlink()


def test_admin_upload_music_assigns_user_scoped_record(api_client, db_session, monkeypatch):
    from app import api as api_module
    from core.models import MusicLibrary, User

    alice = User(username="alice", password_hash="secret", role="user")
    db_session.add(alice)
    db_session.commit()
    db_session.refresh(alice)

    monkeypatch.setattr(api_module.time, "time", lambda: 1234567891)

    response = api_client.post(
        "/api/admin/upload_music",
        data={"emotion": "sad", "target_user": "alice"},
        files={"file": ("comfort.mp3", b"ID3", "audio/mpeg")},
    )

    assert response.status_code == 200

    music = db_session.query(MusicLibrary).one()
    created_path = Path(__file__).resolve().parents[2] / "backend" / music.filepath

    try:
        assert music.title == "comfort.mp3"
        assert music.emotion_tag == "sad"
        assert music.user_id == alice.id
        assert music.filepath == "assets/music/alice_sad_1234567891.mp3"
        assert created_path.exists()
    finally:
        if created_path.exists():
            created_path.unlink()

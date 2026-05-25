from pathlib import Path

import pytest


pytestmark = pytest.mark.api


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_get_user_music_returns_only_user_records(api_client, db_session):
    from core.models import MusicLibrary, User

    alice = User(username="alice", password_hash="secret", role="user")
    bob = User(username="bob", password_hash="secret", role="user")
    db_session.add_all([alice, bob])
    db_session.commit()
    db_session.refresh(alice)
    db_session.refresh(bob)

    db_session.add_all(
        [
            MusicLibrary(title="alice-song.mp3", filepath="assets/music/alice-song.mp3", emotion_tag="happy", user_id=alice.id),
            MusicLibrary(title="bob-song.mp3", filepath="assets/music/bob-song.mp3", emotion_tag="sad", user_id=bob.id),
        ]
    )
    db_session.commit()

    response = api_client.get("/api/user/music", params={"username": "alice"})

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 1,
            "title": "alice-song.mp3",
            "emotion_tag": "happy",
            "filepath": "assets/music/alice-song.mp3",
        }
    ]


def test_upload_user_music_creates_record_and_file(api_client, db_session, monkeypatch):
    from app import api as api_module
    from core.models import MusicLibrary, User

    alice = User(username="alice", password_hash="secret", role="user")
    db_session.add(alice)
    db_session.commit()
    db_session.refresh(alice)

    monkeypatch.setattr(api_module.time, "time", lambda: 1234567892)

    response = api_client.post(
        "/api/user/upload_music",
        data={"emotion": "happy", "username": "alice"},
        files={"file": ("uplift.mp3", b"ID3", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "上传成功"}

    music = db_session.query(MusicLibrary).one()
    created_path = REPO_ROOT / music.filepath

    try:
        assert music.title == "uplift.mp3"
        assert music.emotion_tag == "happy"
        assert music.user_id == alice.id
        assert music.filepath == "assets/music/music_alice_happy_1234567892.mp3"
        assert created_path.exists()
    finally:
        if created_path.exists():
            created_path.unlink()


def test_get_user_scripts_returns_only_owned_scripts(api_client, db_session):
    from core.models import ComfortScript, User

    alice = User(username="alice", password_hash="secret", role="user")
    bob = User(username="bob", password_hash="secret", role="user")
    db_session.add_all([alice, bob])
    db_session.commit()
    db_session.refresh(alice)
    db_session.refresh(bob)

    db_session.add_all(
        [
            ComfortScript(content="你已经做得很好了。", emotion_tag="sad", user_id=alice.id),
            ComfortScript(content="继续加油。", emotion_tag="happy", user_id=bob.id),
        ]
    )
    db_session.commit()

    response = api_client.get("/api/user/scripts", params={"username": "alice"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["content"] == "你已经做得很好了。"
    assert payload[0]["emotion_tag"] == "sad"


def test_add_user_script_creates_owned_record(api_client, db_session):
    from core.models import ComfortScript, User

    alice = User(username="alice", password_hash="secret", role="user")
    db_session.add(alice)
    db_session.commit()
    db_session.refresh(alice)

    response = api_client.post(
        "/api/user/scripts",
        json={
            "username": "alice",
            "content": "别着急，慢慢来。",
            "emotion_tag": "sad",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "添加成功"}

    script = db_session.query(ComfortScript).one()
    assert script.content == "别着急，慢慢来。"
    assert script.emotion_tag == "sad"
    assert script.user_id == alice.id


def test_delete_user_script_removes_owned_record(api_client, db_session):
    from core.models import ComfortScript, User

    alice = User(username="alice", password_hash="secret", role="user")
    db_session.add(alice)
    db_session.commit()
    db_session.refresh(alice)

    script = ComfortScript(content="会好起来的。", emotion_tag="sad", user_id=alice.id)
    db_session.add(script)
    db_session.commit()
    db_session.refresh(script)

    response = api_client.delete(f"/api/user/scripts/{script.id}", params={"username": "alice"})

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "删除成功"}
    assert db_session.query(ComfortScript).filter(ComfortScript.id == script.id).first() is None


def test_personalized_quote_returns_default_for_unknown_user(api_client):
    response = api_client.get("/api/my/personalized_quote", params={"username": "nobody"})

    assert response.status_code == 200
    assert response.json() == {"content": "欢迎回来！", "emotion_detected": "unknown"}


def test_personalized_quote_falls_back_to_local_script(api_client, db_session, monkeypatch):
    from app import api as api_module
    from core.models import ComfortScript, EmotionLog, User

    alice = User(username="alice", password_hash="secret", role="user")
    db_session.add(alice)
    db_session.commit()
    db_session.refresh(alice)

    db_session.add(EmotionLog(user_id=alice.id, is_stranger=False, emotion="sad", score=0.8))
    db_session.add(ComfortScript(content="你并不孤单。", emotion_tag="sad", user_id=None))
    db_session.commit()

    async def fake_hitokoto(emotion):
        return None

    monkeypatch.setattr(api_module, "get_hitokoto_by_emotion", fake_hitokoto)

    response = api_client.get("/api/my/personalized_quote", params={"username": "alice"})

    assert response.status_code == 200
    assert response.json() == {
        "content": "你并不孤单。",
        "emotion_tag": "sad",
        "source": "local",
    }


def test_personalized_quote_prefers_hitokoto_result(api_client, db_session, monkeypatch):
    from app import api as api_module
    from core.models import EmotionLog, User

    alice = User(username="alice", password_hash="secret", role="user")
    db_session.add(alice)
    db_session.commit()
    db_session.refresh(alice)

    db_session.add(EmotionLog(user_id=alice.id, is_stranger=False, emotion="happy", score=0.9))
    db_session.commit()

    async def fake_hitokoto(emotion):
        return "「今天也要继续发光。」"

    monkeypatch.setattr(api_module, "get_hitokoto_by_emotion", fake_hitokoto)

    response = api_client.get("/api/my/personalized_quote", params={"username": "alice"})

    assert response.status_code == 200
    assert response.json() == {
        "content": "「今天也要继续发光。」",
        "emotion_tag": "happy",
        "source": "hitokoto",
    }

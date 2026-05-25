import pytest
from datetime import datetime, timedelta


pytestmark = pytest.mark.api


def test_get_admin_logs_returns_registered_user_logs_only(api_client, db_session):
    from core.models import EmotionLog, User

    alice = User(username="alice", password_hash="secret", role="user")
    bob = User(username="bob", password_hash="secret", role="user")
    db_session.add_all([alice, bob])
    db_session.commit()
    db_session.refresh(alice)
    db_session.refresh(bob)

    db_session.add_all(
        [
            EmotionLog(
                user_id=alice.id,
                is_stranger=False,
                emotion="happy",
                score=0.9,
                timestamp=datetime.now() - timedelta(minutes=2),
            ),
            EmotionLog(
                user_id=bob.id,
                is_stranger=False,
                emotion="sad",
                score=0.3,
                timestamp=datetime.now() - timedelta(minutes=1),
            ),
            EmotionLog(
                user_id=None,
                is_stranger=True,
                emotion="fear",
                score=0.7,
                timestamp=datetime.now(),
            ),
        ]
    )
    db_session.commit()

    response = api_client.get("/api/admin/logs", params={"page": 1, "page_size": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert [item["username"] for item in payload["data"]] == ["bob", "alice"]
    assert {item["emotion"] for item in payload["data"]} == {"happy", "sad"}


def test_get_admin_logs_supports_username_filter(api_client, db_session):
    from core.models import EmotionLog, User

    alice = User(username="alice", password_hash="secret", role="user")
    bob = User(username="bob", password_hash="secret", role="user")
    db_session.add_all([alice, bob])
    db_session.commit()
    db_session.refresh(alice)
    db_session.refresh(bob)

    db_session.add_all(
        [
            EmotionLog(user_id=alice.id, is_stranger=False, emotion="happy", score=0.9),
            EmotionLog(user_id=bob.id, is_stranger=False, emotion="sad", score=0.3),
        ]
    )
    db_session.commit()

    response = api_client.get("/api/admin/logs", params={"username": "alice"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["data"][0]["username"] == "alice"
    assert payload["data"][0]["emotion"] == "happy"


def test_admin_get_users_includes_face_flag(api_client, db_session):
    from core.models import User

    db_session.add_all(
        [
            User(username="admin", password_hash="secret", role="admin"),
            User(username="alice", password_hash="secret", role="user", face_encoding=[0.1, 0.2]),
            User(username="bob", password_hash="secret", role="user", face_encoding=None),
        ]
    )
    db_session.commit()

    response = api_client.get("/api/admin/users")

    assert response.status_code == 200
    users = {item["username"]: item for item in response.json()["users"]}
    assert users["alice"]["has_face"] is True
    assert users["bob"]["has_face"] is False
    assert users["admin"]["role"] == "admin"


def test_admin_scripts_support_global_and_user_scopes(api_client, db_session):
    from core.models import ComfortScript, User

    alice = User(username="alice", password_hash="secret", role="user")
    db_session.add(alice)
    db_session.commit()
    db_session.refresh(alice)

    db_session.add_all(
        [
            ComfortScript(content="global text", emotion_tag="happy", user_id=None),
            ComfortScript(content="alice text", emotion_tag="sad", user_id=alice.id),
        ]
    )
    db_session.commit()

    global_response = api_client.get("/api/admin/scripts", params={"target_user": "global"})
    scoped_response = api_client.get("/api/admin/scripts", params={"target_user": "alice"})

    assert global_response.status_code == 200
    assert scoped_response.status_code == 200
    assert [item["content"] for item in global_response.json()] == ["global text"]
    assert [item["content"] for item in scoped_response.json()] == ["alice text"]


def test_admin_add_script_creates_user_scoped_script(api_client, db_session):
    from core.models import ComfortScript, User

    alice = User(username="alice", password_hash="secret", role="user")
    db_session.add(alice)
    db_session.commit()
    db_session.refresh(alice)

    response = api_client.post(
        "/api/admin/scripts",
        json={"content": "坚持一下", "emotion_tag": "sad", "target_user": "alice"},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True

    script = db_session.query(ComfortScript).one()
    assert script.content == "坚持一下"
    assert script.emotion_tag == "sad"
    assert script.user_id == alice.id


def test_create_test_user_is_idempotent(api_client, db_session):
    from core.models import User

    first_response = api_client.get("/api/debug/create_test_user")
    second_response = api_client.get("/api/debug/create_test_user")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["message"] == "测试用户 admin 已创建，密码 123456"
    assert second_response.json()["message"] == "用户已存在"
    assert db_session.query(User).filter(User.username == "admin").count() == 1


def test_seed_data_inserts_stranger_logs(api_client, db_session):
    from core.models import EmotionLog

    response = api_client.get("/api/debug/seed_data")

    assert response.status_code == 200
    assert response.json() == {"message": "生成完成"}
    logs = db_session.query(EmotionLog).all()
    assert len(logs) == 50
    assert all(log.user_id is None for log in logs)
    assert all(log.is_stranger is True for log in logs)

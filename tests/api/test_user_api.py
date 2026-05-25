import pytest


pytestmark = pytest.mark.api


def test_get_my_stats_returns_counts_and_face_flag(api_client, db_session):
    from core.models import EmotionLog, User

    user = User(username="alice", password_hash="secret", role="user", face_encoding=[0.1, 0.2])
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    db_session.add_all(
        [
            EmotionLog(user_id=user.id, is_stranger=False, emotion="happy", score=0.9),
            EmotionLog(user_id=user.id, is_stranger=False, emotion="happy", score=0.8),
            EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=0.4),
        ]
    )
    db_session.commit()

    response = api_client.get("/api/my/stats", params={"username": "alice"})

    assert response.status_code == 200
    assert response.json() == {
        "user_id": user.id,
        "pie_data": [
            {"name": "sad", "value": 1},
            {"name": "happy", "value": 2},
        ],
        "total_records": 3,
        "has_face": True,
    }


def test_get_my_history_returns_paginated_logs(api_client, db_session):
    from core.models import EmotionLog, User

    user = User(username="alice", password_hash="secret", role="user")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    for index in range(12):
        db_session.add(
            EmotionLog(
                user_id=user.id,
                is_stranger=False,
                emotion=f"emotion-{index}",
                score=0.5,
            )
        )
    db_session.commit()

    response = api_client.get("/api/my/history", params={"username": "alice", "page": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 12
    assert len(payload["data"]) == 10


def test_create_update_and_delete_diary(api_client, db_session):
    from core.models import User

    user = User(username="alice", password_hash="secret", role="user")
    db_session.add(user)
    db_session.commit()

    create_response = api_client.post(
        "/api/my/diaries",
        json={"username": "alice", "content": "今天心情不错", "emotion": "Happy"},
    )

    assert create_response.status_code == 200
    diary_id = create_response.json()["id"]

    list_response = api_client.get("/api/my/diaries", params={"username": "alice"})
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1

    update_response = api_client.put(
        f"/api/my/diaries/{diary_id}",
        json={"content": "更新后的内容", "emotion": "Sad"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["success"] is True

    delete_response = api_client.delete(f"/api/my/diaries/{diary_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["success"] is True

    final_list = api_client.get("/api/my/diaries", params={"username": "alice"})
    assert final_list.json() == []

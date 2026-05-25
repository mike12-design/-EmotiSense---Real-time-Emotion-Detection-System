from types import SimpleNamespace
from datetime import datetime, timedelta

import pytest


pytestmark = pytest.mark.api


class _StaticQuery:
    def __init__(self, items):
        self._items = items

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def group_by(self, *args, **kwargs):
        return self

    def join(self, *args, **kwargs):
        return self

    def distinct(self, *args, **kwargs):
        return self

    def all(self):
        return list(self._items)

    def count(self):
        return len(self._items)

    def first(self):
        return self._items[0] if self._items else None


class _AnalyticsSessionProxy:
    def __init__(self, real_session, system_events=None, users=None, logs=None):
        self._real_session = real_session
        self._system_events = system_events or []
        self._users = users or []
        self._logs = logs or []

    def query(self, model, *args, **kwargs):
        if getattr(model, "__name__", "") == "SystemEvent":
            return _StaticQuery(self._system_events)
        if getattr(model, "__name__", "") == "User":
            return _StaticQuery(self._users)
        if getattr(model, "__name__", "") == "EmotionLog":
            return _StaticQuery(self._logs)
        return self._real_session.query(model, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real_session, name)


def test_admin_analytics_stats_returns_global_overview(api_client, db_session):
    from core.models import EmotionLog, User

    admin = User(username="admin", password_hash="secret", role="admin")
    alice = User(username="alice", password_hash="secret", role="user")
    db_session.add_all([admin, alice])
    db_session.commit()
    db_session.refresh(alice)

    db_session.add_all(
        [
            EmotionLog(user_id=alice.id, is_stranger=False, emotion="happy", score=0.9, timestamp=datetime.now() - timedelta(hours=1)),
            EmotionLog(user_id=alice.id, is_stranger=False, emotion="sad", score=0.4, timestamp=datetime.now() - timedelta(hours=2)),
            EmotionLog(user_id=alice.id, is_stranger=False, emotion="surprise", score=0.8, timestamp=datetime.now() - timedelta(hours=3)),
        ]
    )
    db_session.commit()

    response = api_client.get("/api/admin/analytics/stats", params={"time_range": "24h"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["overview"]["total_users"] == 2
    assert payload["overview"]["total_logs"] == 3
    assert payload["overview"]["avg_positive_rate"] == 66.7
    assert payload["top_users"]["users"][0]["username"] == "alice"
    assert sorted(item["name"] for item in payload["pie_data"]) == ["happy", "sad", "surprise"]


def test_admin_analytics_stats_supports_user_scope(api_client, db_session):
    from core.models import EmotionLog, User

    alice = User(username="alice", password_hash="secret", role="user")
    bob = User(username="bob", password_hash="secret", role="user")
    db_session.add_all([alice, bob])
    db_session.commit()
    db_session.refresh(alice)
    db_session.refresh(bob)

    db_session.add_all(
        [
            EmotionLog(user_id=alice.id, is_stranger=False, emotion="happy", score=0.9, timestamp=datetime.now() - timedelta(hours=1)),
            EmotionLog(user_id=alice.id, is_stranger=False, emotion="sad", score=0.4, timestamp=datetime.now() - timedelta(hours=2)),
            EmotionLog(user_id=bob.id, is_stranger=False, emotion="sad", score=0.4, timestamp=datetime.now() - timedelta(hours=1)),
        ]
    )
    db_session.commit()

    response = api_client.get("/api/admin/analytics/stats", params={"time_range": "24h", "user_id": alice.id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["overview"]["total_users"] == 1
    assert payload["overview"]["total_logs"] == 2
    assert payload["overview"]["active_users_today"] == 2
    assert payload["top_users"]["users"] == []
    assert payload["user_comparison"]["series"] == [{"name": "该用户", "data": [0, 0, 0, 1, 1]}]


def test_admin_analytics_advanced_handles_no_data(api_client, db_session):
    from core.models import User

    user = User(username="alice", password_hash="secret", role="user")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    response = api_client.get(f"/api/admin/analytics/advanced/{user.id}")

    assert response.status_code == 200
    assert response.json()["error"] == "无足够数据"


def test_admin_analytics_advanced_returns_analysis_payload(api_client, db_session, monkeypatch):
    from app import api as api_module
    from core.models import EmotionLog, User

    user = User(username="alice", password_hash="secret", role="user")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    base_time = datetime.now() - timedelta(days=1)
    db_session.add_all(
        [
            EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=80, timestamp=base_time),
            EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=70, timestamp=base_time + timedelta(minutes=10)),
            EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=60, timestamp=base_time + timedelta(minutes=20)),
            EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=50, timestamp=base_time + timedelta(minutes=30)),
        ]
    )
    db_session.commit()

    monkeypatch.setattr(
        api_module.advanced_analyzer,
        "analyze",
        lambda logs, days=7: SimpleNamespace(
            attractor=-0.5,
            attractor_std=0.1,
            rmssd=0.05,
            current_valence=-0.6,
            deviation=1.2,
            smoothed_valence=[-0.5, -0.55, -0.58, -0.6],
            intervention_needed=True,
            intervention_type="tts_urgency",
            risk_level="high",
        ),
    )
    monkeypatch.setattr(api_module.advanced_analyzer, "convert_to_valence_series", lambda logs: ([], [(-0.4), (-0.5), (-0.6), (-0.7)]))
    monkeypatch.setattr(api_module.advanced_analyzer, "get_trend_direction", lambda series: "falling")
    monkeypatch.setattr(api_module.advanced_analyzer, "calculate_emotion_inertia", lambda series: 0.82)

    response = api_client.get(f"/api/admin/analytics/advanced/{user.id}", params={"days": 7})

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == user.id
    assert payload["intervention"]["type"] == "tts_urgency"
    assert payload["intervention"]["risk_level"] == "high"
    assert payload["trend"]["direction"] == "falling"
    assert payload["suggestions"][0]["type"] == "tts"


def test_admin_diary_validation_reports_insufficient_data(api_client, db_session):
    from core.models import User

    user = User(username="alice", password_hash="secret", role="user")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    response = api_client.post(f"/api/admin/analytics/diary/validate/{user.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["consistency"] == "insufficient_data"
    assert payload["diary_count"] == 0
    assert payload["vision_count"] == 0


def test_admin_diary_validation_triggers_questionnaire(api_client, db_session, monkeypatch):
    from app import api as api_module
    from core.models import Diary, EmotionLog, User

    user = User(username="alice", password_hash="secret", role="user")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    now = datetime.now()
    db_session.add_all(
        [
            Diary(user_id=user.id, title="d1", content="今天很糟", emotion="Sad", timestamp=now - timedelta(days=1)),
            Diary(user_id=user.id, title="d2", content="仍然难过", emotion="Sad", timestamp=now - timedelta(hours=12)),
            EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=80, timestamp=now - timedelta(days=1)),
            EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=70, timestamp=now - timedelta(hours=12)),
        ]
    )
    db_session.commit()

    monkeypatch.setattr(
        api_module.diary_analyzer,
        "validate_visual_emotion",
        lambda diary_entries, vision_entries: {
            "visual_avg": -0.8,
            "diary_avg": 0.5,
            "agreement_rate": 0.1,
        },
    )

    response = api_client.post(f"/api/admin/analytics/diary/validate/{user.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["trigger_questionnaire"] is True
    assert payload["agreement_rate"] == 0.1


def test_admin_intervention_suggest_returns_current_state(api_client, db_session, monkeypatch):
    from app import api as api_module
    from core.models import EmotionLog, User

    user = User(username="alice", password_hash="secret", role="user")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    db_session.add(
        EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=80, timestamp=datetime.now() - timedelta(hours=1))
    )
    db_session.commit()

    monkeypatch.setattr(
        api_module.advanced_analyzer,
        "analyze",
        lambda logs, days=1: SimpleNamespace(
            current_valence=-0.5,
            risk_level="high",
            intervention_type="music",
            intervention_needed=True,
            attractor=-0.5,
            attractor_std=0.1,
            rmssd=0.05,
            deviation=1.0,
            smoothed_valence=[-0.5],
        ),
    )

    response = api_client.get(f"/api/admin/analytics/intervention/suggest/{user.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_state"]["risk_level"] == "high"
    assert payload["suggestions"][0]["type"] == "music"


def test_admin_alert_feed_returns_system_and_user_alerts(api_client, db_session):
    from core.models import EmotionLog, User

    user = User(username="alice", password_hash="secret", role="user")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    now = datetime.now()
    logs = [
        EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=90, timestamp=now - timedelta(minutes=50)),
        EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=85, timestamp=now - timedelta(minutes=40)),
        EmotionLog(user_id=user.id, is_stranger=False, emotion="sad", score=80, timestamp=now - timedelta(minutes=35)),
    ]
    db_session.add_all(logs)
    db_session.commit()

    system_events = [SimpleNamespace(timestamp=now - timedelta(minutes=30), event_type="music intervention")]
    proxy = _AnalyticsSessionProxy(db_session, system_events=system_events, users=[user], logs=logs)

    from app.api import get_db as api_get_db

    def override_get_db():
        yield proxy

    api_client.app.dependency_overrides[api_get_db] = override_get_db
    try:
        response = api_client.get("/api/admin/analytics/alerts", params={"limit": 10})
    finally:
        api_client.app.dependency_overrides.pop(api_get_db, None)

    assert response.status_code == 200
    alerts = response.json()["alerts"]
    assert any(item["username"] == "系统" for item in alerts)
    assert any(item["username"] == "alice" for item in alerts)


def test_admin_quadrant_returns_user_points(api_client, db_session):
    from core.models import EmotionLog, User

    admin = User(username="admin", password_hash="secret", role="admin")
    alice = User(username="alice", password_hash="secret", role="user")
    bob = User(username="bob", password_hash="secret", role="user")
    db_session.add_all([admin, alice, bob])
    db_session.commit()
    db_session.refresh(alice)
    db_session.refresh(bob)

    now = datetime.now()
    for index in range(5):
        db_session.add(
            EmotionLog(
                user_id=alice.id,
                is_stranger=False,
                emotion="happy" if index % 2 == 0 else "sad",
                score=80,
                timestamp=now - timedelta(hours=index),
            )
        )
        db_session.add(
            EmotionLog(
                user_id=bob.id,
                is_stranger=False,
                emotion="sad",
                score=70,
                timestamp=now - timedelta(hours=index),
            )
        )
    db_session.commit()

    response = api_client.get("/api/admin/analytics/quadrant")

    assert response.status_code == 200
    users = {item["username"] for item in response.json()["users"]}
    assert users == {"alice", "bob"}
    assert "admin" not in users


def test_admin_interventions_returns_timeline(api_client, db_session):
    now = datetime.now()
    system_events = [
        SimpleNamespace(timestamp=now - timedelta(days=1), event_type="music intervention"),
        SimpleNamespace(timestamp=now - timedelta(hours=6), event_type="voice intervention"),
    ]
    proxy = _AnalyticsSessionProxy(db_session, system_events=system_events)

    from app.api import get_db as api_get_db

    def override_get_db():
        yield proxy

    api_client.app.dependency_overrides[api_get_db] = override_get_db
    try:
        response = api_client.get("/api/admin/analytics/interventions/1", params={"days": 7})
    finally:
        api_client.app.dependency_overrides.pop(api_get_db, None)

    assert response.status_code == 200
    events = response.json()["events"]
    assert [item["type"] for item in events] == ["music", "tts"]


def test_admin_system_health_returns_summary(api_client, db_session):
    from core.models import EmotionLog, User

    alice = User(username="alice", password_hash="secret", role="user")
    bob = User(username="bob", password_hash="secret", role="user")
    db_session.add_all([alice, bob])
    db_session.commit()
    db_session.refresh(alice)
    db_session.refresh(bob)

    now = datetime.now()
    db_session.add_all(
        [
            EmotionLog(user_id=alice.id, is_stranger=False, emotion="happy", score=90, timestamp=now - timedelta(hours=1)),
            EmotionLog(user_id=bob.id, is_stranger=False, emotion="sad", score=30, timestamp=now - timedelta(hours=2)),
            EmotionLog(user_id=bob.id, is_stranger=False, emotion="sad", score=40, timestamp=now - timedelta(hours=3)),
        ]
    )
    db_session.commit()

    response = api_client.get("/api/admin/analytics/system-health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["totalRecords"] == 3
    assert len(payload["confidenceDistribution"]) == 5
    assert payload["modelAccuracy"] == 1.0

import pytest


pytestmark = pytest.mark.api


def test_register_creates_user(api_client):
    response = api_client.post(
        "/api/register",
        json={"username": "alice", "password": "secret123"},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_login_returns_role_and_username(api_client, seeded_admin):
    response = api_client.post(
        "/api/login",
        json={"username": "admin", "password": "123456"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "role": "admin",
        "username": "admin",
    }

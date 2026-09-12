def test_register_creates_user(client):
    resp = client.post(
        "/auth/register", json={"email": "new.user@example.com", "password": "supersecret1"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "new.user@example.com"
    assert "hashed_password" not in body  # never leak the hash to the client


def test_register_rejects_short_password(client):
    resp = client.post(
        "/auth/register", json={"email": "short.pw@example.com", "password": "abc123"}
    )
    assert resp.status_code == 422


def test_register_rejects_duplicate_email(client, registered_user):
    resp = client.post(
        "/auth/register",
        json={"email": registered_user["email"], "password": "anotherpassword1"},
    )
    assert resp.status_code == 400


def test_login_succeeds_with_correct_credentials(client, registered_user):
    resp = client.post(
        "/auth/login",
        data={"username": registered_user["email"], "password": registered_user["password"]},
    )
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_fails_with_wrong_password(client, registered_user):
    resp = client.post(
        "/auth/login",
        data={"username": registered_user["email"], "password": "wrong-password"},
    )
    assert resp.status_code == 401


def test_me_requires_token(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_me_returns_current_user(client, auth_headers, registered_user):
    resp = client.get("/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["email"] == registered_user["email"]
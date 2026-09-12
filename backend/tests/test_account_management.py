import io
import re

from PIL import Image


def _tiny_jpeg_bytes():
    img = Image.new("RGB", (64, 64), color=(120, 140, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_change_password_requires_correct_current_password(client, auth_headers, registered_user):
    resp = client.post(
        "/auth/change-password",
        headers=auth_headers,
        json={"current_password": "wrong", "new_password": "brandnewpassword1"},
    )
    assert resp.status_code == 401


def test_change_password_succeeds_and_old_password_stops_working(
    client, auth_headers, registered_user
):
    resp = client.post(
        "/auth/change-password",
        headers=auth_headers,
        json={
            "current_password": registered_user["password"],
            "new_password": "brandnewpassword1",
        },
    )
    assert resp.status_code == 204

    old_login = client.post(
        "/auth/login",
        data={"username": registered_user["email"], "password": registered_user["password"]},
    )
    assert old_login.status_code == 401

    new_login = client.post(
        "/auth/login",
        data={"username": registered_user["email"], "password": "brandnewpassword1"},
    )
    assert new_login.status_code == 200


def test_forgot_password_returns_202_for_unknown_email_too(client):
    """Must not reveal whether the email is registered (enumeration protection)."""
    resp = client.post("/auth/forgot-password", json={"email": "nobody@example.com"})
    assert resp.status_code == 202


def test_forgot_and_reset_password_flow(client, registered_user, caplog):
    with caplog.at_level("INFO"):
        resp = client.post("/auth/forgot-password", json={"email": registered_user["email"]})
    assert resp.status_code == 202

    match = re.search(r"reset_token=(\S+)", caplog.text)
    assert match, "Expected the reset token to be logged (no email provider wired up yet)"
    token = match.group(1)

    reset_resp = client.post(
        "/auth/reset-password", json={"token": token, "new_password": "resetpassword1"}
    )
    assert reset_resp.status_code == 204

    login = client.post(
        "/auth/login",
        data={"username": registered_user["email"], "password": "resetpassword1"},
    )
    assert login.status_code == 200


def test_reset_password_rejects_garbage_token(client):
    resp = client.post(
        "/auth/reset-password", json={"token": "not-a-real-token", "new_password": "abcdefgh1"}
    )
    assert resp.status_code == 400


def test_reset_token_cannot_be_used_as_login_token(client, registered_user, caplog):
    with caplog.at_level("INFO"):
        client.post("/auth/forgot-password", json={"email": registered_user["email"]})
    match = re.search(r"reset_token=(\S+)", caplog.text)
    token = match.group(1)

    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_delete_account_removes_user_and_history(client, auth_headers):
    predict_resp = client.post(
        "/predict",
        headers=auth_headers,
        files={"file": ("face.jpg", _tiny_jpeg_bytes(), "image/jpeg")},
    )
    assert predict_resp.status_code == 200
    assert predict_resp.json()["saved_to_history"] is True

    resp = client.delete("/auth/me", headers=auth_headers)
    assert resp.status_code == 204

    me_resp = client.get("/auth/me", headers=auth_headers)
    assert me_resp.status_code == 401
from __future__ import annotations

import os

from services.Email.email_service import EmailService


def _make_service(monkeypatch) -> EmailService:
    monkeypatch.setenv("EMAIL_NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("EMAIL_FROM", "no-reply@example.com")
    monkeypatch.setenv("EMAIL_APP_PASSWORD", "dummy-password")
    monkeypatch.setenv("EMAIL_TO", "recipient@example.com")
    service = EmailService()
    service.enabled = True
    return service


def test_generation_started_email(monkeypatch):
    captured: dict[str, str] = {}
    service = _make_service(monkeypatch)

    def fake_send(subject: str, message: str) -> bool:
        captured["subject"] = subject
        captured["message"] = message
        return True

    service.send_email = fake_send  # type: ignore[assignment]

    courses = ["EEE 101", "EEE 201", "EEE 301"]
    service.send_generation_started(
        courses,
        theory_per_request=10,
        calc_per_request=5,
        resume=True,
        store_firestore=True,
        model="gemini-2.5-flash-lite",
        temperature=0.2,
    )

    assert captured["subject"] == "CourseGen Question Generation Started"
    assert "EEE 101" in captured["message"]
    assert "Theory batch size" in captured["message"]
    assert "Temperature: 0.2" in captured["message"]


def test_course_finished_email(monkeypatch):
    captured: dict[str, str] = {}
    service = _make_service(monkeypatch)

    def fake_send(subject: str, message: str) -> bool:
        captured["subject"] = subject
        captured["message"] = message
        return True

    service.send_email = fake_send  # type: ignore[assignment]

    service.send_course_finished(
        course_code="EEE 101",
        course_title="Signals",
        question_count=40,
        duration_seconds=180,
        status="completed",
    )

    assert captured["subject"] == "CourseGen: EEE 101 Completed"
    assert "Questions generated: 40" in captured["message"]
    assert "Duration" in captured["message"]

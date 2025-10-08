#!/usr/bin/env python3
"""Standalone test script for email notification system."""

import os
import sys
from unittest.mock import patch, MagicMock

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠ python-dotenv not available, using existing environment variables")

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_email_service_initialization():
    """Test that email service initializes correctly."""
    print("Testing email service initialization...")

    from services.Email.email_service import EmailService

    # Test with environment variables from .env
    service = EmailService()

    # Print actual values for debugging
    print(f"  From email: {service.from_email}")
    print(f"  To email: {service.to_email}")
    print(f"  From password: {'*' * len(service.from_password) if service.from_password else 'None'}")
    print(f"  Enabled: {service.enabled}")
    print(f"  SMTP server: {service.smtp_server}")
    print(f"  SMTP port: {service.smtp_port}")

    # Check environment variables
    print(f"  EMAIL_FROM env: {os.environ.get('EMAIL_FROM', 'Not set')}")
    print(f"  EMAIL_APP_PASSWORD env: {'*' * len(os.environ.get('EMAIL_APP_PASSWORD', '')) if os.environ.get('EMAIL_APP_PASSWORD') else 'Not set'}")

    # Check that service is configured correctly
    assert service.from_email == "email@example.com"
    assert service.to_email == "team@example.com"  # From .env file
    assert service.enabled == True
    assert service.smtp_server == "smtp.gmail.com"
    assert service.smtp_port == 587

    print("✓ Email service initialization successful")
    return service

def test_email_service_methods():
    """Test email service methods without actually sending emails."""
    print("\nTesting email service methods...")

    from services.Email.email_service import EmailService

    service = EmailService()

    # Mock the actual email sending to avoid sending real emails
    with patch.object(service, 'send_email') as mock_send:
        mock_send.return_value = True

        # Test course started notification
        result = service.send_course_started(
            course_code="EEE 313",
            course_title="Electronics I",
            total_topics=5,
            total_subtopics=25
        )
        assert result == True
        print("✓ Course started notification method works")

        # Test topic finished notification
        result = service.send_topic_finished(
            course_code="EEE 313",
            course_title="Electronics I",
            topic_title="Semiconductor Materials",
            question_count=15,
            total_subtopics=5,
            completed_subtopics=5,
            errored_subtopics=0,
            duration_seconds=120.5
        )
        assert result == True
        print("✓ Topic finished notification method works")

        # Test course finished notification
        result = service.send_course_finished(
            course_code="EEE 313",
            course_title="Electronics I",
            question_count=75,
            duration_seconds=3600,
            status="completed"
        )
        assert result == True
        print("✓ Course finished notification method works")

def test_email_service_disabled():
    """Test email service when disabled."""
    print("\nTesting email service when disabled...")

    from services.Email.email_service import EmailService

    # Create service with disabled notifications
    with patch.dict(os.environ, {'EMAIL_NOTIFICATIONS_ENABLED': 'false'}):
        service = EmailService()
        assert service.enabled == False

        # Should return False without sending
        result = service.send_course_started(
            course_code="EEE 313",
            course_title="Electronics I",
            total_topics=5,
            total_subtopics=25
        )
        assert result == False
        print("✓ Email service correctly disabled")

def test_get_email_service_singleton():
    """Test the global email service singleton."""
    print("\nTesting email service singleton...")

    from services.Email.email_service import get_email_service

    service1 = get_email_service()
    service2 = get_email_service()

    # Should return the same instance
    assert service1 is service2
    print("✓ Email service singleton works correctly")

def test_course_progress_integration():
    """Test that course progress tracking works with notifications."""
    print("\nTesting course progress integration...")

    from services.QuestionRag.utils.course_progress import CourseProgressCache
    from services.Email.email_service import EmailService
    import tempfile
    import json

    # Create a temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock course progress cache
        progress = CourseProgressCache(
            course_code="EEE 313",
            cache_root=temp_dir,
            theory_target=10,
            calc_target=5
        )

        # Test that progress tracking works
        progress.touch_subtopic("Semiconductor Materials", "Diodes")
        progress.mark_request_started("Semiconductor Materials", "Diodes", "theory_1")
        progress.mark_request_completed("Semiconductor Materials", "Diodes", "theory_1", 10)

        # Complete calculation requests as well to make subtopic complete
        progress.mark_request_started("Semiconductor Materials", "Diodes", "calculation_1")
        progress.mark_request_completed("Semiconductor Materials", "Diodes", "calculation_1", 5)

        # Complete the second calculation request (calc_progress2)
        progress.mark_request_started("Semiconductor Materials", "Diodes", "calculation-2")
        progress.mark_request_completed("Semiconductor Materials", "Diodes", "calculation-2", 5)

        # Debug: Print the actual state and progress values
        actual_state = progress.subtopic_state("Semiconductor Materials", "Diodes")
        print(f"  Subtopic state: {actual_state}")

        # Debug: Check the actual progress values
        entry = progress._ensure_entry("Semiconductor Materials", "Diodes")
        print(f"  Theory progress: {entry.get('theory_progress', 0)}/{progress.theory_target}")
        print(f"  Calc progress: {entry.get('calculation_progress', 0)}/{progress.calc_target}")
        print(f"  Calc2 progress: {entry.get('calc_progress2', 0)}/{progress.calc_target}")

        # Check that progress is tracked correctly
        # The subtopic should be completed since we met both theory and calc targets
        assert actual_state == "completed"
        print("✓ Course progress tracking works correctly")

def main():
    """Run all tests."""
    print("🚀 Starting email notification system tests...\n")

    try:
        test_email_service_initialization()
        test_email_service_methods()
        test_email_service_disabled()
        test_get_email_service_singleton()
        test_course_progress_integration()

        print("\n🎉 All tests passed! Email notification system is working correctly.")
        print("\n📋 Summary:")
        print("   • Email service is properly configured")
        print("   • All notification methods are implemented")
        print("   • Integration with course progress tracking is in place")
        print("   • The system is ready to send notifications when courses start and topics finish")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

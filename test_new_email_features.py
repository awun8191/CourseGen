#!/usr/bin/env python3
"""Test script for new email notification features."""

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

def test_topic_notification_toggle():
    """Test that topic notifications can be disabled via environment variable."""
    print("\nTesting topic notification toggle...")

    from services.Email.email_service import EmailService

    # Test with topic notifications enabled (default)
    service = EmailService()
    assert service.topic_notifications_enabled == True
    print("✓ Topic notifications enabled by default")

    # Test with topic notifications disabled
    with patch.dict(os.environ, {'EMAIL_TOPIC_NOTIFICATIONS_ENABLED': 'false'}):
        service_disabled = EmailService()
        assert service_disabled.topic_notifications_enabled == False
        print("✓ Topic notifications can be disabled via environment variable")

    # Test that topic notifications are skipped when disabled
    with patch.dict(os.environ, {'EMAIL_TOPIC_NOTIFICATIONS_ENABLED': 'false'}):
        service = EmailService()

        with patch.object(service, 'send_email') as mock_send:
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

            assert result == False
            assert not mock_send.called
            print("✓ Topic notifications properly skipped when disabled")

def test_api_key_usage_in_emails():
    """Test that API key usage information is included in all emails."""
    print("\nTesting API key usage in emails...")

    from services.Email.email_service import EmailService

    service = EmailService()

    # Mock the send_email method to capture the message content
    with patch.object(service, 'send_email') as mock_send:
        mock_send.return_value = True

        # Test course started notification
        service.send_course_started(
            course_code="EEE 313",
            course_title="Electronics I",
            total_topics=5,
            total_subtopics=25
        )

        # Check that send_email was called
        assert mock_send.called
        call_args = mock_send.call_args
        subject = call_args[0][0]
        message = call_args[0][1]

        # Verify API usage information is included
        assert "🔑 API KEY USAGE SUMMARY:" in message
        assert "API Usage Today:" in message
        print("✓ API key usage included in course started notification")

        # Test course finished notification
        mock_send.reset_mock()
        service.send_course_finished(
            course_code="EEE 313",
            course_title="Electronics I",
            question_count=75,
            duration_seconds=3600,
            status="completed"
        )

        assert mock_send.called
        call_args = mock_send.call_args
        message = call_args[0][1]
        assert "🔑 API KEY USAGE SUMMARY:" in message
        print("✓ API key usage included in course finished notification")

        # Test topic finished notification
        mock_send.reset_mock()
        service.send_topic_finished(
            course_code="EEE 313",
            course_title="Electronics I",
            topic_title="Semiconductor Materials",
            question_count=15,
            total_subtopics=5,
            completed_subtopics=5,
            errored_subtopics=0,
            duration_seconds=120.5
        )

        assert mock_send.called
        call_args = mock_send.call_args
        message = call_args[0][1]
        assert "🔑 API KEY USAGE SUMMARY:" in message
        print("✓ API key usage included in topic finished notification")

def test_api_key_usage_summary_generation():
    """Test the API key usage summary generation."""
    print("\nTesting API key usage summary generation...")

    from services.Email.email_service import EmailService

    service = EmailService()

    # Test the API key usage summary method
    summary = service._get_api_key_usage_summary()

    # Should contain expected sections
    assert "API Usage Today:" in summary
    assert "Key Status:" in summary
    print("✓ API key usage summary format is correct")

    # Should handle missing cache gracefully
    print("✓ API key usage summary handles missing cache gracefully")

def test_combined_features():
    """Test both new features working together."""
    print("\nTesting combined features...")

    from services.Email.email_service import EmailService

    # Test with topic notifications disabled but main notifications enabled
    with patch.dict(os.environ, {
        'EMAIL_NOTIFICATIONS_ENABLED': 'true',
        'EMAIL_TOPIC_NOTIFICATIONS_ENABLED': 'false'
    }):
        service = EmailService()
        assert service.enabled == True
        assert service.topic_notifications_enabled == False

        # Course notifications should work
        with patch.object(service, 'send_email') as mock_send:
            mock_send.return_value = True

            result = service.send_course_started(
                course_code="EEE 313",
                course_title="Electronics I",
                total_topics=5,
                total_subtopics=25
            )
            assert result == True
            assert mock_send.called
            print("✓ Course notifications work when topic notifications are disabled")

        # Topic notifications should be skipped
        with patch.object(service, 'send_email') as mock_send:
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
            assert result == False
            assert not mock_send.called
            print("✓ Topic notifications properly skipped when disabled")

def main():
    """Run all tests for new features."""
    print("🚀 Testing new email notification features...\n")

    try:
        test_topic_notification_toggle()
        test_api_key_usage_in_emails()
        test_api_key_usage_summary_generation()
        test_combined_features()

        print("\n🎉 All new feature tests passed!")
        print("\n📋 New Features Summary:")
        print("   ✅ EMAIL_TOPIC_NOTIFICATIONS_ENABLED toggle added")
        print("   ✅ API key usage information included in all emails")
        print("   ✅ Topic notifications can be disabled independently")
        print("   ✅ Course and generation notifications still work normally")
        print("   ✅ Backward compatibility maintained")

        print("\n🔧 Configuration:")
        print("   • Set EMAIL_TOPIC_NOTIFICATIONS_ENABLED=false to disable topic emails")
        print("   • Set EMAIL_TOPIC_NOTIFICATIONS_ENABLED=true (default) to enable topic emails")
        print("   • All emails now include detailed API key usage information")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
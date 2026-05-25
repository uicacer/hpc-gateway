"""Tests for hpc_as_api.utils — message utility functions."""

import pytest
from hpc_as_api.utils import (
    count_images,
    extract_text_content,
    has_images,
    strip_old_images,
)


# ---------------------------------------------------------------------------
# extract_text_content
# ---------------------------------------------------------------------------

def test_extract_text_plain_string():
    assert extract_text_content("Hello world") == "Hello world"


def test_extract_text_from_multimodal():
    content = [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
    ]
    assert extract_text_content(content) == "What is this?"


def test_extract_text_image_only():
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}]
    assert extract_text_content(content) == ""


def test_extract_text_multiple_text_blocks():
    content = [
        {"type": "text", "text": "First"},
        {"type": "text", "text": "second"},
    ]
    assert extract_text_content(content) == "First second"


# ---------------------------------------------------------------------------
# has_images
# ---------------------------------------------------------------------------

def test_has_images_false_for_text_only():
    messages = [{"role": "user", "content": "Hello"}]
    assert has_images(messages) is False


def test_has_images_true():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    ]
    assert has_images(messages) is True


def test_has_images_empty():
    assert has_images([]) is False


# ---------------------------------------------------------------------------
# count_images
# ---------------------------------------------------------------------------

def test_count_images_none():
    messages = [{"role": "user", "content": "text"}]
    assert count_images(messages) == 0


def test_count_images_multiple():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:..."}},
                {"type": "image_url", "image_url": {"url": "data:..."}},
                {"type": "text", "text": "Compare these"},
            ],
        }
    ]
    assert count_images(messages) == 2


def test_count_images_across_messages():
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "a"}}]},
        {"role": "assistant", "content": "I see a cat."},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "b"}}]},
    ]
    assert count_images(messages) == 2


# ---------------------------------------------------------------------------
# strip_old_images
# ---------------------------------------------------------------------------

def test_strip_old_images_empty():
    assert strip_old_images([]) == []


def test_strip_old_images_no_images():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = strip_old_images(messages)
    assert result == messages


def test_strip_old_images_keeps_latest_user():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "img1"}},
            ],
        },
        {"role": "assistant", "content": "I see a cat."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "and this?"},
                {"type": "image_url", "image_url": {"url": "img2"}},
            ],
        },
    ]
    result = strip_old_images(messages)

    # First user message: image stripped, text kept
    assert result[0]["content"] == [{"type": "text", "text": "first"}]
    # Assistant: unchanged
    assert result[1] == messages[1]
    # Last user message: kept intact with image
    assert result[2] == messages[2]


def test_strip_old_images_image_only_message_gets_placeholder():
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "img"}}]},
        {"role": "user", "content": "follow-up"},
    ]
    result = strip_old_images(messages)
    # First message had no text blocks — replaced with placeholder
    assert result[0]["content"] == "(image)"
    # Last user message kept as-is
    assert result[1] == messages[1]


def test_strip_old_images_does_not_mutate_input():
    img_block = {"type": "image_url", "image_url": {"url": "img"}}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a"}, img_block]},
        {"role": "user", "content": "b"},
    ]
    original_first_content = list(messages[0]["content"])
    strip_old_images(messages)
    # Original list unchanged
    assert messages[0]["content"] == original_first_content

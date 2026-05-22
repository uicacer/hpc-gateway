"""
Message utility functions for hpc-gateway.

Helpers for working with OpenAI-format messages, including multimodal
content (text + images). Used by the compute client to manage payload
size before submitting jobs to Globus Compute.

BACKGROUND: THE OPENAI MESSAGE FORMAT
======================================
Every message in a conversation has a "role" and "content":

  Text-only message:
    {"role": "user", "content": "What is Python?"}

  Multimodal message (text + image):
    {"role": "user", "content": [
        {"type": "text",      "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}}
    ]}

When images are present, "content" changes from a plain string to a list
of typed blocks. These utilities handle both formats transparently.

WHY IMAGES NEED SPECIAL HANDLING
==================================
Globus Compute has an 8 MB payload limit per job submission. A base64-encoded
image is ~1.3x the original file size — a 2 MB photo becomes ~2.6 MB of base64.
A conversation with 3-4 image exchanges can easily blow the 8 MB limit.

The solution: strip images from older messages before submitting. The model's
prior text responses about old images provide enough context for follow-up
questions. Only the current (latest) user message keeps its images — that's
what the model needs to process right now.
"""


def extract_text_content(content: str | list[dict]) -> str:
    """
    Extract the text portion from a message's content field.

    Handles both formats:
      - String content (text-only): returns as-is
      - List content (multimodal): joins all "text" blocks, ignores image blocks

    Used by the complexity judge and logging to get readable text from any message.

    Examples:
        extract_text_content("What is Python?")
        → "What is Python?"

        extract_text_content([
            {"type": "text",      "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ])
        → "What is in this image?"

        extract_text_content([
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ])
        → ""  (no text blocks — image-only message)
    """
    if isinstance(content, str):
        return content

    # Multimodal: collect text from every "text" block, ignore "image_url" blocks
    return " ".join(block.get("text", "") for block in content if block.get("type") == "text")


def has_images(messages: list[dict]) -> bool:
    """
    Return True if ANY message in the conversation contains image content.

    Scans all messages — not just the latest — because some models need to
    know if images appeared anywhere in the conversation history.

    Short-circuits on the first image found for efficiency on long histories.

    Example:
        has_images([{"role": "user", "content": "Hello"}])
        → False

        has_images([{"role": "user", "content": [
            {"type": "text",      "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]}])
        → True
    """
    for msg in messages:
        content = msg.get("content", "")
        # Only list-type content can contain images — plain strings are always text-only
        if isinstance(content, list) and any(
            block.get("type") == "image_url" for block in content
        ):
            return True
    return False


def count_images(messages: list[dict]) -> int:
    """
    Count the total number of image_url blocks across all messages.

    Used to estimate payload size before submitting to Globus Compute:
    each image adds roughly (original_file_size * 1.33) bytes to the payload.

    Example:
        count_images([{"role": "user", "content": [
            {"type": "text",      "text": "Compare these"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "image_url", "image_url": {"url": "data:..."}}
        ]}])
        → 2
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            total += sum(1 for block in content if block.get("type") == "image_url")
    return total


def strip_old_images(messages: list[dict]) -> list[dict]:
    """
    Remove images from all messages EXCEPT the latest user message.

    WHY THIS EXISTS:
    ----------------
    Globus Compute has an 8 MB payload limit. A long conversation with multiple
    image exchanges can exceed this limit. By stripping images from older messages,
    we keep the payload small while preserving:
      - All text content in every message (the model's descriptions of old images)
      - Images in the CURRENT user message (what the model needs to process now)

    The model can reference its own prior text descriptions for context about
    older images. If the user asks "what about that first image?", the model's
    earlier response ("The first image shows a circuit diagram...") provides
    enough context without re-sending the raw image bytes.

    HOW IT WORKS:
    -------------
    1. Find the index of the last user message in the conversation
    2. Walk every message:
       - If it's the last user message → keep intact (images included)
       - If its content is a list → remove image_url blocks, keep text blocks
         If removing images leaves nothing → replace content with "(image)" placeholder
       - Otherwise (plain string) → keep as-is

    Returns a NEW list — the original messages are not modified (no side effects).

    Example:
        messages = [
            {"role": "user",      "content": [{"type": "text", "text": "first"},
                                               {"type": "image_url", ...}]},
            {"role": "assistant", "content": "I see a cat."},
            {"role": "user",      "content": [{"type": "text", "text": "and this?"},
                                               {"type": "image_url", ...}]},
        ]
        strip_old_images(messages)
        # First user message → image stripped, text kept
        # Assistant message → unchanged (plain string)
        # Last user message → kept intact with image
        → [
            {"role": "user",      "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": "I see a cat."},
            {"role": "user",      "content": [{"type": "text", "text": "and this?"},
                                               {"type": "image_url", ...}]},
          ]
    """
    if not messages:
        return messages

    # Step 1: Find the last user message — this one keeps its images
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    # Step 2: Rebuild the list, stripping images from all but the last user message
    result = []
    for i, msg in enumerate(messages):
        content = msg.get("content", "")

        # Last user message — keep everything, including images
        if i == last_user_idx:
            result.append(msg)
            continue

        # Multimodal message — remove image_url blocks, keep text blocks
        if isinstance(content, list):
            text_blocks = [b for b in content if b.get("type") != "image_url"]
            if text_blocks:
                # Keep the message with only its text blocks
                result.append({**msg, "content": text_blocks})
            else:
                # All content was images — replace with a placeholder so the
                # message slot is still present in the history
                result.append({**msg, "content": "(image)"})
        else:
            # Plain string — no images possible, keep as-is
            result.append(msg)

    return result

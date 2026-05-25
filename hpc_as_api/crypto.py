"""
stream/relay/crypto.py — End-to-End Encryption for the WebSocket Relay
=======================================================================

WHY THIS MODULE EXISTS
----------------------
The WebSocket relay forwards token payloads between two parties:

    Lakeshore GPU (producer)  →  relay server  →  STREAM middleware (consumer)

TLS (wss://) protects the *network link*, but the relay server itself can still
read the plaintext JSON that flows through it.  If the relay runs on a shared
VM or a third-party service, the operator sees every token — including sensitive
user data.

This module makes the relay a *dumb pipe*: the producer encrypts each message
before sending, and only the consumer can decrypt it.  The relay sees opaque
ciphertext and forwards it unchanged.

    Lakeshore → {"type":"enc","d":"X7fP2q..."} → relay → {"type":"enc","d":"X7fP2q..."} → STREAM
                                                   ↑
                                            relay sees nothing useful

ALGORITHM: AES-256-GCM
-----------------------
AES (Advanced Encryption Standard) is the global standard for symmetric
encryption — same key encrypts and decrypts.  "256" means the key is 256 bits
(32 bytes), which is the strongest AES variant.

GCM (Galois/Counter Mode) adds *authenticated encryption*: in addition to
hiding the plaintext, it appends a 16-byte authentication tag that lets the
receiver verify the message was not tampered with.  If even one byte of the
ciphertext is flipped in transit, decryption raises `InvalidTag` immediately.
This property is called AEAD — Authenticated Encryption with Associated Data.

THE NONCE
---------
GCM requires a 12-byte number that must NEVER be reused with the same key.
Reusing a nonce catastrophically breaks the encryption.  We generate a fresh
random nonce for every single message using os.urandom(12), which is
cryptographically secure.  The nonce is not secret — it's prepended to the
ciphertext so the receiver can use it for decryption.

WIRE FORMAT
-----------
Each encrypted message sent over the relay is a JSON string:

    {"type": "enc", "d": "<base64url(nonce[12] + ciphertext + authtag[16])>"}

The relay forwards this as-is (it doesn't inspect the "d" field).
The consumer strips the outer envelope, decodes the base64, splits off the
nonce, and decrypts to get back the original inner JSON:

    {"type": "token", "content": "Hello"}

BACKWARD COMPATIBILITY
----------------------
If RELAY_ENCRYPTION_KEY is not set (empty string), everything works exactly
as before — plaintext JSON flows through the relay.  Encryption is opt-in.
decrypt_message() is a passthrough for any message whose type is not "enc".

KEY GENERATION
--------------
Run once, paste into .env on BOTH the STREAM middleware AND Lakeshore side:

    python -c "import secrets; print(secrets.token_hex(32))"

Then in .env and in worker_init on Lakeshore:
    RELAY_ENCRYPTION_KEY=<output from above>

The key is 64 hex characters (0-9, a-f only) — no special characters, safe in
.env files, shell exports, and YAML without quoting.
"""

import base64
import json
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt_message(key_hex: str, plaintext_json: str) -> str:
    """Encrypt a JSON message string and return the relay wire format.

    Args:
        key_hex:        Hex-encoded 32-byte AES-256 key (from RELAY_ENCRYPTION_KEY).
        plaintext_json: The original JSON string, e.g. '{"type":"token","content":"Hi"}'.

    Returns:
        A JSON string ready to send over the relay:
        '{"type": "enc", "d": "<base64(nonce+ciphertext+tag)>"}'
    """
    key = bytes.fromhex(key_hex)  # 32 bytes
    nonce = os.urandom(12)  # Fresh random 12-byte nonce per message
    aesgcm = AESGCM(key)
    ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext_json.encode(), None)
    blob = base64.b64encode(nonce + ciphertext_with_tag).decode()
    return json.dumps({"type": "enc", "d": blob})


def decrypt_message(key_hex: str, msg_str: str) -> str:
    """Decrypt a relay message; pass through unchanged if not encrypted.

    Args:
        key_hex:  Hex-encoded 32-byte AES-256 key (from RELAY_ENCRYPTION_KEY).
        msg_str:  A JSON string received from the relay.

    Returns:
        The decrypted inner JSON string (if the message was encrypted),
        or the original msg_str unchanged (if type != "enc").

    Raises:
        cryptography.exceptions.InvalidTag: if the ciphertext was tampered with
            or the wrong key was used.
    """
    msg = json.loads(msg_str)

    if msg.get("type") != "enc":
        return msg_str

    key = bytes.fromhex(key_hex)
    blob = base64.b64decode(msg["d"])
    nonce = blob[:12]
    ciphertext_with_tag = blob[12:]
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, None)
    return plaintext.decode()

import json
import os
import urllib.request
import urllib.error
from cryptography.fernet import Fernet

SUPABASE_URL = "https://vsyjoxpfrcaxfxquuvzu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZzeWpveHBmcmNheGZ4cXV1dnp1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc0Njg1ODAsImV4cCI6MjA5MzA0NDU4MH0.zy9nNywmxexi7peK0uOaKHY5hKAoJwUMpgTO9YaAuiQ"

KEY_PATH = os.path.join(os.path.dirname(__file__), "..", ".vault_key")

def _get_or_create_key() -> bytes:
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(KEY_PATH, "wb") as f:
        f.write(key)
    return key

_fernet = Fernet(_get_or_create_key())

def store_sensitive_data(session_id: str, data: dict):
    """Encrypts and stores KYC/sensitive data securely in Supabase."""
    # 1. Encrypt locally (Zero-Knowledge for Supabase)
    json_data = json.dumps(data).encode('utf-8')
    encrypted = _fernet.encrypt(json_data).decode('utf-8')
    
    # 2. Push to Supabase via REST
    req = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/kyc_vault",
        data=json.dumps({
            "session_id": session_id,
            "encrypted_data": encrypted
        }).encode("utf-8"),
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        },
        method="POST"
    )
    
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        print(f"Failed to push to Supabase: {e}")


def retrieve_sensitive_data(session_id: str) -> dict:
    """Retrieves and decrypts sensitive data from Supabase."""
    req = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/kyc_vault?session_id=eq.{session_id}",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        },
        method="GET"
    )
    
    try:
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        
        if not data:
            return None
            
        encrypted = data[0]["encrypted_data"].encode('utf-8')
        decrypted = _fernet.decrypt(encrypted)
        return json.loads(decrypted)
        
    except Exception as e:
        print(f"Failed to retrieve/decrypt from Supabase: {e}")
        return None


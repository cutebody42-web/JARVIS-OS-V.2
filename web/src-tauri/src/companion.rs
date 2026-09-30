//! Native mobile companion identity, pairing and signed Brain RPC.
//!
//! The Ed25519 private seed is persisted through the Android native keyring
//! store (Android Keystore-backed) and is never returned to JavaScript.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use chrono::{DateTime, SecondsFormat, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use hmac::{Hmac, Mac};
use rand_core::OsRng;
use reqwest::{redirect::Policy, Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, time::Duration};
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

const MEDIA_TYPE: &str = "application/vnd.nexus-sync+json";
const AUTH_VERSION: u64 = 1;
const PAIRING_VERSION: u64 = 2;
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const SERVICE: &str = "ai.jarvis.app";
const KEY_DEVICE_ID: &str = "companion-device-id";
const KEY_ED25519: &str = "companion-ed25519";
const KEY_DESKTOP: &str = "paired-desktop";

#[derive(Debug, Clone, Deserialize)]
struct PairingOffer {
    version: u64,
    pairing_id: String,
    inviter_device: String,
    inviter_public_key: String,
    inviter_endpoint: String,
    secret: String,
    expires_at: String,
}

#[derive(Debug, Serialize)]
struct PairingRequest {
    version: u64,
    pairing_id: String,
    candidate_device: String,
    candidate_public_key: String,
    candidate_role: String,
    candidate_endpoint: Option<String>,
    proof: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PairedDesktop {
    version: u64,
    device_id: String,
    public_key: String,
    endpoint: String,
}

#[derive(Debug, Serialize)]
pub struct MobileIdentity {
    pub device_id: String,
    pub public_key: String,
}

#[derive(Debug, Serialize)]
pub struct MobilePairResult {
    pub state: String,
    pub device_id: String,
    pub desktop_device: String,
    pub endpoint: String,
}

#[derive(Debug, Serialize)]
pub struct MobilePairStatus {
    pub state: String,
    pub desktop_device: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MobileBrainReply {
    pub identity: String,
    pub text: String,
    pub lane: String,
    pub request_id: String,
}

#[cfg(target_os = "android")]
fn secure_entry(user: &str) -> Result<keyring_core::Entry, String> {
    keyring_core::Entry::new(SERVICE, user).map_err(|error| error.to_string())
}

#[cfg(not(target_os = "android"))]
fn secure_entry(_user: &str) -> Result<(), String> {
    Err("Secure companion identity is currently enabled for Android builds.".into())
}

#[cfg(target_os = "android")]
fn load_or_create_device_id() -> Result<String, String> {
    let entry = secure_entry(KEY_DEVICE_ID)?;
    match entry.get_password() {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) => Err("Stored companion device id is empty.".into()),
        Err(keyring_core::Error::NoEntry) => {
            let value = format!("phone-{}", Uuid::new_v4().simple());
            entry
                .set_password(&value)
                .map_err(|error| error.to_string())?;
            Ok(value)
        }
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(not(target_os = "android"))]
fn load_or_create_device_id() -> Result<String, String> {
    Err("Secure companion identity is currently enabled for Android builds.".into())
}

#[cfg(target_os = "android")]
fn load_or_create_signing_key() -> Result<SigningKey, String> {
    let entry = secure_entry(KEY_ED25519)?;
    match entry.get_secret() {
        Ok(bytes) => {
            let raw: [u8; 32] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| "Stored Ed25519 seed has invalid length.".to_string())?;
            Ok(SigningKey::from_bytes(&raw))
        }
        Err(keyring_core::Error::NoEntry) => {
            let key = SigningKey::generate(&mut OsRng);
            entry
                .set_secret(&key.to_bytes())
                .map_err(|error| error.to_string())?;
            Ok(key)
        }
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(not(target_os = "android"))]
fn load_or_create_signing_key() -> Result<SigningKey, String> {
    Err("Secure companion identity is currently enabled for Android builds.".into())
}

#[cfg(target_os = "android")]
fn save_paired_desktop(value: &PairedDesktop) -> Result<(), String> {
    let entry = secure_entry(KEY_DESKTOP)?;
    let encoded = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    entry
        .set_secret(&encoded)
        .map_err(|error| error.to_string())
}

#[cfg(not(target_os = "android"))]
fn save_paired_desktop(_value: &PairedDesktop) -> Result<(), String> {
    Err("Secure companion identity is currently enabled for Android builds.".into())
}

#[cfg(target_os = "android")]
fn load_paired_desktop() -> Result<Option<PairedDesktop>, String> {
    let entry = secure_entry(KEY_DESKTOP)?;
    match entry.get_secret() {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|error| error.to_string()),
        Err(keyring_core::Error::NoEntry) => Ok(None),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(not(target_os = "android"))]
fn load_paired_desktop() -> Result<Option<PairedDesktop>, String> {
    Ok(None)
}

#[cfg(target_os = "android")]
fn forget_paired_desktop() -> Result<(), String> {
    let entry = secure_entry(KEY_DESKTOP)?;
    match entry.delete_credential() {
        Ok(()) | Err(keyring_core::Error::NoEntry) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(not(target_os = "android"))]
fn forget_paired_desktop() -> Result<(), String> {
    Ok(())
}

fn b64url(bytes: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

fn decode_b64url(value: &str) -> Result<Vec<u8>, String> {
    URL_SAFE_NO_PAD
        .decode(value.as_bytes())
        .map_err(|_| "Invalid base64url value.".to_string())
}

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn canonical_json(value: &Value) -> Result<String, String> {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_string(value).map_err(|error| error.to_string())
        }
        Value::Array(items) => {
            let mut encoded = Vec::with_capacity(items.len());
            for item in items {
                encoded.push(canonical_json(item)?);
            }
            Ok(format!("[{}]", encoded.join(",")))
        }
        Value::Object(values) => {
            let ordered: BTreeMap<&String, &Value> = values.iter().collect();
            let mut fields = Vec::with_capacity(ordered.len());
            for (key, item) in ordered {
                let key = serde_json::to_string(key).map_err(|error| error.to_string())?;
                fields.push(format!("{key}:{}", canonical_json(item)?));
            }
            Ok(format!("{{{}}}", fields.join(",")))
        }
    }
}

fn validate_origin(value: &str) -> Result<String, String> {
    let url = Url::parse(value).map_err(|_| "Invalid desktop endpoint.".to_string())?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("Desktop endpoint must use HTTP(S).".into());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("Desktop endpoint cannot contain credentials.".into());
    }
    if url.path() != "/" || url.query().is_some() || url.fragment().is_some() {
        return Err("Desktop endpoint must be an origin without path/query/fragment.".into());
    }
    Ok(value.trim_end_matches('/').to_string())
}

fn http_client() -> Result<Client, String> {
    Client::builder()
        .redirect(Policy::none())
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|error| error.to_string())
}

fn parse_offer(value: &str) -> Result<PairingOffer, String> {
    let offer: PairingOffer =
        serde_json::from_str(value).map_err(|_| "Invalid JARVIS pairing offer.".to_string())?;
    if offer.version != PAIRING_VERSION {
        return Err("Unsupported JARVIS pairing version.".into());
    }
    if offer.pairing_id.is_empty()
        || offer.inviter_device.is_empty()
        || offer.inviter_public_key.is_empty()
        || offer.secret.len() < 32
    {
        return Err("JARVIS pairing offer is incomplete.".into());
    }
    validate_origin(&offer.inviter_endpoint)?;
    let expires = DateTime::parse_from_rfc3339(&offer.expires_at)
        .map_err(|_| "Pairing expiry is invalid.".to_string())?
        .with_timezone(&Utc);
    if Utc::now() > expires {
        return Err("JARVIS pairing offer has expired.".into());
    }
    let desktop_key = decode_b64url(&offer.inviter_public_key)?;
    if desktop_key.len() != 32 {
        return Err("Desktop Ed25519 public key is invalid.".into());
    }
    Ok(offer)
}

fn identity() -> Result<(String, SigningKey), String> {
    let device_id = load_or_create_device_id()?;
    let key = load_or_create_signing_key()?;
    Ok((device_id, key))
}

fn pairing_proof(
    offer: &PairingOffer,
    candidate_device: &str,
    public_key: &str,
) -> Result<String, String> {
    let unsigned = json!({
        "version": PAIRING_VERSION,
        "pairing_id": offer.pairing_id,
        "candidate_device": candidate_device,
        "candidate_public_key": public_key,
        "candidate_role": "companion",
        "candidate_endpoint": Value::Null,
    });
    let secret_hash = Sha256::digest(offer.secret.as_bytes());
    let hex_key = to_hex(&secret_hash);
    let mut mac =
        HmacSha256::new_from_slice(hex_key.as_bytes()).map_err(|_| "HMAC init failed.".to_string())?;
    mac.update(canonical_json(&unsigned)?.as_bytes());
    Ok(to_hex(&mac.finalize().into_bytes()))
}

fn make_signed_envelope(
    kind: &str,
    receiver: &str,
    message_id: &str,
    payload: Value,
    device_id: &str,
    key: &SigningKey,
) -> Result<Vec<u8>, String> {
    let unsigned = json!({
        "version": AUTH_VERSION,
        "kind": kind,
        "sender_device": device_id,
        "receiver_device": receiver,
        "message_id": message_id,
        "issued_at": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
        "payload": payload,
    });
    let signature = key.sign(canonical_json(&unsigned)?.as_bytes());
    let mut object = unsigned
        .as_object()
        .cloned()
        .ok_or_else(|| "Signed envelope construction failed.".to_string())?;
    object.insert(
        "signature".into(),
        Value::String(b64url(&signature.to_bytes())),
    );
    serde_json::to_vec(&Value::Object(object)).map_err(|error| error.to_string())
}

fn verify_signed_response(
    encoded: &[u8],
    expected_kind: &str,
    expected_sender: &str,
    expected_receiver: &str,
    expected_message_id: &str,
    public_key: &str,
) -> Result<Value, String> {
    if encoded.is_empty() || encoded.len() > MAX_RESPONSE_BYTES {
        return Err("JARVIS response exceeds the accepted size.".into());
    }
    let value: Value =
        serde_json::from_slice(encoded).map_err(|_| "Invalid signed JARVIS response.".to_string())?;
    let object = value
        .as_object()
        .ok_or_else(|| "Signed JARVIS response must be an object.".to_string())?;
    let expected_fields = [
        "version",
        "kind",
        "sender_device",
        "receiver_device",
        "message_id",
        "issued_at",
        "payload",
        "signature",
    ];
    if object.len() != expected_fields.len()
        || expected_fields.iter().any(|field| !object.contains_key(*field))
    {
        return Err("Signed JARVIS response schema mismatch.".into());
    }
    if object.get("version").and_then(Value::as_u64) != Some(AUTH_VERSION)
        || object.get("kind").and_then(Value::as_str) != Some(expected_kind)
        || object.get("sender_device").and_then(Value::as_str) != Some(expected_sender)
        || object.get("receiver_device").and_then(Value::as_str) != Some(expected_receiver)
        || object.get("message_id").and_then(Value::as_str) != Some(expected_message_id)
    {
        return Err("Signed JARVIS response identity mismatch.".into());
    }

    let issued_at = object
        .get("issued_at")
        .and_then(Value::as_str)
        .ok_or_else(|| "Signed JARVIS response timestamp missing.".to_string())?;
    let issued = DateTime::parse_from_rfc3339(issued_at)
        .map_err(|_| "Signed JARVIS response timestamp invalid.".to_string())?
        .with_timezone(&Utc);
    if (Utc::now() - issued).num_seconds().abs() > 15 * 60 {
        return Err("Signed JARVIS response is outside the clock window.".into());
    }

    let signature_text = object
        .get("signature")
        .and_then(Value::as_str)
        .ok_or_else(|| "Signed JARVIS response signature missing.".to_string())?;
    let signature_bytes = decode_b64url(signature_text)?;
    let signature_raw: [u8; 64] = signature_bytes
        .as_slice()
        .try_into()
        .map_err(|_| "Invalid Ed25519 signature length.".to_string())?;
    let signature = Signature::from_bytes(&signature_raw);

    let key_bytes = decode_b64url(public_key)?;
    let key_raw: [u8; 32] = key_bytes
        .as_slice()
        .try_into()
        .map_err(|_| "Invalid desktop Ed25519 key length.".to_string())?;
    let verifying_key =
        VerifyingKey::from_bytes(&key_raw).map_err(|_| "Invalid desktop Ed25519 key.".to_string())?;

    let mut unsigned = object.clone();
    unsigned.remove("signature");
    verifying_key
        .verify(
            canonical_json(&Value::Object(unsigned))?.as_bytes(),
            &signature,
        )
        .map_err(|_| "JARVIS response signature verification failed.".to_string())?;

    object
        .get("payload")
        .cloned()
        .ok_or_else(|| "Signed JARVIS response payload missing.".to_string())
}

async fn post_bytes(url: &str, body: Vec<u8>) -> Result<(StatusCode, Vec<u8>), String> {
    let response = http_client()?
        .post(url)
        .header("content-type", MEDIA_TYPE)
        .header("accept", MEDIA_TYPE)
        .body(body)
        .send()
        .await
        .map_err(|error| format!("JARVIS companion network error: {error}"))?;

    let status = response.status();
    if let Some(length) = response.content_length() {
        if length as usize > MAX_RESPONSE_BYTES {
            return Err("JARVIS response exceeds the accepted size.".into());
        }
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("JARVIS companion response error: {error}"))?;
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err("JARVIS response exceeds the accepted size.".into());
    }
    Ok((status, bytes.to_vec()))
}

#[tauri::command]
pub fn mobile_companion_identity() -> Result<MobileIdentity, String> {
    let (device_id, key) = identity()?;
    Ok(MobileIdentity {
        device_id,
        public_key: b64url(&key.verifying_key().to_bytes()),
    })
}

#[tauri::command]
pub async fn mobile_pair(offer_json: String) -> Result<MobilePairResult, String> {
    let offer = parse_offer(&offer_json)?;
    let (device_id, key) = identity()?;
    let public_key = b64url(&key.verifying_key().to_bytes());
    let proof = pairing_proof(&offer, &device_id, &public_key)?;

    let request = PairingRequest {
        version: PAIRING_VERSION,
        pairing_id: offer.pairing_id.clone(),
        candidate_device: device_id.clone(),
        candidate_public_key: public_key,
        candidate_role: "companion".into(),
        candidate_endpoint: None,
        proof,
    };
    let url = format!(
        "{}/nexus/pair/v1/request",
        validate_origin(&offer.inviter_endpoint)?
    );
    let response = http_client()?
        .post(url)
        .json(&request)
        .send()
        .await
        .map_err(|error| format!("JARVIS pairing network error: {error}"))?;

    if response.status() != StatusCode::ACCEPTED {
        return Err(match response.status() {
            StatusCode::FORBIDDEN => "JARVIS rejected the pairing proof.".into(),
            _ => format!("JARVIS pairing request failed ({})", response.status()),
        });
    }

    let desktop = PairedDesktop {
        version: 1,
        device_id: offer.inviter_device.clone(),
        public_key: offer.inviter_public_key.clone(),
        endpoint: validate_origin(&offer.inviter_endpoint)?,
    };
    save_paired_desktop(&desktop)?;

    Ok(MobilePairResult {
        state: "pending_owner_approval".into(),
        device_id,
        desktop_device: desktop.device_id,
        endpoint: desktop.endpoint,
    })
}

#[tauri::command]
pub async fn mobile_pair_status() -> Result<MobilePairStatus, String> {
    let Some(desktop) = load_paired_desktop()? else {
        return Ok(MobilePairStatus {
            state: "unpaired".into(),
            desktop_device: None,
        });
    };
    let (device_id, key) = identity()?;
    let nonce = Uuid::new_v4().simple().to_string();
    let message_id = format!("ping:{nonce}");
    let body = make_signed_envelope(
        "companion.ping",
        &desktop.device_id,
        &message_id,
        json!({"version": 1, "nonce": nonce}),
        &device_id,
        &key,
    )?;
    let url = format!("{}/nexus/companion/v1/ping", desktop.endpoint);
    let (status, response) = post_bytes(&url, body).await?;

    if status == StatusCode::FORBIDDEN {
        return Ok(MobilePairStatus {
            state: "pending_owner_approval".into(),
            desktop_device: Some(desktop.device_id),
        });
    }
    if status != StatusCode::OK {
        return Ok(MobilePairStatus {
            state: "unreachable".into(),
            desktop_device: Some(desktop.device_id),
        });
    }

    let payload = verify_signed_response(
        &response,
        "companion.pong",
        &desktop.device_id,
        &device_id,
        &format!("pong:{nonce}"),
        &desktop.public_key,
    )?;
    if payload.get("approved").and_then(Value::as_bool) != Some(true)
        || payload.get("identity").and_then(Value::as_str) != Some("JARVIS")
    {
        return Err("JARVIS companion approval response is invalid.".into());
    }

    Ok(MobilePairStatus {
        state: "paired".into(),
        desktop_device: Some(desktop.device_id),
    })
}

#[tauri::command]
pub async fn mobile_brain_message(
    message: String,
    task: Option<String>,
) -> Result<MobileBrainReply, String> {
    if message.trim().is_empty() || message.len() > 16_000 {
        return Err("JARVIS message must contain 1..16000 characters.".into());
    }
    if let Some(ref value) = task {
        if !matches!(value.as_str(), "general" | "realtime" | "coding") {
            return Err("Unsupported JARVIS task kind.".into());
        }
    }

    let desktop = load_paired_desktop()?
        .ok_or_else(|| "Pair this phone with your JARVIS desktop first.".to_string())?;
    let (device_id, key) = identity()?;
    let request_id = Uuid::new_v4().simple().to_string();
    let payload = json!({
        "version": 1,
        "request_id": request_id,
        "message": message,
        "task": task,
    });
    let body = make_signed_envelope(
        "brain.request",
        &desktop.device_id,
        &request_id,
        payload,
        &device_id,
        &key,
    )?;
    let url = format!("{}/nexus/brain/v1/message", desktop.endpoint);
    let (status, response) = post_bytes(&url, body).await?;

    if status == StatusCode::FORBIDDEN {
        return Err("This phone is waiting for desktop approval or was revoked.".into());
    }
    if status != StatusCode::OK {
        return Err(format!("JARVIS Brain request failed ({status})."));
    }

    let response_id = format!("brain-response:{request_id}");
    let payload = verify_signed_response(
        &response,
        "brain.response",
        &desktop.device_id,
        &device_id,
        &response_id,
        &desktop.public_key,
    )?;

    let identity = payload
        .get("identity")
        .and_then(Value::as_str)
        .ok_or_else(|| "JARVIS Brain response identity missing.".to_string())?;
    if identity != "JARVIS" {
        return Err("Remote Brain did not identify as JARVIS.".into());
    }
    if payload.get("request_id").and_then(Value::as_str) != Some(&request_id) {
        return Err("JARVIS Brain response request id mismatch.".into());
    }

    Ok(MobileBrainReply {
        identity: identity.into(),
        text: payload
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| "JARVIS Brain response text missing.".to_string())?
            .into(),
        lane: payload
            .get("lane")
            .and_then(Value::as_str)
            .unwrap_or("general")
            .into(),
        request_id,
    })
}

#[tauri::command]
pub fn mobile_forget_pairing() -> Result<(), String> {
    forget_paired_desktop()
}

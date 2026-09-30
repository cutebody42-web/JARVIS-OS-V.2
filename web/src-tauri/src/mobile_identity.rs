use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use hmac::{Hmac, Mac};
use rand_core::OsRng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

const PAIRING_VERSION: u8 = 2;
const AUTH_VERSION: u8 = 1;
const BRAIN_RPC_VERSION: u8 = 1;
const MEDIA_PATH: &str = "/nexus/brain/v1/message";

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PairingOffer {
    pub version: u8,
    pub pairing_id: String,
    pub inviter_device: String,
    pub inviter_public_key: String,
    pub inviter_endpoint: String,
    pub secret: String,
    pub expires_at: String,
}

#[derive(Debug, Serialize)]
pub struct MobileIdentity {
    pub device_id: String,
    pub public_key: String,
    pub fingerprint: String,
    pub key_protection: String,
}

#[derive(Debug, Serialize)]
pub struct MobilePairingBundle {
    pub identity: MobileIdentity,
    pub request: Value,
    pub status_url: String,
    pub submit_url: String,
}

#[derive(Debug, Serialize)]
pub struct MobileBrainRequest {
    pub request_id: String,
    pub endpoint: String,
    pub body: String,
}

#[derive(Debug, Serialize)]
pub struct MobileApprovalRequest {
    pub request_id: String,
    pub endpoint: String,
    pub body: String,
}

#[derive(Debug, Serialize)]
pub struct MobileCompanionStatus {
    pub identity: MobileIdentity,
    pub paired: bool,
    pub desktop_device: Option<String>,
    pub key_protection: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct DesktopTrust {
    desktop_device: String,
    desktop_public_key: String,
    desktop_endpoint: String,
}

fn local_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_local_data_dir()
        .map_err(|error| format!("JARVIS mobile storage is unavailable ({error})."))
}

fn seed_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(local_data_dir(app)?.join("mobile-device.ed25519"))
}

fn trust_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(local_data_dir(app)?.join("desktop-trust.json"))
}

fn ensure_private_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = local_data_dir(app)?;
    fs::create_dir_all(&dir)
        .map_err(|error| format!("JARVIS mobile storage could not be prepared ({error})."))?;
    Ok(dir)
}

#[cfg(unix)]
fn restrict_permissions(path: &PathBuf) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = fs::metadata(path)
        .map_err(|error| format!("JARVIS key permissions could not be read ({error})."))?
        .permissions();
    permissions.set_mode(0o600);
    fs::set_permissions(path, permissions)
        .map_err(|error| format!("JARVIS key permissions could not be applied ({error})."))
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &PathBuf) -> Result<(), String> {
    Ok(())
}

#[cfg(target_os = "android")]
fn signing_key(_app: &AppHandle) -> Result<SigningKey, String> {
    const SERVICE: &str = "ai.jarvis.app";
    const USER: &str = "mobile-ed25519";
    let entry = keyring_core::Entry::new(SERVICE, USER)
        .map_err(|error| format!("Android Keystore entry could not be opened ({error})."))?;
    match entry.get_secret() {
        Ok(bytes) => {
            let seed: [u8; 32] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| "Android Keystore JARVIS identity has invalid length.".to_string())?;
            Ok(SigningKey::from_bytes(&seed))
        }
        Err(keyring_core::Error::NoEntry) => {
            let key = SigningKey::generate(&mut OsRng);
            entry
                .set_secret(&key.to_bytes())
                .map_err(|error| format!("JARVIS identity could not be stored in Android Keystore ({error})."))?;
            Ok(key)
        }
        Err(error) => Err(format!("Android Keystore JARVIS identity could not be read ({error}).")),
    }
}

#[cfg(not(target_os = "android"))]
fn signing_key(app: &AppHandle) -> Result<SigningKey, String> {
    ensure_private_dir(app)?;
    let path = seed_path(app)?;
    if path.is_file() {
        let bytes = fs::read(&path)
            .map_err(|error| format!("JARVIS mobile identity could not be read ({error})."))?;
        let seed: [u8; 32] = bytes
            .try_into()
            .map_err(|_| "JARVIS mobile identity file is invalid.".to_string())?;
        return Ok(SigningKey::from_bytes(&seed));
    }

    let key = SigningKey::generate(&mut OsRng);
    fs::write(&path, key.to_bytes())
        .map_err(|error| format!("JARVIS mobile identity could not be stored ({error})."))?;
    restrict_permissions(&path)?;
    Ok(key)
}

fn key_protection() -> &'static str {
    if cfg!(target_os = "android") {
        "android_keystore"
    } else {
        "native_app_sandbox"
    }
}

fn hex(data: impl AsRef<[u8]>) -> String {
    data.as_ref().iter().map(|value| format!("{value:02x}")).collect()
}

fn public_b64(key: &SigningKey) -> String {
    URL_SAFE_NO_PAD.encode(key.verifying_key().as_bytes())
}

fn identity_for(key: &SigningKey) -> MobileIdentity {
    let public = key.verifying_key().to_bytes();
    let digest = Sha256::digest(public);
    let short = hex(&digest[..8]);
    let fingerprint_raw = hex(&digest[..12]);
    let fingerprint = fingerprint_raw
        .as_bytes()
        .chunks(4)
        .map(|chunk| std::str::from_utf8(chunk).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("-");
    MobileIdentity {
        device_id: format!("mobile-{short}"),
        public_key: URL_SAFE_NO_PAD.encode(public),
        fingerprint,
        key_protection: key_protection().to_string(),
    }
}

fn canonical(value: &Value) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|error| format!("JARVIS canonical JSON failed ({error})."))
}

fn offer_endpoint(offer: &PairingOffer) -> Result<String, String> {
    let endpoint = offer.inviter_endpoint.trim().trim_end_matches('/');
    if !(endpoint.starts_with("http://") || endpoint.starts_with("https://")) {
        return Err("JARVIS pairing endpoint must use HTTP(S).".into());
    }
    if endpoint.contains('@') || endpoint.contains('?') || endpoint.contains('#') {
        return Err("JARVIS pairing endpoint is invalid.".into());
    }
    Ok(endpoint.to_string())
}

fn pairing_unsigned(
    offer: &PairingOffer,
    identity: &MobileIdentity,
) -> Value {
    let mut map = BTreeMap::<String, Value>::new();
    map.insert("candidate_device".into(), json!(identity.device_id));
    map.insert("candidate_endpoint".into(), Value::Null);
    map.insert("candidate_public_key".into(), json!(identity.public_key));
    map.insert("candidate_role".into(), json!("companion"));
    map.insert("pairing_id".into(), json!(offer.pairing_id));
    map.insert("version".into(), json!(PAIRING_VERSION));
    serde_json::to_value(map).expect("BTreeMap JSON serialization cannot fail")
}

fn pairing_request_value(
    offer: &PairingOffer,
    identity: &MobileIdentity,
) -> Result<Value, String> {
    if offer.version != PAIRING_VERSION {
        return Err("Unsupported JARVIS pairing invitation version.".into());
    }
    let unsigned = pairing_unsigned(offer, identity);
    let secret_hash = Sha256::digest(offer.secret.as_bytes());
    let secret_hex = hex(secret_hash);
    let mut mac = HmacSha256::new_from_slice(secret_hex.as_bytes())
        .map_err(|_| "JARVIS pairing proof could not initialize.".to_string())?;
    mac.update(&canonical(&unsigned)?);
    let proof = hex(mac.finalize().into_bytes());

    let mut object = unsigned
        .as_object()
        .cloned()
        .ok_or_else(|| "JARVIS pairing request is invalid.".to_string())?;
    object.insert("proof".into(), json!(proof));
    Ok(Value::Object(object))
}

fn decode_public(value: &str) -> Result<VerifyingKey, String> {
    let bytes = URL_SAFE_NO_PAD
        .decode(value)
        .map_err(|_| "JARVIS public key is invalid.".to_string())?;
    let raw: [u8; 32] = bytes
        .try_into()
        .map_err(|_| "JARVIS public key length is invalid.".to_string())?;
    VerifyingKey::from_bytes(&raw)
        .map_err(|_| "JARVIS public key is invalid.".to_string())
}

fn verify_signed_envelope(
    encoded: &str,
    public_key: &str,
    expected_kind: &str,
    expected_sender: &str,
    expected_receiver: &str,
    expected_message_id: &str,
) -> Result<Value, String> {
    let value: Value = serde_json::from_str(encoded)
        .map_err(|_| "JARVIS signed response is invalid JSON.".to_string())?;
    let object = value
        .as_object()
        .ok_or_else(|| "JARVIS signed response must be an object.".to_string())?;

    let required = [
        "version", "kind", "sender_device", "receiver_device",
        "message_id", "issued_at", "payload", "signature",
    ];
    if object.len() != required.len() || required.iter().any(|key| !object.contains_key(*key)) {
        return Err("JARVIS signed response schema mismatch.".into());
    }
    if object.get("version").and_then(Value::as_u64) != Some(AUTH_VERSION as u64)
        || object.get("kind").and_then(Value::as_str) != Some(expected_kind)
        || object.get("sender_device").and_then(Value::as_str) != Some(expected_sender)
        || object.get("receiver_device").and_then(Value::as_str) != Some(expected_receiver)
        || object.get("message_id").and_then(Value::as_str) != Some(expected_message_id)
    {
        return Err("JARVIS signed response identity mismatch.".into());
    }

    let signature_text = object
        .get("signature")
        .and_then(Value::as_str)
        .ok_or_else(|| "JARVIS response signature is missing.".to_string())?;
    let signature_bytes = URL_SAFE_NO_PAD
        .decode(signature_text)
        .map_err(|_| "JARVIS response signature is invalid.".to_string())?;
    let signature = Signature::from_slice(&signature_bytes)
        .map_err(|_| "JARVIS response signature is invalid.".to_string())?;

    let mut unsigned = BTreeMap::<String, Value>::new();
    for key in required.iter().filter(|key| **key != "signature") {
        unsigned.insert(
            (*key).to_string(),
            object.get(*key).cloned().unwrap_or(Value::Null),
        );
    }
    decode_public(public_key)?
        .verify(
            &canonical(&serde_json::to_value(unsigned).map_err(|error| error.to_string())?)?,
            &signature,
        )
        .map_err(|_| "JARVIS response signature verification failed.".to_string())?;

    Ok(object.get("payload").cloned().unwrap_or(Value::Null))
}

fn save_trust(app: &AppHandle, trust: &DesktopTrust) -> Result<(), String> {
    ensure_private_dir(app)?;
    let path = trust_path(app)?;
    let encoded = serde_json::to_vec(trust)
        .map_err(|error| format!("JARVIS desktop trust could not be encoded ({error})."))?;
    fs::write(&path, encoded)
        .map_err(|error| format!("JARVIS desktop trust could not be stored ({error})."))?;
    restrict_permissions(&path)
}

fn load_trust(app: &AppHandle) -> Result<DesktopTrust, String> {
    let data = fs::read(trust_path(app)?)
        .map_err(|_| "This mobile JARVIS is not paired with a desktop Brain.".to_string())?;
    serde_json::from_slice(&data)
        .map_err(|_| "Stored JARVIS desktop trust is invalid.".to_string())
}

#[tauri::command]
pub fn mobile_identity(app: AppHandle) -> Result<MobileIdentity, String> {
    let key = signing_key(&app)?;
    Ok(identity_for(&key))
}

#[tauri::command]
pub fn mobile_companion_status(app: AppHandle) -> Result<MobileCompanionStatus, String> {
    let key = signing_key(&app)?;
    let identity = identity_for(&key);
    let trust = load_trust(&app).ok();
    Ok(MobileCompanionStatus {
        key_protection: identity.key_protection.clone(),
        identity,
        paired: trust.is_some(),
        desktop_device: trust.map(|value| value.desktop_device),
    })
}

#[tauri::command]
pub fn mobile_prepare_pairing(
    app: AppHandle,
    offer: PairingOffer,
) -> Result<MobilePairingBundle, String> {
    let key = signing_key(&app)?;
    let identity = identity_for(&key);
    let endpoint = offer_endpoint(&offer)?;
    let request = pairing_request_value(&offer, &identity)?;
    Ok(MobilePairingBundle {
        identity,
        request,
        submit_url: format!("{endpoint}/nexus/pair/v1/request"),
        status_url: format!("{endpoint}/nexus/pair/v1/status"),
    })
}

#[tauri::command]
pub fn mobile_accept_pairing_approval(
    app: AppHandle,
    offer: PairingOffer,
    signed_approval: String,
) -> Result<MobileIdentity, String> {
    let key = signing_key(&app)?;
    let identity = identity_for(&key);
    let payload = verify_signed_envelope(
        &signed_approval,
        &offer.inviter_public_key,
        "pair.approved",
        &offer.inviter_device,
        &identity.device_id,
        &format!("pair-approved:{}", offer.pairing_id),
    )?;
    if payload.get("version").and_then(Value::as_u64) != Some(PAIRING_VERSION as u64)
        || payload.get("pairing_id").and_then(Value::as_str) != Some(offer.pairing_id.as_str())
        || payload.get("state").and_then(Value::as_str) != Some("approved")
        || payload.get("desktop_device").and_then(Value::as_str) != Some(offer.inviter_device.as_str())
        || payload.get("desktop_public_key").and_then(Value::as_str) != Some(offer.inviter_public_key.as_str())
    {
        return Err("JARVIS pairing approval payload mismatch.".into());
    }

    let desktop_endpoint = offer_endpoint(&offer)?;
    save_trust(
        &app,
        &DesktopTrust {
            desktop_device: offer.inviter_device,
            desktop_public_key: offer.inviter_public_key,
            desktop_endpoint,
        },
    )?;
    Ok(identity)
}

#[tauri::command]
pub fn mobile_sign_brain_request(
    app: AppHandle,
    message: String,
    task: Option<String>,
) -> Result<MobileBrainRequest, String> {
    let clean = message.trim();
    if clean.is_empty() || clean.chars().count() > 16_000 {
        return Err("JARVIS message is empty or too long.".into());
    }
    if let Some(ref value) = task {
        if !matches!(value.as_str(), "general" | "realtime" | "coding") {
            return Err("Unsupported JARVIS task kind.".into());
        }
    }

    let trust = load_trust(&app)?;
    let key = signing_key(&app)?;
    let identity = identity_for(&key);
    let request_id = Uuid::new_v4().simple().to_string();

    let mut payload = BTreeMap::<String, Value>::new();
    payload.insert("message".into(), json!(clean));
    payload.insert("request_id".into(), json!(request_id));
    payload.insert("task".into(), task.map(Value::String).unwrap_or(Value::Null));
    payload.insert("version".into(), json!(BRAIN_RPC_VERSION));

    let issued_at = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .map_err(|error| format!("JARVIS request timestamp failed ({error})."))?;

    let mut unsigned = BTreeMap::<String, Value>::new();
    unsigned.insert("issued_at".into(), json!(issued_at));
    unsigned.insert("kind".into(), json!("brain.request"));
    unsigned.insert("message_id".into(), json!(request_id));
    unsigned.insert("payload".into(), serde_json::to_value(payload).map_err(|error| error.to_string())?);
    unsigned.insert("receiver_device".into(), json!(trust.desktop_device));
    unsigned.insert("sender_device".into(), json!(identity.device_id));
    unsigned.insert("version".into(), json!(AUTH_VERSION));

    let unsigned_value = serde_json::to_value(&unsigned).map_err(|error| error.to_string())?;
    let signature = URL_SAFE_NO_PAD.encode(key.sign(&canonical(&unsigned_value)?).to_bytes());
    let mut envelope = unsigned;
    envelope.insert("signature".into(), json!(signature));
    let body = serde_json::to_string(&envelope)
        .map_err(|error| format!("JARVIS request encoding failed ({error})."))?;

    Ok(MobileBrainRequest {
        request_id,
        endpoint: format!("{}{}", trust.desktop_endpoint.trim_end_matches('/'), MEDIA_PATH),
        body,
    })
}

#[tauri::command]
pub fn mobile_verify_brain_response(
    app: AppHandle,
    request_id: String,
    signed_response: String,
) -> Result<Value, String> {
    let trust = load_trust(&app)?;
    let key = signing_key(&app)?;
    let identity = identity_for(&key);
    let payload = verify_signed_envelope(
        &signed_response,
        &trust.desktop_public_key,
        "brain.response",
        &trust.desktop_device,
        &identity.device_id,
        &format!("brain-response:{request_id}"),
    )?;
    if payload.get("version").and_then(Value::as_u64) != Some(BRAIN_RPC_VERSION as u64)
        || payload.get("request_id").and_then(Value::as_str) != Some(request_id.as_str())
        || payload.get("identity").and_then(Value::as_str) != Some("JARVIS")
    {
        return Err("JARVIS Brain response payload mismatch.".into());
    }
    Ok(payload)
}

fn sign_companion_request(
    app: &AppHandle,
    kind: &str,
    message_id: &str,
    payload: Value,
    path: &str,
) -> Result<(String, String), String> {
    let trust = load_trust(app)?;
    let key = signing_key(app)?;
    let identity = identity_for(&key);
    let issued_at = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .map_err(|error| format!("JARVIS approval timestamp failed ({error})."))?;

    let mut unsigned = BTreeMap::<String, Value>::new();
    unsigned.insert("issued_at".into(), json!(issued_at));
    unsigned.insert("kind".into(), json!(kind));
    unsigned.insert("message_id".into(), json!(message_id));
    unsigned.insert("payload".into(), payload);
    unsigned.insert("receiver_device".into(), json!(trust.desktop_device));
    unsigned.insert("sender_device".into(), json!(identity.device_id));
    unsigned.insert("version".into(), json!(AUTH_VERSION));

    let unsigned_value = serde_json::to_value(&unsigned).map_err(|error| error.to_string())?;
    let signature = URL_SAFE_NO_PAD.encode(key.sign(&canonical(&unsigned_value)?).to_bytes());
    let mut envelope = unsigned;
    envelope.insert("signature".into(), json!(signature));
    let body = serde_json::to_string(&envelope)
        .map_err(|error| format!("JARVIS approval request encoding failed ({error})."))?;
    Ok((
        format!("{}{}", trust.desktop_endpoint.trim_end_matches('/'), path),
        body,
    ))
}

#[tauri::command]
pub fn mobile_sign_approval_list(app: AppHandle) -> Result<MobileApprovalRequest, String> {
    let request_id = Uuid::new_v4().simple().to_string();
    let payload = json!({
        "version": 1,
        "request_id": request_id,
    });
    let message_id = format!("approval-list:{request_id}");
    let (endpoint, body) = sign_companion_request(
        &app,
        "approval.list",
        &message_id,
        payload,
        "/nexus/approval/v1/pending",
    )?;
    Ok(MobileApprovalRequest { request_id, endpoint, body })
}

#[tauri::command]
pub fn mobile_verify_approval_list_response(
    app: AppHandle,
    request_id: String,
    signed_response: String,
) -> Result<Value, String> {
    let trust = load_trust(&app)?;
    let key = signing_key(&app)?;
    let identity = identity_for(&key);
    let payload = verify_signed_envelope(
        &signed_response,
        &trust.desktop_public_key,
        "approval.pending",
        &trust.desktop_device,
        &identity.device_id,
        &format!("approval-pending:{request_id}"),
    )?;
    if payload.get("version").and_then(Value::as_u64) != Some(1)
        || payload.get("request_id").and_then(Value::as_str) != Some(request_id.as_str())
        || !payload.get("pending").map(Value::is_array).unwrap_or(false)
    {
        return Err("JARVIS approval list response mismatch.".into());
    }
    Ok(payload)
}

#[tauri::command]
pub fn mobile_sign_approval_decision(
    app: AppHandle,
    approval_id: String,
    approved: bool,
) -> Result<MobileApprovalRequest, String> {
    let approval_id = approval_id.trim().to_string();
    if approval_id.is_empty() || approval_id.len() > 128 {
        return Err("Invalid JARVIS approval id.".into());
    }
    let request_id = Uuid::new_v4().simple().to_string();
    let payload = json!({
        "version": 1,
        "request_id": request_id,
        "approval_id": approval_id,
        "approved": approved,
        "user_verified": true,
    });
    let message_id = format!("approval-decision:{request_id}");
    let (endpoint, body) = sign_companion_request(
        &app,
        "approval.decision",
        &message_id,
        payload,
        "/nexus/approval/v1/decision",
    )?;
    Ok(MobileApprovalRequest { request_id, endpoint, body })
}

#[tauri::command]
pub fn mobile_verify_approval_receipt(
    app: AppHandle,
    request_id: String,
    signed_response: String,
) -> Result<Value, String> {
    let trust = load_trust(&app)?;
    let key = signing_key(&app)?;
    let identity = identity_for(&key);
    let payload = verify_signed_envelope(
        &signed_response,
        &trust.desktop_public_key,
        "approval.receipt",
        &trust.desktop_device,
        &identity.device_id,
        &format!("approval-receipt:{request_id}"),
    )?;
    if payload.get("version").and_then(Value::as_u64) != Some(1)
        || payload.get("request_id").and_then(Value::as_str) != Some(request_id.as_str())
        || !matches!(
            payload.get("state").and_then(Value::as_str),
            Some("approved") | Some("rejected")
        )
    {
        return Err("JARVIS approval receipt mismatch.".into());
    }
    Ok(payload)
}


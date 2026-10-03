//! Native transport for the explicitly paired desktop, including tailnet HTTP.
//!
//! WebView fetch is subject to CSP, CORS and mixed-content restrictions. These
//! requests stay in Rust; the WebView receives only bounded response data. The
//! caller cannot select arbitrary origins, paths, headers or redirect targets.

use std::time::Duration;

use reqwest::{redirect::Policy, Url};
use serde::Serialize;
use tauri::AppHandle;

use super::mobile_identity::{offer_endpoint, trusted_desktop_endpoint, PairingOffer};

const NEXUS_MEDIA_TYPE: &str = "application/vnd.nexus-sync+json";
const MAX_BODY_BYTES: usize = 1024 * 1024;

#[derive(Debug, Serialize)]
pub struct CompanionHttpResponse {
    status: u16,
    content_type: String,
    body: String,
}

pub(crate) fn normalize_companion_origin(value: &str) -> Result<String, String> {
    let url = Url::parse(value.trim())
        .map_err(|_| "JARVIS companion endpoint is invalid.".to_string())?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err("JARVIS companion endpoint must be an HTTP(S) origin.".into());
    }
    Ok(url.as_str().trim_end_matches('/').to_string())
}

fn companion_target(endpoint: &str, origin: &str, pairing: bool) -> Result<Url, String> {
    let base = normalize_companion_origin(origin)?;
    let routes: &[&str] = if pairing {
        &["/nexus/pair/v1/request", "/nexus/pair/v1/status"]
    } else {
        &[
            "/nexus/brain/v1/message",
            "/nexus/approval/v1/pending",
            "/nexus/approval/v1/decision",
        ]
    };
    let target = Url::parse(endpoint)
        .map_err(|_| "JARVIS companion request URL is invalid.".to_string())?;
    if !routes.iter().any(|path| {
        Url::parse(&format!("{base}{path}"))
            .map(|allowed| allowed == target)
            .unwrap_or(false)
    }) {
        return Err("JARVIS request does not target the paired desktop route.".into());
    }
    Ok(target)
}

#[tauri::command]
pub async fn mobile_companion_post(
    app: AppHandle,
    endpoint: String,
    body: String,
    offer: Option<PairingOffer>,
) -> Result<CompanionHttpResponse, String> {
    let pairing = offer.is_some();
    let origin = match offer {
        Some(ref invitation) => offer_endpoint(invitation)?,
        None => trusted_desktop_endpoint(&app)?,
    };
    let target = companion_target(&endpoint, &origin, pairing)?;
    let request_limit = if pairing { 16 * 1024 } else { MAX_BODY_BYTES };
    if body.is_empty() || body.len() > request_limit {
        return Err("JARVIS companion request exceeds its byte budget.".into());
    }
    let content_type = if pairing { "application/json" } else { NEXUS_MEDIA_TYPE };
    let client = reqwest::Client::builder()
        .redirect(Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|_| "JARVIS companion network could not initialize.".to_string())?;
    let mut response = client
        .post(target)
        .header(reqwest::header::CONTENT_TYPE, content_type)
        .header(reqwest::header::ACCEPT, content_type)
        .body(body)
        .send()
        .await
        .map_err(|_| "JARVIS cannot reach your laptop. Check its Brain and phone network/VPN connection.".to_string())?;
    let status = response.status().as_u16();
    let response_type = response.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_string();
    if response.content_length().is_some_and(|size| size > MAX_BODY_BYTES as u64) {
        return Err("JARVIS companion response exceeds its byte budget.".into());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await
        .map_err(|_| "JARVIS companion response could not be read.".to_string())?
    {
        if bytes.len() + chunk.len() > MAX_BODY_BYTES {
            return Err("JARVIS companion response exceeds its byte budget.".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    let body = String::from_utf8(bytes)
        .map_err(|_| "JARVIS companion response is invalid text.".to_string())?;
    Ok(CompanionHttpResponse { status, content_type: response_type, body })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tailnet_http_and_ipv6_origins_are_supported() {
        assert_eq!(normalize_companion_origin("http://100.90.10.3:8765/").unwrap(), "http://100.90.10.3:8765");
        assert!(normalize_companion_origin("http://[fd7a:115c:a1e0::1]:8765").is_ok());
        assert!(normalize_companion_origin("https://jarvis.example.test").is_ok());
    }

    #[test]
    fn url_confusion_cannot_change_the_trusted_origin() {
        for invalid in [
            "file:///tmp/test", "http://owner:password@100.90.10.3:8765",
            "http://100.90.10.3:8765/path", "http://100.90.10.3:8765?to=evil",
            "http://100.90.10.3:8765#evil", "http://100.90.10.3:99999",
        ] {
            assert!(normalize_companion_origin(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn requests_are_bound_to_origin_and_route() {
        let origin = "http://100.90.10.3:8765";
        assert!(companion_target(&format!("{origin}/nexus/approval/v1/decision"), origin, false).is_ok());
        for invalid in [
            "http://100.90.10.4:8765/nexus/approval/v1/decision",
            "http://100.90.10.3:8766/nexus/approval/v1/decision",
            "http://100.90.10.3:8765/v1/setup/local-brain",
            "http://100.90.10.3:8765/nexus/approval/v1/decision?redirect=evil",
        ] {
            assert!(companion_target(invalid, origin, false).is_err(), "{invalid}");
        }
    }

    #[test]
    fn pairing_invitation_does_not_authorize_brain_or_approval_routes() {
        let origin = "http://100.90.10.3:8765";
        assert!(companion_target(&format!("{origin}/nexus/pair/v1/request"), origin, true).is_ok());
        assert!(companion_target(&format!("{origin}/nexus/pair/v1/status"), origin, true).is_ok());
        assert!(companion_target(&format!("{origin}/nexus/brain/v1/message"), origin, true).is_err());
        assert!(companion_target(&format!("{origin}/nexus/pair/v1/request"), origin, false).is_err());
    }
}

#![cfg(target_os = "android")]

use serde::{Deserialize, Serialize};
use tauri::{
    plugin::{Builder, PluginHandle, TauriPlugin},
    Manager, Runtime,
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FingerprintStatus {
    pub is_available: bool,
    pub biometry_type: u8,
    pub error: Option<String>,
}

#[derive(Serialize)]
struct AuthenticateRequest {
    reason: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FingerprintProof {
    user_verified: bool,
    biometry_type: String,
}

pub struct Fingerprint<R: Runtime>(PluginHandle<R>);

impl<R: Runtime> Fingerprint<R> {
    pub fn status(&self) -> Result<FingerprintStatus, String> {
        self.0
            .run_mobile_plugin("status", ())
            .map_err(|_| "JARVIS could not read fingerprint availability.".to_string())
    }

    pub fn authenticate(&self, reason: String) -> Result<(), String> {
        let proof: FingerprintProof = self
            .0
            .run_mobile_plugin("authenticate", AuthenticateRequest { reason })
            .map_err(|_| "JARVIS fingerprint verification failed or was cancelled.".to_string())?;
        if !proof.user_verified || proof.biometry_type != "fingerprint" {
            return Err("JARVIS requires fingerprint sensor verification.".into());
        }
        Ok(())
    }
}

pub trait FingerprintExt<R: Runtime> {
    fn fingerprint(&self) -> &Fingerprint<R>;
}

impl<R: Runtime, T: Manager<R>> FingerprintExt<R> for T {
    fn fingerprint(&self) -> &Fingerprint<R> {
        self.state::<Fingerprint<R>>().inner()
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("jarvis-fingerprint")
        .setup(|app, api| {
            let handle =
                api.register_android_plugin("ai.jarvis.fingerprint", "FingerprintPlugin")?;
            app.manage(Fingerprint(handle));
            Ok(())
        })
        .build()
}

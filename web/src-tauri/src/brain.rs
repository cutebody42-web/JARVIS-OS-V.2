use serde::Serialize;
use std::sync::Mutex;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrainConnection {
    pub endpoint: String,
    pub token: String,
}

#[derive(Default)]
pub struct BrainState(pub Mutex<Option<BrainConnection>>);

#[tauri::command]
pub fn platform_mode() -> &'static str {
    if cfg!(mobile) { "mobile" } else { "desktop" }
}

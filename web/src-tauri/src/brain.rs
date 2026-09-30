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


#[cfg(desktop)]
#[tauri::command]
pub fn ensure_brain_sidecar(
    app: tauri::AppHandle,
    state: tauri::State<'_, BrainState>,
) -> Result<BrainConnection, String> {
    use std::net::TcpListener;
    use tauri_plugin_shell::ShellExt;

    let mut guard = state
        .0
        .lock()
        .map_err(|_| "JARVIS Brain state lock was poisoned".to_string())?;

    if let Some(connection) = guard.as_ref() {
        return Ok(connection.clone());
    }

    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|error| format!("Could not reserve local JARVIS Brain port: {error}"))?;
    let port = listener
        .local_addr()
        .map_err(|error| format!("Could not inspect local JARVIS Brain port: {error}"))?
        .port();
    drop(listener);

    let token = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let parent_pid = std::process::id().to_string();
    let port_arg = port.to_string();

    let command = app
        .shell()
        .sidecar("binaries/jarvis-brain")
        .map_err(|error| format!("JARVIS Brain sidecar is unavailable: {error}"))?
        .args([
            "--port",
            port_arg.as_str(),
            "--ui-token",
            token.as_str(),
            "--parent-pid",
            parent_pid.as_str(),
        ]);

    command
        .spawn()
        .map_err(|error| format!("Could not start JARVIS Brain: {error}"))?;

    let connection = BrainConnection {
        endpoint: format!("http://127.0.0.1:{port}"),
        token,
    };
    *guard = Some(connection.clone());
    Ok(connection)
}

#[cfg(mobile)]
#[tauri::command]
pub fn ensure_brain_sidecar() -> Result<BrainConnection, String> {
    Err("Mobile JARVIS connects to the paired desktop Brain.".to_string())
}

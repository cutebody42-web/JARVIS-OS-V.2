use serde::Serialize;
use std::sync::Mutex;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrainConnection {
    pub endpoint: String,
    pub token: String,
}

#[derive(Default)]
pub struct BrainState(Mutex<BrainLifecycle>);

#[derive(Default)]
struct BrainLifecycle {
    connection: Option<BrainConnection>,
    #[cfg(desktop)]
    child: Option<tauri_plugin_shell::process::CommandChild>,
    starts: u8,
    stopping: bool,
}

impl BrainState {
    pub fn stop_recovery(&self) -> Result<(), String> {
        self.0.lock()
            .map_err(|_| "JARVIS Brain state lock was poisoned".to_string())?
            .stopping = true;
        Ok(())
    }
}

impl BrainLifecycle {
    fn may_start(&self) -> Result<(), String> {
        if self.stopping {
            return Err("JARVIS is shutting down for an update; Brain recovery is paused.".into());
        }
        // One initial start and at most three automatic replacements per app
        // session. Repeated startup failures need an explicit desktop restart.
        if self.starts >= 4 {
            return Err("JARVIS Brain stopped repeatedly. Reopen JARVIS after checking the installation.".into());
        }
        Ok(())
    }

    fn record_termination(&mut self, token: &str) {
        // A late event from an old child must not clear a newer connection.
        if self.connection.as_ref().is_some_and(|value| value.token == token) {
            self.connection = None;
            #[cfg(desktop)]
            { self.child = None; }
        }
    }
}

#[tauri::command]
pub fn platform_mode() -> &'static str {
    if cfg!(mobile) { "mobile" } else { "desktop" }
}


#[cfg(desktop)]
#[tauri::command]
pub fn choose_model_store() -> Result<Option<String>, String> {
    let selected = rfd::FileDialog::new()
        .set_title("Choose JARVIS local model folder")
        .pick_folder();
    Ok(selected.map(|path| path.to_string_lossy().into_owned()))
}

#[cfg(mobile)]
#[tauri::command]
pub fn choose_model_store() -> Result<Option<String>, String> {
    Err("Local model storage is configured on the desktop JARVIS device.".to_string())
}

#[cfg(desktop)]
#[tauri::command]
pub fn choose_model_file() -> Result<Option<String>, String> {
    let selected = rfd::FileDialog::new()
        .set_title("Choose a local GGUF expert model")
        .add_filter("GGUF models", &["gguf"])
        .pick_file();
    Ok(selected.map(|path| path.to_string_lossy().into_owned()))
}

#[cfg(mobile)]
#[tauri::command]
pub fn choose_model_file() -> Result<Option<String>, String> {
    Err("Import local models on your laptop JARVIS device.".to_string())
}


#[cfg(desktop)]
#[tauri::command]
pub fn ensure_brain_sidecar(
    app: tauri::AppHandle,
    state: tauri::State<'_, BrainState>,
) -> Result<BrainConnection, String> {
    use std::net::TcpListener;
    use tauri::Manager;
    use tauri_plugin_shell::process::CommandEvent;
    use tauri_plugin_shell::ShellExt;

    let mut guard = state
        .0
        .lock()
        .map_err(|_| "JARVIS Brain state lock was poisoned".to_string())?;

    if guard.stopping {
        return Err("JARVIS is shutting down for an update; Brain recovery is paused.".into());
    }
    if let Some(connection) = guard.connection.as_ref() {
        return Ok(connection.clone());
    }
    guard.may_start()?;

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

    let (mut events, child) = command
        .spawn()
        .map_err(|error| format!("Could not start JARVIS Brain: {error}"))?;

    let connection = BrainConnection {
        endpoint: format!("http://127.0.0.1:{port}"),
        token,
    };
    guard.starts += 1;
    guard.child = Some(child);
    guard.connection = Some(connection.clone());
    let watched_token = connection.token.clone();
    let watched_app = app.clone();
    tauri::async_runtime::spawn(async move {
        // Drain stdout/stderr so backpressure cannot stall the sidecar. Only a
        // confirmed termination permits replacement; slow status/model calls
        // and HTTP timeouts never kill an otherwise running Brain.
        while let Some(event) = events.recv().await {
            if matches!(event, CommandEvent::Terminated(_)) {
                let state = watched_app.state::<BrainState>();
                if let Ok(mut lifecycle) = state.0.lock() {
                    lifecycle.record_termination(&watched_token);
                }
                break;
            }
        }
    });
    Ok(connection)
}

#[cfg(mobile)]
#[tauri::command]
pub fn ensure_brain_sidecar() -> Result<BrainConnection, String> {
    Err("Mobile JARVIS connects to the paired desktop Brain.".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_current_child_termination_clears_the_connection() {
        let mut lifecycle = BrainLifecycle::default();
        lifecycle.connection = Some(BrainConnection { endpoint: "http://127.0.0.1:1".into(), token: "fresh".into() });
        lifecycle.record_termination("old");
        assert!(lifecycle.connection.is_some());
        lifecycle.record_termination("fresh");
        assert!(lifecycle.connection.is_none());
    }

    #[test]
    fn recovery_is_bounded_and_cannot_start_during_update_shutdown() {
        let mut lifecycle = BrainLifecycle::default();
        assert!(lifecycle.may_start().is_ok());
        lifecycle.starts = 4;
        assert!(lifecycle.may_start().is_err());
        lifecycle.starts = 0;
        lifecycle.stopping = true;
        assert!(lifecycle.may_start().is_err());
    }
}

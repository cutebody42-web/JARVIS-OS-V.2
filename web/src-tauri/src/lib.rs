mod brain;

use brain::{ensure_brain_sidecar, platform_mode, BrainState};
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default()
        .manage(BrainState::default())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_websocket::init())
        .plugin(tauri_plugin_deep_link::init())
        .setup(|app| {
            let salt_dir = app
                .path()
                .app_local_data_dir()
                .map_err(|error| error.to_string())?;
            std::fs::create_dir_all(&salt_dir)?;
            let salt_path = salt_dir.join("stronghold.salt");
            app.handle().plugin(
                tauri_plugin_stronghold::Builder::with_argon2(&salt_path).build(),
            )?;
            Ok(())
        });

    #[cfg(desktop)]
    let builder = builder.plugin(tauri_plugin_shell::init());

    #[cfg(mobile)]
    let builder = builder
        .plugin(tauri_plugin_biometric::init())
        .plugin(tauri_plugin_barcode_scanner::init())
        .plugin(tauri_plugin_haptics::init());

    builder
        .invoke_handler(tauri::generate_handler![
            platform_mode,
            ensure_brain_sidecar
        ])
        .run(tauri::generate_context!())
        .expect("error while running JARVIS");
}

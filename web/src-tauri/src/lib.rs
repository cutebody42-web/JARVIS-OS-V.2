mod brain;
mod mobile_identity;
mod mobile_transport;

use brain::{choose_model_file, choose_model_store, ensure_brain_sidecar, platform_mode, BrainState};
use mobile_identity::{
    mobile_accept_pairing_approval,
    mobile_companion_status,
    mobile_fingerprint_status,
    mobile_identity,
    mobile_prepare_pairing,
    mobile_sign_approval_decision,
    mobile_sign_approval_list,
    mobile_sign_brain_request,
    mobile_verify_approval_list_response,
    mobile_verify_approval_receipt,
    mobile_verify_brain_response,
    mobile_verify_owner_presence,
};
use tauri::Manager;
use mobile_transport::mobile_companion_post;

#[tauri::command]
fn shutdown_for_update(app: tauri::AppHandle, state: tauri::State<'_, BrainState>) -> Result<(), String> {
    #[cfg(desktop)]
    {
        state.stop_recovery()?;
        app.exit(0);
        Ok(())
    }
    #[cfg(mobile)]
    {
        let _ = (app, state);
        Err("Install JARVIS desktop updates on your laptop.".into())
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default()
        .manage(BrainState::default())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_websocket::init())
        .plugin(tauri_plugin_deep_link::init())
        .setup(|app| {
            #[cfg(target_os = "android")]
            {
                let store = android_native_keyring_store::Store::new()
                    .map_err(|error| format!("Android JARVIS Keystore initialization failed: {error}"))?;
                keyring_core::set_default_store(store);
            }
            #[cfg(target_os = "ios")]
            {
                let store = apple_native_keyring_store::protected::Store::new()
                    .map_err(|error| format!("iOS JARVIS Keychain initialization failed: {error}"))?;
                keyring_core::set_default_store(store);
            }
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

    #[cfg(any(target_os = "android", target_os = "ios"))]
    let builder = builder.plugin(tauri_plugin_jarvis_fingerprint::init());

    #[cfg(mobile)]
    let builder = builder
        .plugin(tauri_plugin_biometric::init())
        .plugin(tauri_plugin_barcode_scanner::init())
        .plugin(tauri_plugin_haptics::init());

    builder
        .invoke_handler(tauri::generate_handler![
            platform_mode,
            choose_model_store,
            choose_model_file,
            ensure_brain_sidecar,
            shutdown_for_update,
            mobile_identity,
            mobile_companion_status,
            mobile_fingerprint_status,
            mobile_verify_owner_presence,
            mobile_prepare_pairing,
            mobile_accept_pairing_approval,
            mobile_sign_approval_list,
            mobile_verify_approval_list_response,
            mobile_sign_approval_decision,
            mobile_verify_approval_receipt,
            mobile_companion_post,
            mobile_sign_brain_request,
            mobile_verify_brain_response
        ])
        .run(tauri::generate_context!())
        .expect("error while running JARVIS");
}

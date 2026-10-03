fn main() {
    // Native-only commands: WebView code cannot manufacture authentication.
    tauri_plugin::Builder::new(&[])
        .android_path("android")
        .ios_path("ios")
        .build();
}

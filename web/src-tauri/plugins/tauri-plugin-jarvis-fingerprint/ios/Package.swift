// swift-tools-version:5.3
import PackageDescription

let package = Package(
  name: "tauri-plugin-jarvis-fingerprint",
  platforms: [.iOS(.v15)],
  products: [
    .library(name: "tauri-plugin-jarvis-fingerprint", type: .static,
             targets: ["tauri-plugin-jarvis-fingerprint"])
  ],
  dependencies: [.package(name: "Tauri", path: "../.tauri/tauri-api")],
  targets: [
    .target(name: "tauri-plugin-jarvis-fingerprint", dependencies: [.byName(name: "Tauri")],
            path: "Sources")
  ]
)

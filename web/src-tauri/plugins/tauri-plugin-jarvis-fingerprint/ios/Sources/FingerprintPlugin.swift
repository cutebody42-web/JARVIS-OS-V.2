import Foundation
import LocalAuthentication
import Tauri
import UIKit
import WebKit

private struct AuthenticateRequest: Decodable {
  let reason: String
}

private final class FingerprintSession {
  let id = UUID()
  let context: LAContext
  let invoke: Invoke
  var timeout: DispatchWorkItem?

  init(context: LAContext, invoke: Invoke) {
    self.context = context
    self.invoke = invoke
  }
}

final class FingerprintPlugin: Plugin {
  private var active: FingerprintSession?
  private var backgroundObserver: NSObjectProtocol?

  public override func load(webview: WKWebView) {
    backgroundObserver = NotificationCenter.default.addObserver(
      forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main
    ) { [weak self] _ in
      guard let self = self, let session = self.active else { return }
      self.finish(session, error: "JARVIS fingerprint verification was interrupted.")
    }
  }

  deinit {
    if let observer = backgroundObserver {
      NotificationCenter.default.removeObserver(observer)
    }
    active?.timeout?.cancel()
    active?.context.invalidate()
  }

  // Availability is read fresh, rather than cached before enrollment changes.
  private func touchIDError(_ context: LAContext) -> String? {
    var error: NSError?
    let available = context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error)
    guard context.biometryType == .touchID else {
      return "JARVIS owner approval requires Touch ID. Face ID does not provide fingerprint verification."
    }
    guard available else {
      return error?.localizedDescription ?? "Enroll and enable Touch ID in iOS Settings."
    }
    return nil
  }

  @objc func status(_ invoke: Invoke) {
    let error = touchIDError(LAContext())
    var result: JsonObject = ["isAvailable": error == nil, "biometryType": error == nil ? 1 : 0]
    if let error = error { result["error"] = error }
    invoke.resolve(result)
  }

  @objc func authenticate(_ invoke: Invoke) throws {
    let args = try invoke.parseArgs(AuthenticateRequest.self)
    DispatchQueue.main.async { [weak self] in
      guard let self = self else {
        invoke.reject("JARVIS fingerprint service is unavailable.")
        return
      }
      guard self.active == nil else {
        invoke.reject("A JARVIS fingerprint verification is already in progress.")
        return
      }
      let reason = args.reason.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !reason.isEmpty, reason.count <= 500 else {
        invoke.reject("Invalid JARVIS owner authentication reason.")
        return
      }
      let context = LAContext()
      context.localizedFallbackTitle = ""
      context.localizedCancelTitle = "Cancel"
      context.touchIDAuthenticationAllowableReuseDuration = 0
      if let error = self.touchIDError(context) {
        invoke.reject(error)
        return
      }
      let session = FingerprintSession(context: context, invoke: invoke)
      self.active = session
      let timeout = DispatchWorkItem { [weak self, weak session] in
        guard let self = self, let session = session else { return }
        self.finish(session, error: "JARVIS fingerprint verification timed out.")
      }
      session.timeout = timeout
      DispatchQueue.main.asyncAfter(deadline: .now() + 30, execute: timeout)
      // Biometrics-only policy never substitutes the device passcode. The
      // fresh context is gated to Touch ID before a sensor callback can sign.
      context.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, localizedReason: reason) {
        [weak self, weak session] success, error in
        DispatchQueue.main.async {
          guard let self = self, let session = session else { return }
          let failure = success && context.biometryType == .touchID
            ? nil : (error?.localizedDescription ?? "JARVIS fingerprint verification failed.")
          self.finish(session, error: failure)
        }
      }
    }
  }

  private func finish(_ session: FingerprintSession, error: String?) {
    guard active?.id == session.id else { return }
    active = nil
    session.timeout?.cancel()
    session.context.invalidate()
    if let error = error {
      session.invoke.reject(error)
    } else {
      session.invoke.resolve(["userVerified": true, "biometryType": "fingerprint"])
    }
  }
}

@_cdecl("init_plugin_jarvis_fingerprint")
func initPluginJarvisFingerprint() -> Plugin {
  FingerprintPlugin()
}

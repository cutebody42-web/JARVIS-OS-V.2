export type PlatformMode = "desktop" | "mobile";

export type BrainConnection = {
  endpoint: string;
  token: string;
};

export type BrainStatus = {
  identity: "JARVIS";
  mode: string;
  brain_ready: boolean;
  model_store: string | null;
  manual_model: string | null;
  available_models: string[];
  council: {
    enabled: boolean;
    core_model: string;
    parallel_experts: number;
  };
  owner_identity: {
    face_enrolled: boolean;
    face_recognized: boolean;
    face_score: number;
    face_engine: string;
  };
  voice: {
    available: boolean;
    engine: string | null;
    state: "idle" | "listening" | "processing" | "speaking" | "error";
    last_error: string | null;
    privacy: string;
    wake_word: boolean;
  };
  approvals: {
    pending: number;
    pending_exact_actions: number;
    recent: Array<{
      approval_id: string;
      request_id: string;
      state: string;
      message: string;
      action_id: string | null;
    }>;
  };
  setup: {
    phase: string;
    percent: number;
    message: string;
    error: string | null;
  };
  device: {
    device_id: string;
    ram_total_gb: number | null;
    ram_available_gb: number | null;
    system_pressure: number | null;
    power_source: string;
    battery_pct: number | null;
  };
  companion: {
    available: boolean;
    endpoint: string | null;
  };
  paired_devices: Array<{ peer_id: string; endpoint: string }>;
};

export type MobileIdentity = {
  device_id: string;
  public_key: string;
  fingerprint: string;
  key_protection: string;
};

export type MobilePairingBundle = {
  identity: MobileIdentity;
  request: Record<string, unknown>;
  submit_url: string;
  status_url: string;
};

export type MobileCompanionStatus = {
  identity: MobileIdentity;
  paired: boolean;
  desktop_device: string | null;
  key_protection: string;
};

export type MobileBiometricStatus = {
  available: boolean;
  fingerprint: boolean;
  biometryType: number;
  error: string | null;
};

export type MobileBrainRequest = {
  request_id: string;
  endpoint: string;
  body: string;
};

export type MobileApprovalRequest = {
  request_id: string;
  endpoint: string;
  body: string;
};

export type PendingApproval = {
  approval_id: string;
  summary: string;
  action_digest: string;
  expires_at: string;
};

export type BrainReply = {
  identity: "JARVIS";
  text: string;
  lane: string;
};

type TauriCore = {
  invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
};

declare global {
  interface Window {
    __TAURI__?: {
      core?: TauriCore;
      biometric?: {
        authenticate(reason: string, options?: Record<string, unknown>): Promise<void>;
        checkStatus(): Promise<{ isAvailable: boolean; biometryType: number; error?: string }>;
      };
      barcodeScanner?: {
        scan(options?: Record<string, unknown>): Promise<{ content: string; format: string }>;
        requestPermissions(): Promise<string>;
      };
      haptics?: {
        vibrate(options?: Record<string, unknown>): Promise<void>;
      };
      deepLink?: {
        getCurrent(): Promise<string[] | null>;
        onOpenUrl(handler: (urls: string[]) => void): Promise<() => void>;
      };
    };
  }
}

function core(): TauriCore {
  const value = window.__TAURI__?.core;
  if (!value) throw new Error("JARVIS native bridge is unavailable.");
  return value;
}

export function isNativeJarvis() {
  return typeof window !== "undefined" && Boolean(window.__TAURI__?.core);
}

export async function platformMode(): Promise<PlatformMode> {
  return core().invoke<PlatformMode>("platform_mode");
}

export async function mobileCompanionStatus(): Promise<MobileCompanionStatus> {
  return core().invoke<MobileCompanionStatus>("mobile_companion_status");
}

export async function mobileBiometricStatus(): Promise<MobileBiometricStatus> {
  const biometric = window.__TAURI__?.biometric;
  if (!biometric) {
    return {
      available: false,
      fingerprint: false,
      biometryType: 0,
      error: "Mobile biometric authentication is unavailable in this build.",
    };
  }
  try {
    const status = await biometric.checkStatus();
    return {
      available: Boolean(status.isAvailable),
      fingerprint: Boolean(status.isAvailable && status.biometryType === 1),
      biometryType: Number(status.biometryType || 0),
      error: status.error || null,
    };
  } catch (error) {
    return {
      available: false,
      fingerprint: false,
      biometryType: 0,
      error: error instanceof Error ? error.message : "Biometric status check failed.",
    };
  }
}

export async function ensureDesktopBrain(): Promise<BrainConnection> {
  return core().invoke<BrainConnection>("ensure_brain_sidecar");
}

export async function chooseModelStore(): Promise<string | null> {
  return core().invoke<string | null>("choose_model_store");
}

export class LocalBrainClient {
  constructor(readonly connection: BrainConnection) {}

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(this.connection.endpoint + path, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.connection.token}`,
        ...init.headers,
      },
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `JARVIS Brain request failed (${response.status})`);
    }
    return response.json() as Promise<T>;
  }

  async waitUntilReachable(timeoutMs = 30000) {
    const started = Date.now();
    let lastError: unknown = null;
    while (Date.now() - started < timeoutMs) {
      try {
        const response = await fetch(this.connection.endpoint + "/v1/health");
        if (response.ok) return;
      } catch (error) {
        lastError = error;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 250));
    }
    throw lastError instanceof Error
      ? lastError
      : new Error("JARVIS Brain did not become reachable.");
  }

  status() {
    return this.request<BrainStatus>("/v1/status");
  }

  setupLocalBrain(modelStore?: string) {
    return this.request<BrainStatus>("/v1/setup/local-brain", {
      method: "POST",
      body: JSON.stringify({
        approved: true,
        model_store: modelStore?.trim() || null,
      }),
    });
  }

  enrollOwnerFace(cameraIndex = 0) {
    return this.request<{ enrolled: boolean; samples: number; recognized: boolean; score: number }>(
      "/v1/identity/face/enroll",
      { method: "POST", body: JSON.stringify({ camera_index: cameraIndex }) },
    );
  }

  verifyOwnerFace(cameraIndex = 0) {
    return this.request<{
      enrolled: boolean;
      recognized: boolean;
      score: number;
      matched_frames: number;
      total_frames: number;
    }>("/v1/identity/face/verify", {
      method: "POST",
      body: JSON.stringify({ camera_index: cameraIndex }),
    });
  }

  forgetOwnerFace() {
    return this.request<{ enrolled: boolean; recognized: boolean }>(
      "/v1/identity/face/forget",
      { method: "POST", body: "{}" },
    );
  }

  voiceStatus() {
    return this.request<BrainStatus["voice"]>("/v1/voice/status");
  }

  voiceListen(language = "en-US", timeoutSeconds = 8) {
    return this.request<{
      text: string;
      confidence: number | null;
      engine: string;
      state: string;
    }>("/v1/voice/listen", {
      method: "POST",
      body: JSON.stringify({
        language,
        timeout_seconds: timeoutSeconds,
      }),
    });
  }

  voiceSpeak(text: string) {
    return this.request<{ spoken: boolean; state: string }>("/v1/voice/speak", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
  }

  voiceStop() {
    return this.request<{ stopped: boolean; state: string }>("/v1/voice/stop", {
      method: "POST",
      body: "{}",
    });
  }

  selectManualModel(model?: string) {
    return this.request<BrainStatus>("/v1/models/manual", {
      method: "POST",
      body: JSON.stringify({ model: model?.trim() || null }),
    });
  }

  message(message: string, task?: string) {
    return this.request<BrainReply>("/v1/message", {
      method: "POST",
      body: JSON.stringify({ message, task: task || null }),
    });
  }

  createPairingOffer(endpoint?: string, ttlSeconds = 300) {
    return this.request<Record<string, unknown>>("/v1/pair/offer", {
      method: "POST",
      body: JSON.stringify({
        endpoint: endpoint?.trim() || null,
        ttl_seconds: ttlSeconds,
      }),
    });
  }

  pendingPairings() {
    return this.request<{ pending: Array<Record<string, unknown>> }>("/v1/pair/pending");
  }

  approvePairing(pairingId: string) {
    return this.request<Record<string, unknown>>("/v1/pair/approve", {
      method: "POST",
      body: JSON.stringify({ pairing_id: pairingId }),
    });
  }

  cancelPairing(pairingId: string) {
    return this.request<Record<string, unknown>>("/v1/pair/cancel", {
      method: "POST",
      body: JSON.stringify({ pairing_id: pairingId }),
    });
  }
}

export async function bootstrapDesktopBrain() {
  const connection = await ensureDesktopBrain();
  const client = new LocalBrainClient(connection);
  await client.waitUntilReachable();
  return client;
}


export function decodePairingDeepLink(url: string): Record<string, unknown> | null {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "jarvis:" || parsed.hostname !== "pair") return null;
    const encoded = parsed.searchParams.get("offer");
    if (!encoded) return null;
    const normalized = encoded.replace(/-/g, "+").replace(/_/g, "/");
    const padded = normalized + "=".repeat((4 - (normalized.length % 4 || 4)) % 4);
    const json = decodeURIComponent(escape(atob(padded)));
    const value = JSON.parse(json);
    return value && typeof value === "object" ? value as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

export async function currentPairingDeepLink() {
  const deepLink = window.__TAURI__?.deepLink;
  if (!deepLink) return null;
  const urls = await deepLink.getCurrent();
  for (const url of urls || []) {
    const offer = decodePairingDeepLink(url);
    if (offer) return offer;
  }
  return null;
}

export async function listenForPairingDeepLinks(
  handler: (offer: Record<string, unknown>) => void,
) {
  const deepLink = window.__TAURI__?.deepLink;
  if (!deepLink) return () => undefined;
  return deepLink.onOpenUrl((urls) => {
    for (const url of urls) {
      const offer = decodePairingDeepLink(url);
      if (offer) {
        handler(offer);
        break;
      }
    }
  });
}


const NEXUS_MEDIA_TYPE = "application/vnd.nexus-sync+json";

async function requireMobileOwnerPresence(
  reason: string,
  { requireFingerprint = false }: { requireFingerprint?: boolean } = {},
) {
  const biometric = window.__TAURI__?.biometric;
  if (!biometric) throw new Error("Mobile owner authentication is unavailable.");
  const status = await biometric.checkStatus();
  if (!status.isAvailable) {
    throw new Error(status.error || "Fingerprint/biometric approval is unavailable on this phone.");
  }
  // Tauri BiometryType.TouchID (1) maps to Android fingerprint. Sensitive
  // approvals can explicitly require it instead of silently falling back to
  // a lock-screen PIN/password.
  if (requireFingerprint && status.biometryType !== 1) {
    throw new Error("Enroll and enable a fingerprint on this phone to approve sensitive JARVIS actions.");
  }
  await biometric.authenticate(reason, {
    allowDeviceCredential: false,
    confirmationRequired: true,
    cancelTitle: "Cancel",
    title: "JARVIS owner approval",
    subtitle: requireFingerprint ? "Use your fingerprint to continue" : "Verify owner presence",
  });
  return status;
}

export function encodePairingDeepLink(offer: Record<string, unknown>) {
  const json = JSON.stringify(offer);
  const bytes = new TextEncoder().encode(json);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  const encoded = btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
  return `jarvis://pair?offer=${encoded}`;
}

export async function pairMobileCompanion(
  offer: Record<string, unknown>,
  onState?: (state: string) => void,
) {
  await requireMobileOwnerPresence(
    "Confirm pairing this phone with your JARVIS Brain",
    { requireFingerprint: true },
  );
  const bundle = await core().invoke<MobilePairingBundle>("mobile_prepare_pairing", { offer });

  onState?.("requesting");
  const submitted = await fetch(bundle.submit_url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bundle.request),
  });
  if (!submitted.ok && submitted.status !== 409) {
    const body = await submitted.json().catch(() => ({}));
    throw new Error(body.detail || `JARVIS pairing request failed (${submitted.status})`);
  }

  const expiresAt = Date.parse(String(offer.expires_at || ""));
  const deadline = Number.isFinite(expiresAt)
    ? expiresAt
    : Date.now() + 10 * 60_000;

  onState?.("awaiting-owner");
  while (Date.now() < deadline) {
    const response = await fetch(bundle.status_url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bundle.request),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Pairing status failed (${response.status})`);
    }

    const contentType = response.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
    if (contentType === NEXUS_MEDIA_TYPE) {
      const signedApproval = await response.text();
      const identity = await core().invoke<MobileIdentity>(
        "mobile_accept_pairing_approval",
        { offer, signedApproval },
      );
      onState?.("paired");
      return identity;
    }

    const state = await response.json() as { state?: string };
    if (state.state === "cancelled" || state.state === "expired") {
      throw new Error(`JARVIS pairing ${state.state}.`);
    }
    await new Promise((resolve) => window.setTimeout(resolve, 1500));
  }
  throw new Error("JARVIS pairing invitation expired before approval.");
}

export async function mobileBrainMessage(
  message: string,
  task?: "general" | "realtime" | "coding",
): Promise<BrainReply> {
  const request = await core().invoke<MobileBrainRequest>(
    "mobile_sign_brain_request",
    { message, task: task || null },
  );
  const response = await fetch(request.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": NEXUS_MEDIA_TYPE,
      Accept: NEXUS_MEDIA_TYPE,
    },
    body: request.body,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `JARVIS Brain request failed (${response.status})`);
  }
  const signedResponse = await response.text();
  const payload = await core().invoke<{
    version: number;
    request_id: string;
    identity: "JARVIS";
    text: string;
    lane: string;
  }>("mobile_verify_brain_response", {
    requestId: request.request_id,
    signedResponse,
  });
  return {
    identity: "JARVIS",
    text: payload.text,
    lane: payload.lane,
  };
}

export async function mobilePendingApprovals(): Promise<PendingApproval[]> {
  const request = await core().invoke<MobileApprovalRequest>("mobile_sign_approval_list");
  const response = await fetch(request.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": NEXUS_MEDIA_TYPE,
      Accept: NEXUS_MEDIA_TYPE,
    },
    body: request.body,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `JARVIS approval check failed (${response.status})`);
  }
  const signedResponse = await response.text();
  const payload = await core().invoke<{
    version: number;
    request_id: string;
    pending: PendingApproval[];
  }>("mobile_verify_approval_list_response", {
    requestId: request.request_id,
    signedResponse,
  });
  return Array.isArray(payload.pending) ? payload.pending : [];
}

export async function mobileDecideApproval(
  approvalId: string,
  approved: boolean,
): Promise<Record<string, unknown>> {
  const biometric = await requireMobileOwnerPresence(
    approved
      ? "Confirm this JARVIS action with your fingerprint"
      : "Confirm rejecting this JARVIS action with your fingerprint",
    { requireFingerprint: true },
  );
  if (biometric.biometryType !== 1) {
    throw new Error("JARVIS sensitive approvals require fingerprint verification.");
  }
  const request = await core().invoke<MobileApprovalRequest>(
    "mobile_sign_approval_decision",
    { approvalId, approved, biometryType: "fingerprint" },
  );
  const response = await fetch(request.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": NEXUS_MEDIA_TYPE,
      Accept: NEXUS_MEDIA_TYPE,
    },
    body: request.body,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `JARVIS approval decision failed (${response.status})`);
  }
  const signedResponse = await response.text();
  return core().invoke<Record<string, unknown>>(
    "mobile_verify_approval_receipt",
    {
      requestId: request.request_id,
      signedResponse,
    },
  );
}


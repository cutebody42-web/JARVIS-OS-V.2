export type PlatformMode = "desktop" | "mobile";

export type BrainConnection = {
  endpoint: string;
  token: string;
};

export type UpdatePlan = {
  version: string;
  commit_sha: string;
  artifact_sha256: string;
  changed_paths: string[];
  notes: string;
  update_class: "patch" | "minor" | "major";
  release_tag: string;
  artifact_name: string;
};

export type UpdateStatus = {
  supported: boolean;
  busy?: boolean;
  phase: string;
  message: string;
  plan: UpdatePlan | null;
  checkpoint_id: string | null;
  approval_id: string | null;
  approval_state: "pending" | "approved" | "rejected" | "expired" | "consumed" | null;
  history: Array<{
    state: string;
    message: string;
    checkpoint_id: string;
    version: string;
    previous_version?: string;
  }>;
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
  try {
    return await core().invoke<MobileBiometricStatus>("mobile_fingerprint_status");
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

export async function chooseModelFile(): Promise<string | null> {
  return core().invoke<string | null>("choose_model_file");
}

export async function shutdownForUpdate(): Promise<void> {
  return core().invoke<void>("shutdown_for_update");
}

class BrainHttpError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export class LocalBrainClient {
  private reconnecting: Promise<void> | null = null;
  private recoverySuspended = false;

  constructor(public connection: BrainConnection) {}

  private safeToRetry(init: RequestInit) {
    return ["GET", "HEAD"].includes((init.method || "GET").toUpperCase());
  }

  private async reconnect(observed: BrainConnection) {
    if (!this.reconnecting && (this.connection.token !== observed.token || this.connection.endpoint !== observed.endpoint)) return;
    if (this.recoverySuspended) throw new Error("Brain recovery is paused while the desktop update shuts down.");
    if (!this.reconnecting) {
      this.reconnecting = (async () => {
        this.connection = await ensureDesktopBrain();
        await this.waitUntilReachable(10000);
      })();
    }
    const pending = this.reconnecting;
    try {
      await pending;
    } finally {
      if (this.reconnecting === pending) this.reconnecting = null;
    }
  }

  private fetchOnce(path: string, init: RequestInit = {}) {
    const signal = init.signal || (this.safeToRetry(init)
      && typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function"
      ? AbortSignal.timeout(5000) : undefined);
    return fetch(this.connection.endpoint + path, {
      ...init,
      signal,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.connection.token}`,
        ...init.headers,
      },
    });
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const observed = this.connection;
    const canRecover = this.safeToRetry(init) && isNativeJarvis() && !this.recoverySuspended;
    let response: Response;
    let retried = false;
    try {
      response = await this.fetchOnce(path, init);
    } catch (error) {
      if (!canRecover || init.signal?.aborted) throw error;
      await this.reconnect(observed);
      response = await this.fetchOnce(path, init);
      retried = true;
    }
    // A concurrent request may still hold the token of an exited child. Only
    // read requests get one fresh-connection retry; effectful POSTs never do.
    if (response.status === 401 && canRecover && !retried && !init.signal?.aborted) {
      await this.reconnect(observed);
      response = await this.fetchOnce(path, init);
    }
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new BrainHttpError(body.detail || `JARVIS Brain request failed (${response.status})`, response.status);
    }
    return response.json() as Promise<T>;
  }

  async waitUntilReachable(timeoutMs = 30000) {
    const started = Date.now();
    let lastError: unknown = null;
    let nextEnsure = started;
    while (Date.now() - started < timeoutMs) {
      try {
        const remaining = timeoutMs - (Date.now() - started);
        const signal = typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function"
          ? AbortSignal.timeout(Math.max(1, Math.min(1000, remaining))) : undefined;
        const response = await fetch(this.connection.endpoint + "/v1/health", { signal });
        if (response.ok) {
          const health = await response.json();
          if (health.ok === true && health.identity === "JARVIS") return;
        }
      } catch (error) {
        lastError = error;
      }
      if (isNativeJarvis() && !this.recoverySuspended && Date.now() >= nextEnsure) {
        // Native lifecycle events decide whether the child has exited. An alive
        // but slow sidecar retains its token/process and is never killed here.
        this.connection = await ensureDesktopBrain();
        nextEnsure = Date.now() + 1000;
      }
      const remaining = timeoutMs - (Date.now() - started);
      if (remaining > 0) await new Promise((resolve) => window.setTimeout(resolve, Math.min(250, remaining)));
    }
    throw lastError instanceof Error
      ? lastError
      : new Error("JARVIS Brain did not become reachable.");
  }

  status() {
    return this.request<BrainStatus>("/v1/status");
  }

  async getUpdateStatus() {
    const status = await this.request<UpdateStatus>("/v1/update/status");
    if (status.phase === "applying") this.recoverySuspended = true;
    return status;
  }

  checkUpdates() {
    return this.request<UpdateStatus>("/v1/update/check", { method: "POST", body: "{}" });
  }

  prepareUpdate() {
    return this.request<UpdateStatus>("/v1/update/prepare", { method: "POST", body: "{}" });
  }

  async applyUpdate(checkpointId: string, approvalId: string) {
    this.recoverySuspended = true;
    try {
      const result = await this.request<UpdateStatus>("/v1/update/apply", {
        method: "POST",
        body: JSON.stringify({ checkpoint_id: checkpointId, approval_id: approvalId }),
      });
      if (result.phase !== "applying") this.recoverySuspended = false;
      return result;
    } catch (error) {
      // A lost response may follow an accepted handoff; reconnecting could start
      // another sidecar while the installer waits for the original to exit.
      if (error instanceof BrainHttpError) this.recoverySuspended = false;
      throw error;
    }
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

  importLocalModel(path: string) {
    return this.request<{ model: string; status: BrainStatus }>("/v1/models/import", {
      method: "POST",
      body: JSON.stringify({ path }),
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
  await client.waitUntilReachable(60000);
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

async function mobileCompanionPost(
  endpoint: string,
  body: string,
  offer?: Record<string, unknown>,
): Promise<Response> {
  // Rust checks the saved desktop/invitation origin and fixed companion route.
  // Native HTTP reaches tailnet addresses without weakening WebView CSP or
  // relying on browser CORS, mixed-content or Android cleartext exceptions.
  const reply = await core().invoke<{
    status: number;
    content_type: string;
    body: string;
  }>("mobile_companion_post", { endpoint, body, offer: offer || null });
  return new Response([204, 205, 304].includes(reply.status) ? null : reply.body, {
    status: reply.status,
    headers: { "Content-Type": reply.content_type },
  });
}

async function requireMobileOwnerPresence(
  reason: string,
) {
  const status = await mobileBiometricStatus();
  if (!status.fingerprint) {
    throw new Error(status.error || "Enroll and enable a fingerprint on this phone to approve JARVIS actions.");
  }
  // Android uses a dedicated fingerprint sensor API: generic biometric prompts
  // may accept face/iris when the device supports multiple modalities.
  await core().invoke<void>("mobile_verify_owner_presence", { reason });
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
  );
  const bundle = await core().invoke<MobilePairingBundle>("mobile_prepare_pairing", { offer });

  onState?.("requesting");
  const submitted = await mobileCompanionPost(
    bundle.submit_url, JSON.stringify(bundle.request), offer,
  );
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
    const response = await mobileCompanionPost(
      bundle.status_url, JSON.stringify(bundle.request), offer,
    );

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
  const response = await mobileCompanionPost(request.endpoint, request.body);
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
  const response = await mobileCompanionPost(request.endpoint, request.body);
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
  // The Rust command performs the OS biometric prompt itself immediately
  // before signing, so WebView code cannot assert user verification.
  const request = await core().invoke<MobileApprovalRequest>(
    "mobile_sign_approval_decision",
    { approvalId, approved },
  );
  const response = await mobileCompanionPost(request.endpoint, request.body);
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

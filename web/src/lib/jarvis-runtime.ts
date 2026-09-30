export type PlatformMode = "desktop" | "mobile";

export type BrainConnection = {
  endpoint: string;
  token: string;
};

export type BrainStatus = {
  identity: "JARVIS";
  mode: string;
  brain_ready: boolean;
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
  paired_devices: Array<{ peer_id: string; endpoint: string }>;
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

export async function ensureDesktopBrain(): Promise<BrainConnection> {
  return core().invoke<BrainConnection>("ensure_brain_sidecar");
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

  setupLocalBrain() {
    return this.request<BrainStatus>("/v1/setup/local-brain", {
      method: "POST",
      body: JSON.stringify({ approved: true }),
    });
  }

  message(message: string, task?: string) {
    return this.request<BrainReply>("/v1/message", {
      method: "POST",
      body: JSON.stringify({ message, task: task || null }),
    });
  }

  createPairingOffer(endpoint: string, ttlSeconds = 300) {
    return this.request<Record<string, unknown>>("/v1/pair/offer", {
      method: "POST",
      body: JSON.stringify({ endpoint, ttl_seconds: ttlSeconds }),
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

import { ensureDesktopBrain, LocalBrainClient, type BrainConnection } from "@/lib/jarvis-runtime";

export type WakeWordStatus = {
  configured: boolean;
  enabled: boolean;
  model_dir: string | null;
  threshold: number;
  sample_rate: number;
  chunk_samples: number;
  last_score: number | null;
  last_activation_at: string | null;
  last_error: string | null;
  local_only: boolean;
  microphone_owner: boolean;
  authority: "activation_only";
};

export type WakeWordFrameResult = {
  detected: boolean;
  activated: boolean;
  score: number;
  threshold: number;
  model_name: string;
  authority: "activation_only";
};

async function request<T>(connection: BrainConnection, path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(connection.endpoint + path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${connection.token}`,
      ...init.headers,
    },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Wake-word request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export async function wakeWordConnection() {
  return ensureDesktopBrain();
}

export function wakeWordStatus(connection: BrainConnection) {
  return request<WakeWordStatus>(connection, "/v1/wake-word/status");
}

export function configureWakeWord(connection: BrainConnection, modelDir: string, threshold: number) {
  return request<WakeWordStatus>(connection, "/v1/wake-word/configure", {
    method: "POST",
    body: JSON.stringify({ model_dir: modelDir, threshold }),
  });
}

export function setWakeWordEnabled(connection: BrainConnection, enabled: boolean) {
  return request<WakeWordStatus>(connection, "/v1/wake-word/enabled", {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
}

export function submitWakeWordFrame(connection: BrainConnection, pcm16Base64: string) {
  return request<WakeWordFrameResult>(connection, "/v1/wake-word/frame", {
    method: "POST",
    body: JSON.stringify({ pcm16_base64: pcm16Base64 }),
  });
}

export function voiceClient(connection: BrainConnection) {
  return new LocalBrainClient(connection);
}

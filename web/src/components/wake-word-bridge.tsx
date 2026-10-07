"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { isNativeJarvis, platformMode, type BrainConnection } from "@/lib/jarvis-runtime";
import {
  configureWakeWord,
  setWakeWordEnabled,
  submitWakeWordFrame,
  voiceClient,
  wakeWordConnection,
  wakeWordStatus,
  type WakeWordStatus,
} from "@/lib/wake-word-runtime";

const TARGET_RATE = 16000;
const FALLBACK_CHUNK = 1280;
const MAX_QUEUED_CHUNKS = 8;

function resample(input: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (sourceRate === targetRate) return input.slice();
  if (!Number.isFinite(sourceRate) || sourceRate <= 0 || targetRate <= 0) return new Float32Array();
  const length = Math.max(1, Math.floor(input.length * targetRate / sourceRate));
  const output = new Float32Array(length);
  if (length === 1 || input.length === 1) {
    output[0] = input[0] || 0;
    return output;
  }
  const scale = (input.length - 1) / (length - 1);
  for (let index = 0; index < length; index += 1) {
    const position = index * scale;
    const left = Math.floor(position);
    const right = Math.min(input.length - 1, left + 1);
    const fraction = position - left;
    output[index] = input[left] * (1 - fraction) + input[right] * fraction;
  }
  return output;
}

function toPcm16(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, input[index]));
    output[index] = sample < 0 ? Math.round(sample * 32768) : Math.round(sample * 32767);
  }
  return output;
}

function pcmBase64(input: Int16Array): string {
  const bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    const slice = bytes.subarray(offset, Math.min(bytes.length, offset + 0x8000));
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

export function WakeWordBridge() {
  const [supported, setSupported] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [status, setStatus] = useState<WakeWordStatus | null>(null);
  const [modelDir, setModelDir] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [phase, setPhase] = useState("Wake word off");
  const [error, setError] = useState<string | null>(null);
  const [lastTranscript, setLastTranscript] = useState("");
  const [lastReply, setLastReply] = useState("");

  const connectionRef = useRef<BrainConnection | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const pendingRef = useRef<number[]>([]);
  const queueRef = useRef<Int16Array[]>([]);
  const drainingRef = useRef(false);
  const enabledRef = useRef(false);
  const startingRef = useRef(false);
  const chunkSamplesRef = useRef(FALLBACK_CHUNK);

  const stopCapture = useCallback(async () => {
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    processorRef.current = null;
    sourceRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    pendingRef.current = [];
    queueRef.current = [];
    const context = contextRef.current;
    contextRef.current = null;
    if (context && context.state !== "closed") {
      await context.close().catch(() => undefined);
    }
  }, []);

  const runActivatedConversation = useCallback(async (connection: BrainConnection) => {
    await stopCapture();
    setPhase("JARVIS heard the wake word — listening…");
    try {
      const client = voiceClient(connection);
      const heard = await client.voiceListen("en-US", 8);
      const transcript = heard.text.trim();
      setLastTranscript(transcript);
      if (!transcript) {
        setPhase("No speech heard");
        return;
      }
      setPhase("Thinking locally…");
      const reply = await client.message(transcript);
      setLastReply(reply.text);
      setPhase("Speaking…");
      if (reply.text.trim()) await client.voiceSpeak(reply.text);
      setPhase("Wake word ready");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Wake conversation failed.");
      setPhase("Wake conversation error");
    }
  }, [stopCapture]);

  const drainQueue = useCallback(async () => {
    if (drainingRef.current) return;
    const connection = connectionRef.current;
    if (!connection || !enabledRef.current) return;
    drainingRef.current = true;
    try {
      while (enabledRef.current && queueRef.current.length > 0) {
        const frame = queueRef.current.shift();
        if (!frame) break;
        const result = await submitWakeWordFrame(connection, pcmBase64(frame));
        setStatus((current) => current ? { ...current, last_score: result.score } : current);
        if (result.activated) {
          queueRef.current = [];
          await runActivatedConversation(connection);
          break;
        }
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Wake-word frame failed.");
      setPhase("Wake word paused after error");
      enabledRef.current = false;
      await stopCapture();
      if (connectionRef.current) {
        await setWakeWordEnabled(connectionRef.current, false).catch(() => undefined);
      }
    } finally {
      drainingRef.current = false;
    }
  }, [runActivatedConversation, stopCapture]);

  const enqueuePcm = useCallback((pcm: Int16Array) => {
    const chunkSize = chunkSamplesRef.current || FALLBACK_CHUNK;
    for (let index = 0; index < pcm.length; index += 1) pendingRef.current.push(pcm[index]);
    while (pendingRef.current.length >= chunkSize) {
      const chunk = new Int16Array(pendingRef.current.splice(0, chunkSize));
      if (queueRef.current.length >= MAX_QUEUED_CHUNKS) queueRef.current.shift();
      queueRef.current.push(chunk);
    }
    void drainQueue();
  }, [drainQueue]);

  const startCapture = useCallback(async () => {
    if (startingRef.current || streamRef.current || !enabledRef.current) return;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone capture is unavailable in this WebView.");
    startingRef.current = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      if (!enabledRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      const AudioContextType = window.AudioContext;
      const context = new AudioContextType();
      const source = context.createMediaStreamSource(stream);
      // ScriptProcessor is intentionally used as a compatibility bridge for
      // Tauri WebView2. Frames are bounded and never leave loopback.
      const processor = context.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        if (!enabledRef.current) return;
        const mono = event.inputBuffer.getChannelData(0);
        enqueuePcm(toPcm16(resample(mono, context.sampleRate, TARGET_RATE)));
      };
      source.connect(processor);
      processor.connect(context.destination);
      streamRef.current = stream;
      contextRef.current = context;
      sourceRef.current = source;
      processorRef.current = processor;
      setPhase("Wake word listening locally");
    } finally {
      startingRef.current = false;
    }
  }, [enqueuePcm]);

  useEffect(() => {
    let cancelled = false;
    if (!isNativeJarvis()) return;
    void (async () => {
      try {
        if (await platformMode() !== "desktop") return;
        const connection = await wakeWordConnection();
        const current = await wakeWordStatus(connection);
        if (cancelled) return;
        connectionRef.current = connection;
        chunkSamplesRef.current = current.chunk_samples || FALLBACK_CHUNK;
        setStatus(current);
        setModelDir(current.model_dir || "");
        setThreshold(current.threshold);
        setSupported(true);
        enabledRef.current = false;
        setPhase(current.configured ? "Wake word ready — click Enable" : "Configure local wake model");
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause.message : "Wake-word status unavailable.");
      }
    })();
    return () => {
      cancelled = true;
      enabledRef.current = false;
      void stopCapture();
      const connection = connectionRef.current;
      if (connection) void setWakeWordEnabled(connection, false).catch(() => undefined);
    };
  }, [stopCapture]);

  const configure = useCallback(async () => {
    const connection = connectionRef.current;
    if (!connection) return;
    setError(null);
    try {
      const next = await configureWakeWord(connection, modelDir.trim(), threshold);
      chunkSamplesRef.current = next.chunk_samples || FALLBACK_CHUNK;
      enabledRef.current = false;
      setStatus(next);
      setPhase("Configured — click Enable to open the microphone");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Wake-word configuration failed.");
    }
  }, [modelDir, threshold]);

  const enable = useCallback(async () => {
    const connection = connectionRef.current;
    if (!connection) return;
    setError(null);
    try {
      const next = await setWakeWordEnabled(connection, true);
      enabledRef.current = true;
      setStatus(next);
      try {
        await startCapture();
      } catch (cause) {
        enabledRef.current = false;
        await setWakeWordEnabled(connection, false).catch(() => undefined);
        throw cause;
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Wake-word enable failed.");
      setPhase("Wake word off");
    }
  }, [startCapture]);

  const disable = useCallback(async () => {
    const connection = connectionRef.current;
    enabledRef.current = false;
    await stopCapture();
    if (!connection) return;
    try {
      const next = await setWakeWordEnabled(connection, false);
      setStatus(next);
      setPhase("Wake word off");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Wake-word disable failed.");
    }
  }, [stopCapture]);

  // Resume wake capture after an activation conversation finishes, but only if
  // the owner has not disabled the feature meanwhile.
  useEffect(() => {
    if (enabledRef.current && !streamRef.current && phase === "Wake word ready") {
      void startCapture().catch((cause) => {
        setError(cause instanceof Error ? cause.message : "Wake-word microphone restart failed.");
      });
    }
  }, [phase, startCapture]);

  if (!supported) return null;

  const enabled = Boolean(status?.enabled && enabledRef.current);
  return (
    <div className="fixed bottom-4 right-4 z-[80] w-[min(92vw,390px)] rounded-2xl border border-cyan-400/25 bg-slate-950/90 p-3 text-xs text-slate-200 shadow-2xl backdrop-blur-xl">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-3 text-left"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
      >
        <span className="font-semibold tracking-wide">Local Wake Word</span>
        <span className={enabled ? "text-emerald-300" : "text-slate-400"}>{enabled ? "ON" : "OFF"}</span>
      </button>
      <div className="mt-1 truncate text-[11px] text-slate-400">{phase}</div>

      {expanded && (
        <div className="mt-3 space-y-3 border-t border-white/10 pt-3">
          <label className="block">
            <span className="mb-1 block text-slate-400">Owner-local model folder</span>
            <input
              value={modelDir}
              onChange={(event) => setModelDir(event.target.value)}
              disabled={enabled}
              placeholder="C:\\JARVIS\\models\\wakeword"
              className="w-full rounded-lg border border-white/10 bg-black/30 px-2 py-1.5 outline-none focus:border-cyan-400/50 disabled:opacity-50"
            />
          </label>
          <label className="block">
            <span className="mb-1 flex justify-between text-slate-400"><span>Threshold</span><span>{threshold.toFixed(2)}</span></span>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={threshold}
              disabled={enabled}
              onChange={(event) => setThreshold(Number(event.target.value))}
              className="w-full"
            />
          </label>
          <div className="flex gap-2">
            <button type="button" onClick={() => void configure()} disabled={enabled || !modelDir.trim()} className="rounded-lg border border-cyan-400/30 px-2 py-1.5 disabled:opacity-40">Configure</button>
            {enabled ? (
              <button type="button" onClick={() => void disable()} className="rounded-lg border border-rose-400/30 px-2 py-1.5 text-rose-200">Disable</button>
            ) : (
              <button type="button" onClick={() => void enable()} disabled={!status?.configured} className="rounded-lg border border-emerald-400/30 px-2 py-1.5 text-emerald-200 disabled:opacity-40">Enable microphone</button>
            )}
          </div>
          <div className="space-y-1 text-[11px] text-slate-400">
            <div>Audio: local loopback only · {TARGET_RATE / 1000} kHz PCM · activation has zero action authority.</div>
            {status?.last_score != null && <div>Last score: {status.last_score.toFixed(3)}</div>}
            {lastTranscript && <div className="truncate">Heard: {lastTranscript}</div>}
            {lastReply && <div className="line-clamp-2">JARVIS: {lastReply}</div>}
            {error && <div className="text-rose-300">{error}</div>}
          </div>
        </div>
      )}
    </div>
  );
}

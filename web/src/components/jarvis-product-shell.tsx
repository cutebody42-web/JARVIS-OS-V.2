"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, ArrowUp, Cpu, Link2, LoaderCircle, Mic2, RadioTower, ShieldCheck, Smartphone, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Reactor } from "@/components/reactor";
import {
  BrainStatus,
  LocalBrainClient,
  bootstrapDesktopBrain,
  chooseModelStore,
  currentPairingDeepLink,
  listenForPairingDeepLinks,
  mobileBrainMessage,
  mobileCompanionStatus,
  mobileDecideApproval,
  mobilePendingApprovals,
  pairMobileCompanion,
  platformMode,
  type MobileIdentity,
  type PendingApproval,
  type PlatformMode,
} from "@/lib/jarvis-runtime";

type LocalMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  at: string;
};

type VisualProfile = {
  tier: "eco" | "balanced" | "high" | "ultra";
  refreshHz: number;
};

function useVisualProfile(systemPressure: number | null | undefined): VisualProfile {
  const [profile, setProfile] = useState<VisualProfile>({ tier: "balanced", refreshHz: 60 });

  useEffect(() => {
    let cancelled = false;
    let frame = 0;
    const stamps: number[] = [];

    function sample(now: number) {
      if (cancelled) return;
      stamps.push(now);
      frame += 1;
      if (frame < 45) {
        requestAnimationFrame(sample);
        return;
      }
      const duration = stamps[stamps.length - 1] - stamps[0];
      const measured = duration > 0 ? ((stamps.length - 1) * 1000) / duration : 60;
      const common = [30, 60, 75, 90, 120, 144, 165, 180, 240];
      const refreshHz = common.reduce((best, value) =>
        Math.abs(value - measured) < Math.abs(best - measured) ? value : best
      , 60);
      const cores = navigator.hardwareConcurrency || 4;
      const memory = Number((navigator as Navigator & { deviceMemory?: number }).deviceMemory || 4);
      const pressure = systemPressure ?? 0.5;
      let tier: VisualProfile["tier"] = "balanced";
      if (pressure > 0.88 || cores <= 4 || memory <= 4) tier = "eco";
      else if (pressure < 0.45 && cores >= 12 && memory >= 12 && refreshHz >= 120) tier = "ultra";
      else if (pressure < 0.7 && cores >= 8 && memory >= 8) tier = "high";
      if (!cancelled) setProfile({ tier, refreshHz });
    }

    requestAnimationFrame(sample);
    return () => { cancelled = true; };
  }, [systemPressure]);

  return profile;
}

function MobileShell() {
  const [error, setError] = useState("");
  const [scanned, setScanned] = useState<Record<string, unknown> | null>(null);
  const [pairState, setPairState] = useState("unpaired");
  const [identity, setIdentity] = useState<MobileIdentity | null>(null);
  const [pendingApprovals, setPendingApprovals] = useState<PendingApproval[]>([]);
  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const mounted = useRef(true);

  useEffect(() => () => { mounted.current = false; }, []);

  useEffect(() => {
    let cancelled = false;
    mobileCompanionStatus()
      .then((status) => {
        if (cancelled || !status.paired) return;
        setIdentity(status.identity);
        setPairState("paired");
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, []);

  const connectOffer = useCallback(async (offer: Record<string, unknown>) => {
    setScanned(offer);
    setError("");
    setBusy(true);
    try {
      const nextIdentity = await pairMobileCompanion(offer, (state) => {
        if (mounted.current) setPairState(state);
      });
      if (!mounted.current) return;
      setIdentity(nextIdentity);
      setPairState("paired");
      await window.__TAURI__?.haptics?.vibrate({ duration: 80 }).catch(() => undefined);
    } catch (reason) {
      if (!mounted.current) return;
      setPairState("unpaired");
      setError(reason instanceof Error ? reason.message : "JARVIS pairing failed.");
    } finally {
      if (mounted.current) setBusy(false);
    }
  }, []);

  useEffect(() => {
    let unlisten: (() => void) | null = null;
    let cancelled = false;

    currentPairingDeepLink()
      .then((offer) => {
        if (!cancelled && offer) void connectOffer(offer);
      })
      .catch(() => undefined);

    listenForPairingDeepLinks((offer) => {
      if (!cancelled) void connectOffer(offer);
    })
      .then((stop) => { unlisten = stop; })
      .catch(() => undefined);

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [connectOffer]);

  async function scanPairingCode() {
    setError("");
    try {
      const scanner = window.__TAURI__?.barcodeScanner;
      if (!scanner) throw new Error("QR scanner is unavailable in this mobile build.");
      const permission = await scanner.requestPermissions();
      if (!String(permission).startsWith("granted")) {
        throw new Error("Camera permission is required to pair JARVIS.");
      }
      const result = await scanner.scan({ cameraDirection: "back", windowed: false });
      let offer: Record<string, unknown> | null = null;
      try {
        const parsed = JSON.parse(result.content);
        if (parsed && typeof parsed === "object") offer = parsed as Record<string, unknown>;
      } catch {
        const url = result.content;
        const parsed = new URL(url);
        const encoded = parsed.searchParams.get("offer");
        if (parsed.protocol === "jarvis:" && parsed.hostname === "pair" && encoded) {
          const normalized = encoded.replace(/-/g, "+").replace(/_/g, "/");
          const padded = normalized + "=".repeat((4 - (normalized.length % 4 || 4)) % 4);
          const bytes = Uint8Array.from(atob(padded), (char) => char.charCodeAt(0));
          offer = JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>;
        }
      }
      if (!offer) throw new Error("Invalid JARVIS pairing code.");
      await connectOffer(offer);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Pairing scan failed.");
    }
  }

  useEffect(() => {
    if (!identity) return;
    let cancelled = false;
    async function refreshApprovals() {
      try {
        const pending = await mobilePendingApprovals();
        if (!cancelled) setPendingApprovals(pending);
      } catch {
        // The secure desktop link may be temporarily unavailable.
      }
    }
    void refreshApprovals();
    const timer = window.setInterval(refreshApprovals, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [identity]);

  async function decideApproval(approvalId: string, approved: boolean) {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      await mobileDecideApproval(approvalId, approved);
      setPendingApprovals(await mobilePendingApprovals());
      await window.__TAURI__?.haptics?.vibrate({ duration: approved ? 120 : 60 }).catch(() => undefined);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "JARVIS biometric approval failed.");
    } finally {
      setBusy(false);
    }
  }

  async function submitMobile(event: FormEvent) {
    event.preventDefault();
    const content = draft.trim();
    if (!identity || !content || busy) return;
    setDraft("");
    setError("");
    setMessages((items) => [...items, {
      id: crypto.randomUUID(),
      role: "user",
      content,
      at: new Date().toISOString(),
    }]);
    setBusy(true);
    try {
      const reply = await mobileBrainMessage(content);
      setMessages((items) => [...items, {
        id: crypto.randomUUID(),
        role: "assistant",
        content: reply.text,
        at: new Date().toISOString(),
      }]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "JARVIS could not reach the desktop Brain.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="product-shell mobile-product-shell">
      <section className="mobile-pair-stage">
        <div className="wordmark"><span className="wordmark-mark">J</span> JARVIS</div>
        <Reactor state={busy ? "THINKING" : identity ? "LISTENING" : "MUTED"} />
        <p className="section-index">{identity ? "COMPANION / SECURE LINK" : "COMPANION / SECURE PAIRING"}</p>
        <h1>{identity ? "At your service." : scanned ? "Confirm this device." : "Connect to your JARVIS Brain."}</h1>

        {!identity ? (
          <>
            <p>
              {pairState === "awaiting-owner"
                ? "Pairing proof accepted. Approve this phone on the desktop JARVIS app."
                : "Scan the desktop QR. Your phone becomes another authenticated face of the same JARVIS — with the same memory and Brain."}
            </p>
            <Button onClick={scanPairingCode} disabled={busy}>
              {busy ? <LoaderCircle className="spin" size={16} /> : <Smartphone size={16} />}
              {busy ? "Securing device link" : "Scan desktop QR"}
            </Button>
            {scanned && (
              <pre className="pair-preview">{JSON.stringify({
                desktop: scanned.inviter_device,
                expires_at: scanned.expires_at,
                state: pairState,
              }, null, 2)}</pre>
            )}
          </>
        ) : (
          <>
            <p>Authenticated as {identity.device_id}. Messages and approvals are signed locally with the companion key protected by Android Keystore.</p>
            {pendingApprovals.length > 0 && (
              <section className="pair-pending-list">
                <span className="section-index">BIOMETRIC APPROVAL REQUIRED</span>
                {pendingApprovals.map((approval) => (
                  <div className="pair-pending-item" key={approval.approval_id}>
                    <div>
                      <strong>{approval.summary}</strong>
                      <small>Expires {new Date(approval.expires_at).toLocaleTimeString()}</small>
                    </div>
                    <div className="pair-actions">
                      <Button
                        size="sm"
                        onClick={() => void decideApproval(approval.approval_id, true)}
                        disabled={busy}
                      >
                        Approve with fingerprint
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => void decideApproval(approval.approval_id, false)}
                        disabled={busy}
                      >
                        Reject
                      </Button>
                    </div>
                  </div>
                ))}
              </section>
            )}
            <div className="message-stream mobile-message-stream">
              {messages.length === 0 ? (
                <div className="empty-log"><span>Secure link established.</span><p>Your continuous JARVIS memory is available through the desktop Brain.</p></div>
              ) : messages.map((message) => (
                <article key={message.id} className={`message message-${message.role}`}>
                  <div><span>{message.role === "assistant" ? "JARVIS" : "YOU"}</span></div>
                  <p>{message.content}</p>
                </article>
              ))}
            </div>
            <form className="command-composer mobile-composer" onSubmit={submitMobile}>
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="Speak to JARVIS"
                rows={2}
              />
              <Button type="submit" size="icon" disabled={!draft.trim() || busy}>
                {busy ? <LoaderCircle className="spin" size={17} /> : <ArrowUp size={18} />}
              </Button>
            </form>
          </>
        )}
        {error && <p className="console-error">{error}</p>}
      </section>
    </main>
  );
}

export function JarvisProductShell() {
  const [mode, setMode] = useState<PlatformMode | null>(null);
  const [client, setClient] = useState<LocalBrainClient | null>(null);
  const [status, setStatus] = useState<BrainStatus | null>(null);
  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [modelStore, setModelStore] = useState("");
  const [manualModel, setManualModel] = useState("");
  const [pairEndpoint, setPairEndpoint] = useState("");
  const [pairOffer, setPairOffer] = useState<Record<string, unknown> | null>(null);
  const [pendingPairings, setPendingPairings] = useState<Array<Record<string, unknown>>>([]);
  const logEnd = useRef<HTMLDivElement>(null);
  const visual = useVisualProfile(status?.device.system_pressure);

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        const nextMode = await platformMode();
        if (cancelled) return;
        setMode(nextMode);
        if (nextMode === "desktop") {
          const nextClient = await bootstrapDesktopBrain();
          if (cancelled) return;
          setClient(nextClient);
          let nextStatus = await nextClient.status();
          if (nextStatus.owner_identity.face_enrolled) {
            try {
              await nextClient.verifyOwnerFace();
              nextStatus = await nextClient.status();
            } catch {
              // Camera may be unavailable at boot; manual retry remains visible.
            }
          }
          setStatus(nextStatus);
          setModelStore(nextStatus.model_store ?? "");
          setManualModel(nextStatus.manual_model ?? "");
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "JARVIS failed to initialize.");
      } finally {
        if (!cancelled) setBusy(false);
      }
    }

    void boot();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => {
    if (!client) return;
    const timer = window.setInterval(() => {
      client.status().then(setStatus).catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(timer);
  }, [client]);

  useEffect(() => {
    if (!client) return;
    const activeClient = client;
    let cancelled = false;
    async function refreshPairings() {
      try {
        const result = await activeClient.pendingPairings();
        if (!cancelled) setPendingPairings(result.pending);
      } catch {
        // Pairing status is auxiliary; Brain chat stays available.
      }
    }
    void refreshPairings();
    const timer = window.setInterval(refreshPairings, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [client]);

  async function initializeBrain() {
    if (!client) return;
    setBusy(true);
    setError("");
    const poll = window.setInterval(() => client.status().then(setStatus).catch(() => undefined), 800);
    try {
      setStatus(await client.setupLocalBrain(modelStore.trim() || undefined));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Local Brain setup failed.");
      setStatus(await client.status().catch(() => status));
    } finally {
      window.clearInterval(poll);
      setBusy(false);
    }
  }

  async function sendContent(content: string) {
    const clean = content.trim();
    if (!client || !clean || !status?.brain_ready || busy) return;
    setDraft("");
    setError("");
    setMessages((items) => [...items, {
      id: crypto.randomUUID(), role: "user", content: clean, at: new Date().toISOString(),
    }]);
    setBusy(true);
    try {
      const reply = await client.message(clean);
      setMessages((items) => [...items, {
        id: crypto.randomUUID(), role: "assistant", content: reply.text, at: new Date().toISOString(),
      }]);
      if (status.voice.available) {
        void client.voiceSpeak(reply.text)
          .then(() => client.status())
          .then(setStatus)
          .catch(() => undefined);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "JARVIS could not complete the request.");
    } finally {
      setBusy(false);
      void client.status().then(setStatus).catch(() => undefined);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    await sendContent(draft);
  }

  async function toggleVoice() {
    if (!client || busy) return;
    if (!status?.voice.available) {
      setError("Local microphone recognition is unavailable on this device.");
      return;
    }
    setError("");
    if (status.voice.state === "listening" || status.voice.state === "speaking") {
      await client.voiceStop().catch(() => undefined);
      setStatus(await client.status().catch(() => status));
      return;
    }
    setBusy(true);
    try {
      const listening = client.voiceListen("en-US", 8);
      setStatus({
        ...status,
        voice: { ...status.voice, state: "listening" },
      });
      const result = await listening;
      if (result.text) {
        setBusy(false);
        await sendContent(result.text);
      } else {
        setError("I didn't catch that. Try again or type your request.");
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "JARVIS voice recognition failed.");
    } finally {
      setBusy(false);
      void client.status().then(setStatus).catch(() => undefined);
    }
  }

  async function pickModelStore() {
    if (busy) return;
    setError("");
    try {
      const selected = await chooseModelStore();
      if (selected) setModelStore(selected);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not open the model folder picker.");
    }
  }

  async function enrollFace() {
    if (!client || busy) return;
    setBusy(true);
    setError("");
    try {
      await client.enrollOwnerFace();
      setStatus(await client.status());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Owner face enrollment failed.");
    } finally {
      setBusy(false);
    }
  }

  async function verifyFace() {
    if (!client || busy) return;
    setBusy(true);
    setError("");
    try {
      await client.verifyOwnerFace();
      setStatus(await client.status());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Owner face recognition failed.");
    } finally {
      setBusy(false);
    }
  }

  async function chooseManualModel(value: string) {
    setManualModel(value);
    if (!client) return;
    setError("");
    try {
      const next = await client.selectManualModel(value || undefined);
      setStatus(next);
      setManualModel(next.manual_model ?? "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Local model selection failed.");
      setManualModel(status?.manual_model ?? "");
    }
  }

  async function createPairOffer() {
    if (!client) return;
    setError("");
    try {
      setPairOffer(await client.createPairingOffer(pairEndpoint.trim() || undefined));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Pairing offer could not be created.");
    }
  }

  async function approvePairing(pairingId: string) {
    if (!client) return;
    setError("");
    try {
      await client.approvePairing(pairingId);
      const next = await client.pendingPairings();
      setPendingPairings(next.pending);
      setStatus(await client.status());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Phone approval failed.");
    }
  }

  async function rejectPairing(pairingId: string) {
    if (!client) return;
    setError("");
    try {
      await client.cancelPairing(pairingId);
      const next = await client.pendingPairings();
      setPendingPairings(next.pending);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Pairing cancellation failed.");
    }
  }

  const pressure = status?.device.system_pressure;
  const voiceActive = status?.voice.state === "listening" || status?.voice.state === "speaking";
  const state = busy ? "THINKING" : voiceActive || status?.brain_ready ? "LISTENING" : "MUTED";
  const ram = useMemo(() => {
    if (!status?.device.ram_total_gb) return "—";
    return `${status.device.ram_available_gb?.toFixed(1) ?? "?"} / ${status.device.ram_total_gb.toFixed(1)} GB`;
  }, [status]);

  if (mode === "mobile") return <MobileShell />;

  if (busy && !client) {
    return <div className="boot-screen"><div className="boot-pulse" aria-label="Starting JARVIS Brain" /></div>;
  }

  return (
    <main className="console-shell product-shell" data-visual-tier={visual.tier} data-refresh={visual.refreshHz}>
      <header className="console-header">
        <div className="wordmark"><span className="wordmark-mark">J</span> JARVIS <small>BRAIN / LOCAL-FIRST</small></div>
        <div className="header-state"><span className="state-dot" /><span>{status?.brain_ready ? "ONLINE" : "SETUP"}</span></div>
        <div className="operator-menu"><span>{visual.tier.toUpperCase()} · {visual.refreshHz} HZ</span></div>
      </header>

      <div className="console-grid">
        <section className="mission-log">
          <div className="panel-heading"><div><p className="section-index">MEMORY / CONTINUOUS</p><h2>Mission log</h2></div><RadioTower size={17} /></div>
          <div className="message-stream">
            {messages.length === 0 ? (
              <div className="empty-log">
                <span>{status?.brain_ready ? "JARVIS is ready." : "Local Brain is not initialized yet."}</span>
                <p>Your desktop and paired phone share one identity, one semantic memory and one owner-authority boundary.</p>
              </div>
            ) : messages.map((message) => (
              <article key={message.id} className={`message message-${message.role}`}>
                <div><span>{message.role === "assistant" ? "JARVIS" : "YOU"}</span><time>{new Date(message.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time></div>
                <p>{message.content}</p>
              </article>
            ))}
            <div ref={logEnd} />
          </div>
        </section>

        <section className="command-stage">
          <div className="state-caption"><span className="state-dot" /> JARVIS BRAIN / {status?.setup.phase?.toUpperCase() || "BOOT"}</div>
          <Reactor state={state} />
          <div className="voice-caption">
            <h1>{status?.brain_ready ? (busy ? "Reasoning" : "At your service.") : "Initialize local intelligence."}</h1>
            <p>{status?.setup.message || "Establishing local Brain link."}</p>
          </div>

          {!status?.brain_ready ? (
            <div className="brain-setup-card">
              <div className="pair-actions">
                <Input
                  value={modelStore}
                  onChange={(event) => setModelStore(event.target.value)}
                  placeholder="Local model folder (optional)"
                  aria-label="Local JARVIS model folder"
                  disabled={busy}
                />
                <Button type="button" variant="secondary" onClick={() => void pickModelStore()} disabled={busy}>
                  Choose folder
                </Button>
              </div>
              <Button onClick={initializeBrain} disabled={!client || busy}>
                {busy ? <LoaderCircle className="spin" size={16} /> : <Zap size={16} />}
                {busy ? "Preparing JARVIS Brain" : "Initialize Local Brain"}
              </Button>
              <p>
                Choose the folder where your local AI models already live, or leave it blank
                to use JARVIS defaults. JARVIS keeps its own loopback Ollama runtime and
                never requires a separate Python installation.
              </p>
            </div>
          ) : (
            <form className="command-composer" onSubmit={submit}>
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder="Speak to JARVIS"
                rows={2}
              />
              <Button
                type="button"
                variant="secondary"
                size="icon"
                aria-label={status?.voice.state === "listening" ? "Stop listening" : "Speak to JARVIS"}
                title={
                  status?.voice.available
                    ? status.voice.state === "listening"
                      ? "Stop listening"
                      : "Push to talk — processed locally on this device"
                    : "Local voice is unavailable on this device"
                }
                onClick={() => void toggleVoice()}
                disabled={!client || (!voiceActive && busy) || !status?.voice.available}
              >
                {status?.voice.state === "listening" ? <LoaderCircle className="spin" size={18} /> : <Mic2 size={18} />}
              </Button>
              <Button type="submit" size="icon" disabled={!draft.trim() || busy}><ArrowUp size={18} /></Button>
            </form>
          )}
          {error && <p className="console-error" role="alert">{error}</p>}
        </section>

        <aside className="systems-panel">
          <div className="panel-heading"><div><p className="section-index">SYSTEM / ADAPTIVE</p><h2>Operational state</h2></div><Activity size={17} /></div>
          <dl className="status-readout">
            <div><dt>JARVIS Brain</dt><dd data-on={status?.brain_ready}>{status?.brain_ready ? "LOCAL" : "SETUP"}</dd></div>
            <div><dt>Model store</dt><dd title={status?.model_store || undefined}>{status?.model_store ? "OWNER PATH" : "DEFAULT"}</dd></div>
            <div><dt>JARVIS Core</dt><dd data-on="true">ONLINE</dd></div>
            <div><dt>Cognition</dt><dd data-on="true">ADAPTIVE</dd></div>
            <div><dt>Voice</dt><dd data-on={status?.voice.available}>{status?.voice.available ? status.voice.state.toUpperCase() : "UNAVAILABLE"}</dd></div>
            <div><dt>Owner face</dt><dd data-on={status?.owner_identity.face_recognized}>{status?.owner_identity.face_recognized ? "RECOGNIZED" : status?.owner_identity.face_enrolled ? "ENROLLED" : "NOT ENROLLED"}</dd></div>
            <div><dt>Face engine</dt><dd>{status?.owner_identity.face_engine === "opencv_sface_2021dec" ? "SFACE" : status?.owner_identity.face_engine ?? "—"}</dd></div>
            <div><dt>Phone approvals</dt><dd data-on={(status?.approvals.pending_exact_actions ?? 0) > 0}>{status?.approvals.pending_exact_actions ?? 0} PENDING</dd></div>
            <div><dt>Memory</dt><dd data-on="true"><ShieldCheck size={13} /> CONTINUOUS</dd></div>
            <div><dt>RAM available</dt><dd>{ram}</dd></div>
            <div><dt>System pressure</dt><dd>{pressure == null ? "—" : `${Math.round(pressure * 100)}%`}</dd></div>
            <div><dt>Power</dt><dd>{status?.device.power_source?.toUpperCase() || "—"}</dd></div>
            <div><dt>Visual profile</dt><dd data-on="true"><Cpu size={13} /> {visual.tier.toUpperCase()}</dd></div>
          </dl>

          <section className="device-link-card">
            <div className="capability-heading"><span>Owner identity</span><b>{status?.owner_identity.face_recognized ? "LIVE" : "FACE"}</b></div>
            <p>
              {status?.owner_identity.face_enrolled
                ? "JARVIS can recognize your enrolled face from the laptop camera. Face identity never replaces phone biometric approval for major changes."
                : "Enroll your face once. JARVIS stores numeric templates only; camera frames are discarded."}
            </p>
            <div className="pair-actions">
              <Button variant="secondary" onClick={() => void (status?.owner_identity.face_enrolled ? verifyFace() : enrollFace())} disabled={!client || busy}>
                <ShieldCheck size={15} />
                {status?.owner_identity.face_enrolled ? "Recognize me" : "Enroll owner face"}
              </Button>
            </div>
          </section>

          <section className="device-link-card">
            <div className="capability-heading"><span>Local models</span><b>{status?.available_models.length ?? 0}</b></div>
            <p>JARVIS chooses hidden experts automatically. Override only when you want a specific installed local model to join the council.</p>
            <select
              value={manualModel}
              onChange={(event) => void chooseManualModel(event.target.value)}
              aria-label="Optional manual local model"
              disabled={!client || busy}
            >
              <option value="">Automatic expert selection</option>
              {(status?.available_models ?? []).map((model) => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </section>

          <section className="device-link-card">
            <div className="capability-heading"><span>Device Link</span><b>{status?.paired_devices.length ?? 0}</b></div>
            <p>Pair your phone to this same JARVIS Brain. The secure gateway is detected automatically when Tailscale is available.</p>
            <Input
              value={pairEndpoint}
              onChange={(event) => setPairEndpoint(event.target.value)}
              placeholder="Advanced: override MagicDNS endpoint"
            />
            <Button variant="secondary" onClick={createPairOffer} disabled={!client}>
              <Link2 size={15} /> Create pairing QR
            </Button>
            {pairOffer && (
              <div className="pair-qr-card">
                {typeof pairOffer.qr_svg_data_url === "string" && (
                  <img
                    src={pairOffer.qr_svg_data_url}
                    alt="Scan to pair the JARVIS mobile companion"
                    className="pair-qr-image"
                  />
                )}
                <div>
                  <strong>Scan with JARVIS mobile</strong>
                  <p>Expires {typeof pairOffer.expires_at === "string" ? new Date(pairOffer.expires_at).toLocaleTimeString() : "soon"}.</p>
                </div>
              </div>
            )}
            {pendingPairings.length > 0 && (
              <div className="pair-pending-list">
                <span className="section-index">OWNER APPROVAL REQUIRED</span>
                {pendingPairings.map((item) => {
                  const pairingId = String(item.pairing_id || "");
                  return (
                    <div className="pair-pending-item" key={pairingId}>
                      <div>
                        <strong>{String(item.candidate_device || "Mobile JARVIS")}</strong>
                        <small>{String(item.candidate_role || "companion")}</small>
                      </div>
                      <div className="pair-actions">
                        <Button size="sm" onClick={() => void approvePairing(pairingId)}>Approve</Button>
                        <Button size="sm" variant="secondary" onClick={() => void rejectPairing(pairingId)}>Reject</Button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </aside>
      </div>
    </main>
  );
}

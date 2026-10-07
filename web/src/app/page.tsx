"use client";

import { useEffect, useState } from "react";
import { AuthScreen } from "@/components/auth-screen";
import { JarvisConsole } from "@/components/jarvis-console";
import { JarvisProductShell } from "@/components/jarvis-product-shell";
import { Onboarding } from "@/components/onboarding";
import { WakeWordBridge } from "@/components/wake-word-bridge";
import { api, getToken, setToken, type Session, type User } from "@/lib/api";
import { isNativeJarvis } from "@/lib/jarvis-runtime";

type View = "loading" | "native" | "auth" | "onboarding" | "console";

export default function Home() {
  const [view, setView] = useState<View>("loading");
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    if (isNativeJarvis()) {
      queueMicrotask(() => setView("native"));
      return;
    }
    if (!getToken()) {
      queueMicrotask(() => setView("auth"));
      return;
    }
    api<User>("/auth/me")
      .then((nextUser) => {
        setUser(nextUser);
        setView(nextUser.gemini_configured ? "console" : "onboarding");
      })
      .catch(() => {
        setToken(null);
        setView("auth");
      });
  }, []);

  function handleSession(session: Session) {
    setToken(session.access_token);
    setUser(session.user);
    setView(session.user.gemini_configured ? "console" : "onboarding");
  }

  function signOut() {
    api<void>("/auth/logout", { method: "POST" }).catch(() => undefined);
    setToken(null);
    setUser(null);
    setView("auth");
  }

  if (view === "loading") {
    return <div className="boot-screen"><div className="boot-pulse" aria-label="Initializing JARVIS" /></div>;
  }
  if (view === "native") {
    return (
      <>
        <JarvisProductShell />
        <WakeWordBridge />
      </>
    );
  }
  if (view === "auth") return <AuthScreen onSession={handleSession} />;
  if (view === "onboarding" && user) {
    return <Onboarding user={user} onComplete={() => setView("console")} onSignOut={signOut} />;
  }
  if (view === "console" && user) return <JarvisConsole user={user} onSignOut={signOut} />;
  return null;
}

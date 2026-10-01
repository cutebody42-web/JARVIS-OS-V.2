#!/usr/bin/env bash
set -euo pipefail

APK="${1:-}"
if [[ -z "$APK" || ! -f "$APK" ]]; then
  echo "JARVIS emulator APK is missing: $APK" >&2
  exit 1
fi

mkdir -p artifacts/real-cloud/android

adb wait-for-device

package_ready=false
for attempt in $(seq 1 45); do
  if adb shell cmd package list packages >/dev/null 2>&1; then
    package_ready=true
    break
  fi
  sleep 2
done
if [[ "$package_ready" != "true" ]]; then
  echo "Android package manager did not become responsive" >&2
  adb shell getprop > artifacts/real-cloud/android/getprop-on-package-timeout.txt || true
  exit 1
fi

installed=false
for attempt in 1 2 3; do
  if adb install --no-streaming -r "$APK"; then
    installed=true
    break
  fi
  echo "APK install attempt $attempt failed; waiting for emulator package service" >&2
  adb reconnect device || true
  adb wait-for-device
  sleep $((attempt * 10))
done
if [[ "$installed" != "true" ]]; then
  echo "JARVIS APK could not be installed after 3 attempts" >&2
  adb shell dumpsys package > artifacts/real-cloud/android/packages-on-install-failure.txt || true
  exit 1
fi

adb shell pm path ai.jarvis.app > artifacts/real-cloud/android/package-path.txt
adb shell dumpsys package ai.jarvis.app > artifacts/real-cloud/android/package.txt
cat artifacts/real-cloud/android/package-path.txt

if ! grep -q "android.permission.USE_BIOMETRIC" artifacts/real-cloud/android/package.txt; then
  echo "Installed JARVIS APK is missing Android biometric permission" >&2
  exit 1
fi

adb logcat -c || true
adb shell monkey -p ai.jarvis.app -c android.intent.category.LAUNCHER 1
sleep 12

adb shell dumpsys activity activities > artifacts/real-cloud/android/activity.txt
if ! grep -q "ai.jarvis.app" artifacts/real-cloud/android/activity.txt; then
  echo "JARVIS Android app is not present in activity state" >&2
  exit 1
fi

adb shell am start -W -a android.intent.action.VIEW -d "jarvis://pair?smoke=cloud-validation" ai.jarvis.app > artifacts/real-cloud/android/deep-link.txt
sleep 4

adb exec-out screencap -p > artifacts/real-cloud/android/android-screen.png
adb shell uiautomator dump /sdcard/jarvis-window.xml >/dev/null || true
adb pull /sdcard/jarvis-window.xml artifacts/real-cloud/android/jarvis-window.xml >/dev/null 2>&1 || true
adb logcat -d > artifacts/real-cloud/android/logcat.txt

if grep -E "FATAL EXCEPTION: main|Process: ai\.jarvis\.app.*has died" artifacts/real-cloud/android/logcat.txt; then
  echo "JARVIS Android app crashed in the emulator" >&2
  exit 1
fi

if [[ ! -s artifacts/real-cloud/android/android-screen.png ]]; then
  echo "Android emulator screenshot evidence is empty" >&2
  exit 1
fi

echo "JARVIS Android emulator smoke passed."

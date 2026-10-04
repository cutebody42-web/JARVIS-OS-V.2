//! A phone has one active desktop trust record. Removing it never removes the
//! phone's private identity or sends a decision to the previously paired host.

use serde::{Deserialize, Serialize};
use std::{fs, io::{Read, Write}, path::Path, sync::Mutex};

static TRUST_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct DesktopTrust {
    pub(crate) desktop_device: String,
    pub(crate) desktop_public_key: String,
    pub(crate) desktop_endpoint: String,
}

fn read_trust(path: &Path) -> Result<Option<DesktopTrust>, String> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(value) => value,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("Saved laptop pairing could not be read. Reset it explicitly before pairing again.".into()),
    };
    if !metadata.is_file() || metadata.len() > 65_536 {
        return Err("Saved laptop pairing is invalid. Reset it explicitly before pairing again.".into());
    }
    let mut data = Vec::new();
    fs::File::open(path).and_then(|file| file.take(65_537).read_to_end(&mut data))
        .map_err(|_| "Saved laptop pairing could not be read. Reset it explicitly before pairing again.".to_string())?;
    if data.len() > 65_536 {
        return Err("Saved laptop pairing is too large.".into());
    }
    let trust: DesktopTrust = serde_json::from_slice(&data)
        .map_err(|_| "Saved laptop pairing is damaged. Reset it explicitly before pairing again.".to_string())?;
    if trust.desktop_device.trim().is_empty() || trust.desktop_device.len() > 256
        || trust.desktop_public_key.trim().is_empty() || trust.desktop_public_key.len() > 256
        || trust.desktop_endpoint.trim().is_empty() || trust.desktop_endpoint.len() > 2048 {
        return Err("Saved laptop pairing is invalid. Reset it explicitly before pairing again.".into());
    }
    Ok(Some(trust))
}

pub(crate) fn inspect_desktop_trust(path: &Path) -> Result<Option<DesktopTrust>, String> {
    let _guard = TRUST_LOCK.lock().map_err(|_| "JARVIS desktop trust is unavailable.".to_string())?;
    read_trust(path)
}

pub(crate) fn load_desktop_trust(path: &Path) -> Result<DesktopTrust, String> {
    let _guard = TRUST_LOCK
        .lock()
        .map_err(|_| "JARVIS desktop trust is unavailable.".to_string())?;
    read_trust(path)?.ok_or_else(|| "This mobile JARVIS is not paired with a desktop Brain.".to_string())
}

pub(crate) fn store_desktop_trust(path: &Path, trust: &DesktopTrust) -> Result<(), String> {
    let _guard = TRUST_LOCK
        .lock()
        .map_err(|_| "JARVIS desktop trust is unavailable.".to_string())?;
    if read_trust(path)?.is_some() {
        return Err("Disconnect the current laptop before pairing another JARVIS Brain.".into());
    }
    let data = serde_json::to_vec(trust)
        .map_err(|_| "JARVIS desktop trust could not be encoded.".to_string())?;
    fs::OpenOptions::new().write(true).create_new(true).open(path)
        .and_then(|mut file| file.write_all(&data))
        .map_err(|_| "JARVIS desktop trust could not be stored.".to_string())
}

pub(crate) fn disconnect_desktop_trust(
    path: &Path,
    expected_desktop: Option<&str>,
    confirmed: bool,
) -> Result<(), String> {
    let _guard = TRUST_LOCK
        .lock()
        .map_err(|_| "JARVIS desktop trust is unavailable.".to_string())?;
    match read_trust(path) {
        Ok(None) => return Ok(()),
        Ok(Some(current)) => {
            if !confirmed {
                return Err("Confirm disconnecting your current laptop first.".into());
            }
            if expected_desktop != Some(current.desktop_device.as_str()) {
                return Err("The connected laptop changed. Review it before disconnecting.".into());
            }
        }
        Err(_) if !confirmed => return Err("Confirm resetting the damaged laptop pairing first.".into()),
        Err(_) => {},
    }
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err("JARVIS could not remove the desktop pairing.".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir()
                .join(format!("jarvis-companion-{}-{nonce}", std::process::id()));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
        fn trust(&self) -> std::path::PathBuf {
            self.0.join("desktop-trust.json")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn desktop(name: &str) -> DesktopTrust {
        DesktopTrust {
            desktop_device: name.into(),
            desktop_public_key: "public-key".into(),
            desktop_endpoint: "http://100.90.10.3:8765".into(),
        }
    }

    #[test]
    fn disconnect_requires_confirmation_bound_to_the_current_laptop() {
        let fixture = Fixture::new();
        store_desktop_trust(&fixture.trust(), &desktop("laptop-a")).unwrap();
        assert!(disconnect_desktop_trust(&fixture.trust(), Some("laptop-a"), false).is_err());
        assert!(disconnect_desktop_trust(&fixture.trust(), Some("laptop-b"), true).is_err());
        assert!(disconnect_desktop_trust(&fixture.trust(), None, true).is_err());
        assert_eq!(
            load_desktop_trust(&fixture.trust()).unwrap().desktop_device,
            "laptop-a"
        );
    }

    #[test]
    fn disconnect_preserves_the_phone_identity_and_allows_explicit_new_pairing() {
        let fixture = Fixture::new();
        let seed = fixture.0.join("mobile-device.ed25519");
        fs::write(&seed, b"unchanged-private-phone-identity").unwrap();
        store_desktop_trust(&fixture.trust(), &desktop("laptop-a")).unwrap();
        assert!(store_desktop_trust(&fixture.trust(), &desktop("laptop-b")).is_err());
        disconnect_desktop_trust(&fixture.trust(), Some("laptop-a"), true).unwrap();
        assert!(load_desktop_trust(&fixture.trust()).is_err());
        assert_eq!(
            fs::read(&seed).unwrap(),
            b"unchanged-private-phone-identity"
        );
        store_desktop_trust(&fixture.trust(), &desktop("laptop-b")).unwrap();
        assert_eq!(
            load_desktop_trust(&fixture.trust()).unwrap().desktop_device,
            "laptop-b"
        );
        assert_eq!(
            fs::read(&seed).unwrap(),
            b"unchanged-private-phone-identity"
        );
    }

    #[test]
    fn disconnecting_an_already_unpaired_phone_is_idempotent() {
        let fixture = Fixture::new();
        disconnect_desktop_trust(&fixture.trust(), None, false).unwrap();
        disconnect_desktop_trust(&fixture.trust(), None, true).unwrap();
        assert!(!fixture.trust().exists());
    }

    #[test]
    fn damaged_existing_trust_cannot_be_silently_overwritten_or_removed() {
        let fixture = Fixture::new();
        fs::write(fixture.trust(), b"broken pairing").unwrap();
        assert!(inspect_desktop_trust(&fixture.trust()).is_err());
        assert!(store_desktop_trust(&fixture.trust(), &desktop("laptop-b")).is_err());
        assert!(disconnect_desktop_trust(&fixture.trust(), None, false).is_err());
        assert_eq!(fs::read(fixture.trust()).unwrap(), b"broken pairing");
        disconnect_desktop_trust(&fixture.trust(), None, true).unwrap();
        store_desktop_trust(&fixture.trust(), &desktop("laptop-b")).unwrap();
        assert_eq!(load_desktop_trust(&fixture.trust()).unwrap().desktop_device, "laptop-b");
    }

    #[test]
    fn oversized_or_nonfile_pairing_is_not_treated_as_unpaired() {
        let fixture = Fixture::new();
        fs::write(fixture.trust(), vec![b' '; 65_537]).unwrap();
        assert!(inspect_desktop_trust(&fixture.trust()).is_err());
        assert!(store_desktop_trust(&fixture.trust(), &desktop("laptop-b")).is_err());
        fs::remove_file(fixture.trust()).unwrap();
        fs::create_dir(fixture.trust()).unwrap();
        assert!(inspect_desktop_trust(&fixture.trust()).is_err());
        assert!(disconnect_desktop_trust(&fixture.trust(), None, true).is_err());
    }
}

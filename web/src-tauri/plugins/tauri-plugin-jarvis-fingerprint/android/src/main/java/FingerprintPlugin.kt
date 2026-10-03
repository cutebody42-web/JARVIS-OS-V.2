package ai.jarvis.fingerprint

import android.app.Activity
import android.app.AlertDialog
import android.hardware.fingerprint.FingerprintManager
import android.os.CancellationSignal
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin

@InvokeArg
class AuthenticateArgs { lateinit var reason: String }

/**
 * FingerprintManager is intentionally used instead of generic BiometricPrompt.
 * The latter can accept face/iris on multi-modality phones. This sensor API
 * verifies only enrolled fingerprints, with no face or lock-screen fallback.
 */
@Suppress("DEPRECATION")
@TauriPlugin
class FingerprintPlugin(private val activity: Activity) : Plugin(activity) {
    private val handler = Handler(Looper.getMainLooper())
    private var active: Session? = null

    private class Session(val invoke: Invoke) {
        val cancellation = CancellationSignal()
        var dialog: AlertDialog? = null
        var timeout: Runnable? = null
        var failedAttempts = 0
    }

    private fun sensor(): FingerprintManager? = activity.getSystemService(FingerprintManager::class.java)

    private fun unavailableReason(): String? = try {
        val sensor = sensor()
        when {
            sensor == null || !sensor.isHardwareDetected -> "This phone has no available fingerprint sensor."
            !sensor.hasEnrolledFingerprints() -> "Enroll a fingerprint in Android settings first."
            else -> null
        }
    } catch (_: SecurityException) {
        "JARVIS fingerprint permission is unavailable."
    }

    @Command
    fun status(invoke: Invoke) {
        val error = unavailableReason()
        val result = JSObject()
        result.put("isAvailable", error == null)
        result.put("biometryType", if (error == null) 1 else 0)
        result.put("error", error)
        invoke.resolve(result)
    }

    @Command
    fun authenticate(invoke: Invoke) {
        val args = invoke.parseArgs(AuthenticateArgs::class.java)
        activity.runOnUiThread {
            val error = unavailableReason()
            if (error != null || activity.isFinishing || activity.isDestroyed) {
                invoke.reject(error ?: "JARVIS owner authentication is unavailable.")
                return@runOnUiThread
            }
            if (active != null) {
                invoke.reject("A JARVIS fingerprint approval is already in progress.")
                return@runOnUiThread
            }
            val session = Session(invoke)
            active = session
            try {
                val reason = args.reason.trim().take(500)
                val dialog = AlertDialog.Builder(activity)
                    .setTitle("JARVIS owner approval")
                    .setMessage("$reason\n\nTouch your enrolled fingerprint sensor to continue.")
                    .setNegativeButton("Cancel") { _, _ -> finish(session, "Fingerprint approval was cancelled.") }
                    .create()
                session.dialog = dialog
                dialog.setOnCancelListener { finish(session, "Fingerprint approval was cancelled.") }
                dialog.setOnDismissListener {
                    if (active === session) finish(session, "Fingerprint approval was cancelled.")
                }
                dialog.window?.addFlags(WindowManager.LayoutParams.FLAG_SECURE)
                dialog.show()
                val timeout = Runnable { finish(session, "Fingerprint approval timed out. Please try again.") }
                session.timeout = timeout
                handler.postDelayed(timeout, 30000)
                sensor()!!.authenticate(null, session.cancellation, 0,
                    object : FingerprintManager.AuthenticationCallback() {
                        override fun onAuthenticationSucceeded(result: FingerprintManager.AuthenticationResult) {
                            finish(session, null)
                        }

                        override fun onAuthenticationError(code: Int, message: CharSequence) {
                            finish(session, "Fingerprint verification failed: $message")
                        }

                        override fun onAuthenticationHelp(code: Int, message: CharSequence) {
                            if (active === session) session.dialog?.setMessage(message)
                        }

                        override fun onAuthenticationFailed() {
                            if (active !== session) return
                            session.failedAttempts += 1
                            if (session.failedAttempts >= 3) {
                                finish(session, "Fingerprint was not recognized. Please try again.")
                            } else {
                                session.dialog?.setMessage("Fingerprint was not recognized. Touch your enrolled fingerprint sensor again.")
                            }
                        }
                    }, handler)
            } catch (_: Exception) {
                finish(session, "JARVIS could not start fingerprint verification.")
            }
        }
    }

    private fun finish(session: Session, error: String?) {
        if (active !== session) return
        // Clear ownership first: sensor cancellation and dialog dismissal may
        // deliver callbacks, which must never resolve an old request twice.
        active = null
        session.timeout?.let { handler.removeCallbacks(it) }
        session.cancellation.cancel()
        session.dialog?.dismiss()
        if (error != null) {
            session.invoke.reject(error)
        } else {
            val result = JSObject()
            result.put("userVerified", true)
            result.put("biometryType", "fingerprint")
            session.invoke.resolve(result)
        }
    }

    override fun onPause(activity: AppCompatActivity) {
        active?.let { finish(it, "Fingerprint approval cancelled when JARVIS left the foreground.") }
    }

    override fun onDestroy(activity: AppCompatActivity) {
        active?.let { finish(it, "Fingerprint approval cancelled when JARVIS closed.") }
    }
}

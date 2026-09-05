import { Cpu } from "lucide-react"
import { WORKER_SHORT_NAMES } from "../lib/constants"

function statusFor(phase, result) {
  if (phase === "idle") return { dot: "bg-ink-faint", label: "AWAITING QUERY", text: "text-ink-faint" }
  if (phase === "analyzing")
    return { dot: "bg-accent animate-pulse-soft", label: "ANALYZING REQUEST...", text: "text-accent" }

  if (!result) return { dot: "bg-ink-faint", label: "IDLE", text: "text-ink-faint" }
  if (result.status === "ok") {
    const workerLabel = WORKER_SHORT_NAMES[result.worker] || result.worker
    return {
      dot: "bg-online",
      label: `ROUTED -> ${workerLabel?.toUpperCase() || "WORKER"}`,
      text: "text-online",
    }
  }
  if (result.status === "rejected") {
    return { dot: "bg-warn", label: "REQUEST REJECTED", text: "text-warn" }
  }
  return { dot: "bg-error", label: "ROUTER ERROR", text: "text-error" }
}

export default function BossNode({ phase, result }) {
  const status = statusFor(phase, result)
  const isActive = phase === "analyzing" || phase === "settling"

  return (
    <div
      className={`relative w-full max-w-md rounded-xl border bg-panel p-5 transition-colors ${
        isActive ? "border-accent/50 shadow-[0_0_0_1px_var(--color-accent),0_0_28px_-6px_var(--color-accent)]" : "border-line"
      }`}
    >
      {isActive && (
        <div className="pointer-events-none absolute inset-0 overflow-hidden rounded-xl">
          <div className="absolute inset-y-0 left-0 w-1/3 bg-gradient-to-r from-transparent via-accent/10 to-transparent animate-scan" />
        </div>
      )}

      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-accent/30 bg-accent-soft">
          <Cpu className="h-4.5 w-4.5 text-accent" strokeWidth={1.75} />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-bold tracking-wide text-ink">THE BOSS</p>
          <p className="truncate text-[11px] font-mono text-ink-faint">Qwen 2.5 - 3B . Router / Function-Caller</p>
        </div>
      </div>

      <div className="mt-3.5 flex items-center gap-2 border-t border-line-soft pt-3">
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${status.dot}`} />
        <span className={`font-mono text-[11px] font-semibold tracking-wide ${status.text}`}>
          {status.label}
        </span>
      </div>

      {phase === "result" && result?.audit_summary && (
        <p className="mt-2 text-[11px] leading-relaxed text-ink-soft">{result.audit_summary}</p>
      )}
      {phase === "result" && result?.status === "rejected" && result?.validation?.reason && (
        <p className="mt-1.5 text-[11px] leading-relaxed text-warn">{result.validation.reason}</p>
      )}
    </div>
  )
}

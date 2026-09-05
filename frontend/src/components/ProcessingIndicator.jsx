const STATUS_STYLES = {
  IDLE: { label: "IDLE", dot: "bg-ink-faint", text: "text-ink-faint", pulse: false },
  ANALYZING: { label: "ANALYZING", dot: "bg-accent", text: "text-accent", pulse: true },
  ACTIVE: { label: "ACTIVE", dot: "bg-current", text: "text-current", pulse: true },
  COMPLETED: { label: "COMPLETED", dot: "bg-online", text: "text-online", pulse: false },
  REJECTED: { label: "REJECTED", dot: "bg-error", text: "text-error", pulse: false },
}

export default function ProcessingIndicator({ status = "IDLE", className = "" }) {
  const style = STATUS_STYLES[status] || STATUS_STYLES.IDLE
  return (
    <span className={`inline-flex items-center gap-1.5 font-mono text-[10px] font-semibold tracking-wider ${style.text} ${className}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${style.dot} ${style.pulse ? "animate-pulse-soft" : ""}`} />
      {style.label}
    </span>
  )
}

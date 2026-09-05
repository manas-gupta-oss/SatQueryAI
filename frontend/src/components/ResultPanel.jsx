import { AlertOctagon, CheckCircle2, ChevronDown, FlaskConical, XCircle } from "lucide-react"
import { useState } from "react"
import { TASK_TYPE_LABELS, WORKER_SHORT_NAMES } from "../lib/constants"

function ConfidenceBar({ label, value }) {
  if (value === null || value === undefined) return null
  const pct = Math.round(value * 100)
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[10.5px] font-mono text-ink-faint">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-line-soft">
        <div className="h-full rounded-full bg-accent" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export default function ResultPanel({ result, images }) {
  const [traceOpen, setTraceOpen] = useState(false)
  if (!result) return null

  const isOk = result.status === "ok"
  const isRejected = result.status === "rejected"
  const isError = result.status === "error" || (!isOk && !isRejected)

  return (
    <section className="w-full max-w-4xl animate-fade-in-up rounded-xl border border-line bg-panel p-5 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line-soft pb-4">
        <div className="flex items-center gap-2">
          {isOk && <CheckCircle2 className="h-4.5 w-4.5 text-online" />}
          {isRejected && <XCircle className="h-4.5 w-4.5 text-warn" />}
          {isError && <AlertOctagon className="h-4.5 w-4.5 text-error" />}
          <h3 className="text-sm font-bold tracking-wide text-ink">SatQuery Response</h3>
        </div>
        {result._demo && (
          <span className="flex items-center gap-1.5 rounded-full border border-warn/40 bg-warn/10 px-2.5 py-1 text-[10px] font-mono font-semibold text-warn">
            <FlaskConical className="h-3 w-3" />
            DEMO SIMULATION - NOT A LIVE MODEL RESPONSE
          </span>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <p className="text-[10.5px] font-medium uppercase tracking-wide text-ink-faint">Router Decision</p>
          <p className="mt-1 text-sm font-semibold text-ink">
            {isOk ? `-> ${WORKER_SHORT_NAMES[result.worker] || result.worker}` : "No worker dispatched"}
          </p>
          {result.task_type && (
            <p className="mt-0.5 font-mono text-[11px] text-ink-soft">
              {TASK_TYPE_LABELS[result.task_type] || result.task_type}
            </p>
          )}
        </div>
        <div>
          <p className="text-[10.5px] font-medium uppercase tracking-wide text-ink-faint">Status</p>
          <p
            className={`mt-1 text-sm font-semibold ${
              isOk ? "text-online" : isRejected ? "text-warn" : "text-error"
            }`}
          >
            {isOk ? "Completed" : isRejected ? "Rejected" : "Error"}
          </p>
        </div>
      </div>

      {result.audit_summary && (
        <div className="mt-4">
          <p className="text-[10.5px] font-medium uppercase tracking-wide text-ink-faint">Reason</p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-soft">{result.audit_summary}</p>
        </div>
      )}

      {isRejected && result.validation?.reason && (
        <div className="mt-4 rounded-lg border border-warn/30 bg-warn/5 p-3">
          <p className="text-[10.5px] font-medium uppercase tracking-wide text-warn">Validation Failure</p>
          <p className="mt-1 text-[12.5px] leading-relaxed text-ink-soft">{result.validation.reason}</p>
          {result.validation.code && (
            <p className="mt-1 font-mono text-[10px] text-ink-faint">code: {result.validation.code}</p>
          )}
        </div>
      )}

      {isError && (result.error || result.validation?.reason) && (
        <div className="mt-4 rounded-lg border border-error/30 bg-error/5 p-3">
          <p className="text-[10.5px] font-medium uppercase tracking-wide text-error">Error</p>
          <p className="mt-1 text-[12.5px] leading-relaxed text-ink-soft">
            {result.error || result.validation?.reason}
          </p>
        </div>
      )}

      {isOk && result.answer && (
        <div className="mt-4">
          <p className="text-[10.5px] font-medium uppercase tracking-wide text-ink-faint">Result</p>
          <p className="mt-1.5 whitespace-pre-wrap text-[13.5px] leading-relaxed text-ink">
            {result.answer}
          </p>
        </div>
      )}

      {isOk && (result.router_confidence !== null || result.confidence !== null) && (
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
          <ConfidenceBar label="Router confidence" value={result.router_confidence} />
          <ConfidenceBar label="Worker confidence" value={result.confidence} />
        </div>
      )}

      {result.assumptions?.length > 0 && (
        <div className="mt-4">
          <p className="text-[10.5px] font-medium uppercase tracking-wide text-ink-faint">Assumptions</p>
          <ul className="mt-1 space-y-1">
            {result.assumptions.map((a, i) => (
              <li key={i} className="text-[12px] leading-relaxed text-ink-soft">
                . {a}
              </li>
            ))}
          </ul>
        </div>
      )}

      {images?.length > 0 && (
        <div className="mt-4">
          <p className="text-[10.5px] font-medium uppercase tracking-wide text-ink-faint">Images Used</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {images.map((img) => (
              <img
                key={img.clientId}
                src={img.previewUrl}
                alt={img.file.name}
                className="h-16 w-16 rounded-md border border-line object-cover"
              />
            ))}
          </div>
        </div>
      )}

      {result.execution_trace?.length > 0 && (
        <div className="mt-4 border-t border-line-soft pt-3">
          <button
            type="button"
            onClick={() => setTraceOpen((v) => !v)}
            className="flex items-center gap-1.5 text-[10.5px] font-medium uppercase tracking-wide text-ink-faint hover:text-ink-soft"
          >
            <ChevronDown className={`h-3.5 w-3.5 transition-transform ${traceOpen ? "rotate-180" : ""}`} />
            Execution Trace
          </button>
          {traceOpen && (
            <div className="mt-2 space-y-1 rounded-lg bg-bg-elevated p-3 font-mono text-[10.5px] leading-relaxed text-ink-faint">
              {result.execution_trace.map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

import { GitBranch, ListTree, Cpu, Boxes } from "lucide-react"
import { SYSTEM_PANEL, WORKER_SHORT_NAMES } from "../lib/constants"

function Row({ icon: Icon, label, value }) {
  return (
    <div className="flex items-start gap-2.5">
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint" strokeWidth={1.75} />
      <div className="min-w-0">
        <p className="text-[10px] font-medium uppercase tracking-wide text-ink-faint">{label}</p>
        <p className="text-[12px] text-ink-soft">{value}</p>
      </div>
    </div>
  )
}

export default function ArchitecturePanel({ workers }) {
  return (
    <div className="w-full max-w-4xl rounded-xl border border-line-soft bg-panel-soft p-4 sm:p-5">
      <p className="mb-3 text-[10.5px] font-bold uppercase tracking-[0.14em] text-ink-faint">
        System Architecture
      </p>
      <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-4">
        <Row icon={GitBranch} label="Orchestration" value={SYSTEM_PANEL.orchestration} />
        <Row icon={ListTree} label="Router / Boss" value={SYSTEM_PANEL.router} />
        <Row icon={Cpu} label="Vision-Language Engine" value={SYSTEM_PANEL.engine} />
        <Row
          icon={Boxes}
          label="Workers"
          value={workers.map((w) => WORKER_SHORT_NAMES[w.id] || w.short_name).join(" . ")}
        />
      </div>
    </div>
  )
}

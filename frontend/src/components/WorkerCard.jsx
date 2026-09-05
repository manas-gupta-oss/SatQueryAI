import { Image as ImageIcon, Layers, GitCompareArrows } from "lucide-react"
import { WORKER_ACCENTS, WORKER_SHORT_NAMES } from "../lib/constants"
import ProcessingIndicator from "./ProcessingIndicator"

const ICONS = {
  worker1: ImageIcon,
  worker2: Layers,
  worker3: GitCompareArrows,
}

export default function WorkerCard({ worker, status }) {
  const accent = WORKER_ACCENTS[worker.id] || WORKER_ACCENTS.worker1
  const Icon = ICONS[worker.id] || ImageIcon
  const isEngaged = status === "ACTIVE" || status === "COMPLETED"

  return (
    <div
      className={`flex flex-col gap-2.5 rounded-xl border bg-panel p-4 transition-all duration-300 ${
        isEngaged ? `${accent.border} ${accent.bg}` : "border-line"
      } ${status === "ACTIVE" ? accent.glow : ""}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border ${isEngaged ? accent.border : "border-line"}`}>
          <Icon className={`h-4 w-4 ${isEngaged ? accent.text : "text-ink-faint"}`} strokeWidth={1.75} />
        </div>
        <ProcessingIndicator status={status} className={status === "ACTIVE" ? accent.text : ""} />
      </div>

      <div>
        <p className="text-[13px] font-semibold text-ink">
          {WORKER_SHORT_NAMES[worker.id] || worker.short_name || worker.display_name}
        </p>
        <p className={`mt-0.5 text-[10.5px] font-mono ${isEngaged ? accent.text : "text-ink-faint"}`}>
          {worker.tuned_on}
        </p>
      </div>

      <p className="text-[11px] leading-relaxed text-ink-soft">
        {worker.description}
      </p>

      <p className="text-[10px] font-mono text-ink-faint">
        {worker.min_images === worker.max_images
          ? `${worker.min_images} image${worker.min_images > 1 ? "s" : ""}`
          : `${worker.min_images}-${worker.max_images} images`}
      </p>
    </div>
  )
}

import { Clock, GitCompareArrows, Map, RefreshCw, Search } from "lucide-react"
import { EXAMPLE_QUERIES } from "../lib/constants"

const ICONS = { Search, GitCompareArrows, RefreshCw, Map, Clock }

export default function ExampleQueries({ onPick }) {
  return (
    <div className="space-y-1.5">
      {EXAMPLE_QUERIES.map((ex) => {
        const Icon = ICONS[ex.icon]
        return (
          <button
            key={ex.text}
            type="button"
            onClick={() => onPick(ex.text)}
            className="flex w-full items-center gap-2.5 rounded-lg border border-line-soft bg-panel px-3 py-2 text-left text-[12.5px] text-ink-soft transition-colors hover:border-cyan-500/30 hover:bg-panel-soft hover:text-ink"
          >
            <Icon className="h-3.5 w-3.5 shrink-0 text-cyan-400/80" strokeWidth={1.75} />
            <span className="truncate">{ex.text}</span>
          </button>
        )
      })}
    </div>
  )
}

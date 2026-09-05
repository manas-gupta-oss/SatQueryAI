import { Sparkles } from "lucide-react"

export default function Logo() {
  return (
    <div>
      <div className="flex items-center gap-2">
        <Sparkles className="h-6 w-6 shrink-0 text-cyan-400" strokeWidth={2} fill="currentColor" />
        <h1 className="text-xl font-extrabold tracking-tight">
          <span className="text-cyan-400">SAT</span>
          <span className="text-ink">QUERY</span>
        </h1>
      </div>
      <p className="mt-0.5 text-[13px] text-ink-soft">Intelligent Satellite Imagery Analysis</p>
    </div>
  )
}

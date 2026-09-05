import {
  Boxes,
  Clock,
  Cpu,
  GitCompareArrows,
  Image as ImageIcon,
  Layers,
  MapPinned,
  ScanEye,
  ShieldCheck,
  Target,
  Zap,
} from "lucide-react"
import EarthGlobe from "./EarthGlobe"
import { CAPABILITIES, HERO_FEATURES, STATS } from "../lib/constants"

const ICONS = {
  ScanEye,
  GitCompareArrows,
  MapPinned,
  Image: ImageIcon,
  Layers,
  Clock,
  Boxes,
  Cpu,
  ShieldCheck,
  Target,
  Zap,
}

const COLOR_STYLES = {
  blue: { bg: "bg-blue-500/15", border: "border-blue-500/30", text: "text-blue-400" },
  purple: { bg: "bg-purple-500/15", border: "border-purple-500/30", text: "text-purple-400" },
  teal: { bg: "bg-teal-500/15", border: "border-teal-500/30", text: "text-teal-400" },
  orange: { bg: "bg-orange-500/15", border: "border-orange-500/30", text: "text-orange-400" },
}

export default function HeroLanding() {
  return (
    <div className="flex flex-col gap-6 p-5 sm:p-8">
      <section className="bg-stars relative overflow-hidden rounded-2xl border border-line">
        <div className="absolute inset-0">
          <EarthGlobe className="absolute -right-10 -top-16 h-[130%] w-[75%] opacity-90 sm:w-[60%]" />
        </div>

        <div className="relative z-10 max-w-2xl px-6 py-14 sm:px-10 sm:py-20">
          <h2 className="text-4xl font-extrabold leading-tight tracking-tight text-white sm:text-5xl">
            Transform Satellite Data into{" "}
            <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
              Intelligence
            </span>
          </h2>
          <p className="mt-5 text-[15px] leading-relaxed text-ink-soft sm:text-base">
            Advanced multi-modal AI for scene understanding, change detection, and geospatial
            insights.
          </p>

          <div className="mt-8 h-px w-14 bg-gradient-to-r from-cyan-400 to-purple-400" />

          <div className="mt-6 flex flex-wrap gap-x-8 gap-y-4">
            {HERO_FEATURES.map((f) => {
              const Icon = ICONS[f.icon]
              return (
                <div key={f.label} className="flex items-center gap-2.5">
                  <span className="flex h-9 w-9 items-center justify-center rounded-full border border-line-soft bg-white/5">
                    <Icon className="h-4 w-4 text-cyan-300" strokeWidth={1.75} />
                  </span>
                  <span className="text-[13px] font-medium text-ink">{f.label}</span>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-line-soft bg-panel-soft p-5 sm:p-7">
        <p className="mb-5 text-xs font-bold uppercase tracking-[0.14em] text-ink-faint">
          Key Capabilities
        </p>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {CAPABILITIES.map((c) => {
            const Icon = ICONS[c.icon]
            const style = COLOR_STYLES[c.color]
            return (
              <div key={c.title} className="rounded-xl border border-line bg-panel p-5">
                <span
                  className={`flex h-11 w-11 items-center justify-center rounded-full border ${style.bg} ${style.border}`}
                >
                  <Icon className={`h-5 w-5 ${style.text}`} strokeWidth={1.75} />
                </span>
                <p className="mt-4 text-[15px] font-semibold text-ink">{c.title}</p>
                <p className="mt-1.5 text-[13px] leading-relaxed text-ink-soft">{c.description}</p>
              </div>
            )
          })}
        </div>
      </section>

      <section className="grid grid-cols-1 gap-5 rounded-2xl border border-line-soft bg-panel-soft p-5 sm:grid-cols-2 sm:p-6 lg:grid-cols-4">
        {STATS.map((s, i) => {
          const Icon = ICONS[s.icon]
          return (
            <div
              key={s.label}
              className={`flex items-center gap-3 ${i > 0 ? "sm:border-l sm:border-line-soft sm:pl-5" : ""}`}
            >
              <Icon className="h-5 w-5 shrink-0 text-ink-faint" strokeWidth={1.75} />
              <div className="min-w-0">
                <p className="text-[10.5px] font-bold uppercase tracking-wide text-ink-faint">
                  {s.label}
                </p>
                <p className="text-[12.5px] leading-snug text-ink-soft">{s.value}</p>
              </div>
            </div>
          )
        })}
      </section>
    </div>
  )
}

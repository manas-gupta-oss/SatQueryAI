import { Loader2, Orbit, Send } from "lucide-react"
import Logo from "./Logo"
import DemoModeToggle from "./DemoModeToggle"
import ImageUploader from "./ImageUploader"
import QueryInput from "./QueryInput"
import ExampleQueries from "./ExampleQueries"

function SectionHeading({ title, subtitle }) {
  return (
    <div className="flex gap-2.5">
      <span className="mt-0.5 w-1 shrink-0 rounded-full bg-gradient-to-b from-cyan-400 to-purple-500" />
      <div>
        <h3 className="text-[13px] font-bold uppercase tracking-wide text-ink">{title}</h3>
        {subtitle && <p className="mt-0.5 text-[11.5px] text-ink-faint">{subtitle}</p>}
      </div>
    </div>
  )
}

export default function Sidebar({ sq }) {
  const isBusy = sq.phase === "analyzing" || sq.phase === "settling"

  return (
    <aside className="flex w-full flex-col gap-5 border-b border-line bg-bg-elevated/60 p-5 lg:h-full lg:w-[380px] lg:shrink-0 lg:overflow-y-auto lg:border-b-0 lg:border-r">
      <div className="flex flex-col gap-3">
        <Logo />
        <div className="flex items-center justify-between gap-3 border-t border-line-soft pt-3">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-line-soft bg-gradient-to-r from-cyan-500/10 via-blue-500/10 to-purple-500/10 px-3 py-1 text-[11px] font-medium text-ink-soft">
            <Orbit className="h-3 w-3 text-cyan-400" strokeWidth={2} />
            Autonomous AI Routing
          </span>
          <DemoModeToggle enabled={sq.demoMode} onToggle={sq.setDemoMode} />
        </div>
      </div>

      <div className="h-px bg-line" />

      <div className="space-y-3">
        <SectionHeading title="Upload Imagery" subtitle="Upload satellite images to begin analysis" />
        <ImageUploader
          images={sq.images}
          onAddFiles={sq.addFiles}
          onRemove={sq.removeImage}
          onMetaChange={sq.updateImageMeta}
        />
      </div>

      <div className="h-px bg-line" />

      <div className="space-y-3">
        <SectionHeading title="Ask Your Question" subtitle="Describe your analysis need in natural language" />
        <QueryInput
          query={sq.query}
          onQueryChange={sq.setQuery}
          imageCount={sq.images.length}
          coRegistered={sq.coRegistered}
          onCoRegisteredChange={sq.setCoRegistered}
          sameLocation={sq.sameLocation}
          onSameLocationChange={sq.setSameLocation}
          validationError={sq.validationError}
        />
      </div>

      <div className="h-px bg-line" />

      <div className="space-y-3">
        <SectionHeading title="Try These Examples" />
        <ExampleQueries onPick={sq.setQuery} />
      </div>

      <div className="mt-auto flex flex-col gap-2 pt-2">
        <button
          type="button"
          onClick={sq.run}
          disabled={isBusy}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-500 to-purple-500 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-purple-500/20 transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isBusy ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Send className="h-4 w-4" />
              Run SatQuery
            </>
          )}
        </button>
        <p className="text-center text-[11px] text-ink-faint">
          Powered by Qwen2-VL-2B vision-language AI
        </p>
      </div>
    </aside>
  )
}

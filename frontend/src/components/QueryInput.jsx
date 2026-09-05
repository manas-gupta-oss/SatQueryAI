const MAX_QUERY_LENGTH = 500

function TriToggle({ label, value, onChange }) {
  const options = [
    { v: null, label: "Unknown" },
    { v: true, label: "Yes" },
    { v: false, label: "No" },
  ]
  return (
    <div className="flex items-center justify-between gap-2 text-[11px]">
      <span className="text-ink-soft">{label}</span>
      <div className="flex overflow-hidden rounded-lg border border-line-soft">
        {options.map((opt) => (
          <button
            key={String(opt.v)}
            type="button"
            onClick={() => onChange(opt.v)}
            className={`px-2 py-1 font-medium transition-colors ${
              value === opt.v ? "bg-cyan-500/15 text-cyan-300" : "text-ink-faint hover:text-ink-soft"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function QueryInput({
  query,
  onQueryChange,
  imageCount,
  coRegistered,
  onCoRegisteredChange,
  sameLocation,
  onSameLocationChange,
  validationError,
}) {
  return (
    <div className="space-y-3">
      <div className="relative">
        <textarea
          value={query}
          onChange={(e) => onQueryChange(e.target.value.slice(0, MAX_QUERY_LENGTH))}
          placeholder="Ask something about the satellite imagery..."
          rows={4}
          maxLength={MAX_QUERY_LENGTH}
          className="w-full resize-none rounded-xl border border-line-soft bg-panel px-3.5 py-3 pb-6 text-sm text-ink placeholder:text-ink-faint outline-none focus:border-cyan-500/40"
        />
        <span className="pointer-events-none absolute bottom-2 right-3 text-[10.5px] text-ink-faint">
          {query.length}/{MAX_QUERY_LENGTH}
        </span>
      </div>

      {imageCount === 2 && (
        <div className="space-y-1.5 rounded-xl border border-line-soft bg-panel p-2.5">
          <TriToggle label="Images co-registered" value={coRegistered} onChange={onCoRegisteredChange} />
          <TriToggle label="Same location" value={sameLocation} onChange={onSameLocationChange} />
        </div>
      )}

      {validationError && <p className="text-[11px] text-warn">{validationError}</p>}
    </div>
  )
}

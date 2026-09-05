export default function DemoModeToggle({ enabled, onToggle }) {
  return (
    <button
      type="button"
      onClick={() => onToggle(!enabled)}
      className="group flex shrink-0 items-center gap-2"
      title="Simulate the UI flow without calling the real backend"
    >
      <span className="text-[11px] font-medium text-ink-soft group-hover:text-ink">Demo</span>
      <span
        className={`relative h-5 w-9 shrink-0 rounded-full transition-colors duration-200 ${
          enabled ? "bg-gradient-to-r from-amber-400 to-orange-500" : "bg-line-soft"
        }`}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform duration-200 ${
            enabled ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </span>
    </button>
  )
}

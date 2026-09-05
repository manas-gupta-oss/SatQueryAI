import { WifiOff } from "lucide-react"

export default function ErrorBanner({ message }) {
  if (!message) return null
  return (
    <div className="flex w-full max-w-4xl items-center gap-3 rounded-xl border border-error/40 bg-error/10 px-4 py-3">
      <WifiOff className="h-4.5 w-4.5 shrink-0 text-error" />
      <p className="text-[13px] text-ink">{message}</p>
    </div>
  )
}

import { AlertTriangle, Loader2, X } from "lucide-react"
import { MODALITY_OPTIONS } from "../lib/constants"

export default function ImageThumbnail({ image, onRemove, onMetaChange, showRoleHint }) {
  return (
    <div className="group relative overflow-hidden rounded-lg border border-line bg-panel">
      <div className="relative aspect-video w-full overflow-hidden bg-bg">
        <img
          src={image.previewUrl}
          alt={image.file.name}
          className="h-full w-full object-cover"
        />
        {image.status === "uploading" && (
          <div className="absolute inset-0 flex items-center justify-center bg-bg/70">
            <Loader2 className="h-5 w-5 animate-spin text-accent" />
          </div>
        )}
        {image.status === "error" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-bg/85 px-2 text-center">
            <AlertTriangle className="h-4 w-4 text-error" />
            <span className="text-[10px] text-error">{image.errorMessage || "Upload failed"}</span>
          </div>
        )}
        <button
          type="button"
          onClick={() => onRemove(image.clientId)}
          className="absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-bg/80 text-ink-soft opacity-0 transition-opacity hover:text-error group-hover:opacity-100"
          aria-label="Remove image"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="space-y-1.5 p-2">
        <p className="truncate text-[11px] font-mono text-ink-soft" title={image.file.name}>
          {image.file.name}
        </p>
        <div className="flex items-center gap-1.5">
          <select
            value={image.modality}
            onChange={(e) => onMetaChange(image.clientId, { modality: e.target.value })}
            className="w-full rounded border border-line bg-bg-elevated px-1.5 py-1 text-[10px] text-ink-soft outline-none focus:border-accent/50"
          >
            {MODALITY_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          {showRoleHint && (
            <select
              value={image.roleHint}
              onChange={(e) => onMetaChange(image.clientId, { roleHint: e.target.value })}
              className="w-full rounded border border-line bg-bg-elevated px-1.5 py-1 text-[10px] text-ink-soft outline-none focus:border-accent/50"
            >
              <option value="">pair role</option>
              <option value="pre">pre</option>
              <option value="post">post</option>
            </select>
          )}
        </div>
        {(image.width || image.timestamp) && (
          <p className="truncate text-[10px] text-ink-faint">
            {image.width ? `${image.width}x${image.height}` : ""}
            {image.width && image.format ? " . " : ""}
            {image.format ? image.format.toUpperCase() : ""}
          </p>
        )}
      </div>
    </div>
  )
}

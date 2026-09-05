import { useRef, useState } from "react"
import { UploadCloud } from "lucide-react"
import ImageThumbnail from "./ImageThumbnail"

export default function ImageUploader({ images, onAddFiles, onRemove, onMetaChange }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files?.length) onAddFiles(e.dataTransfer.files)
  }

  return (
    <div className="space-y-3">
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`flex w-full flex-col items-center gap-2.5 rounded-2xl border-2 border-dashed px-4 py-8 text-center transition-colors ${
          dragOver
            ? "border-cyan-400/60 bg-cyan-500/5"
            : "border-line-soft hover:border-line hover:bg-panel-soft"
        }`}
      >
        <span className="flex h-11 w-11 items-center justify-center rounded-full bg-cyan-500/10">
          <UploadCloud className="h-5 w-5 text-cyan-400" strokeWidth={1.75} />
        </span>
        <span className="text-[13px] font-semibold text-ink">Drag &amp; drop your satellite image</span>
        <span className="text-[12px] text-ink-soft">or click to browse</span>
        <span className="text-[10.5px] text-ink-faint">GeoTIFF, TIFF, PNG, JPEG . Max 50MB</span>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          className="hidden"
          onChange={(e) => {
            onAddFiles(e.target.files)
            e.target.value = ""
          }}
        />
      </button>

      {images.length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {images.map((img) => (
            <ImageThumbnail
              key={img.clientId}
              image={img}
              onRemove={onRemove}
              onMetaChange={onMetaChange}
              showRoleHint={images.length === 2}
            />
          ))}
        </div>
      )}
    </div>
  )
}

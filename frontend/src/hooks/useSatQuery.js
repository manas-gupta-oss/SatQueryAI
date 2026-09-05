import { useCallback, useMemo, useRef, useState } from "react"
import { ApiError, runQuery as apiRunQuery, uploadImage } from "../api/satqueryApi"
import { buildDemoResponse } from "../demo/demoScenarios"
import { MAX_UPLOAD_BYTES } from "../lib/constants"

let idCounter = 0
const nextClientId = () => `local_${Date.now().toString(36)}_${idCounter++}`

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function makeImage(file) {
  const tooLarge = file.size > MAX_UPLOAD_BYTES
  return {
    clientId: nextClientId(),
    file,
    previewUrl: URL.createObjectURL(file),
    status: tooLarge ? "error" : "uploading", // 'uploading' | 'ready' | 'error'
    errorMessage: tooLarge ? "File exceeds the 50MB limit." : null,
    serverId: null,
    modality: "optical",
    sensor: "",
    timestamp: "",
    roleHint: "",
    width: null,
    height: null,
    format: null,
  }
}

export function useSatQuery() {
  const [images, setImages] = useState([])
  const [query, setQuery] = useState("")
  const [coRegistered, setCoRegistered] = useState(null) // null | true | false
  const [sameLocation, setSameLocation] = useState(null)
  const [demoMode, setDemoMode] = useState(false)

  // 'idle' | 'analyzing' | 'settling' | 'result'
  const [phase, setPhase] = useState("idle")
  const [result, setResult] = useState(null)
  const [connectionError, setConnectionError] = useState(null)
  const [validationError, setValidationError] = useState(null)

  const runToken = useRef(0)

  const doUpload = useCallback(async (clientId, file, meta) => {
    try {
      const res = await uploadImage(file, meta)
      setImages((prev) =>
        prev.map((img) =>
          img.clientId === clientId
            ? {
                ...img,
                status: "ready",
                errorMessage: null,
                serverId: res.image_id,
                width: res.width,
                height: res.height,
                format: res.format,
              }
            : img
        )
      )
    } catch (err) {
      setImages((prev) =>
        prev.map((img) =>
          img.clientId === clientId
            ? {
                ...img,
                status: "error",
                errorMessage: err instanceof ApiError ? err.message : "Upload failed.",
              }
            : img
        )
      )
    }
  }, [])

  const addFiles = useCallback(
    (fileList) => {
      const files = Array.from(fileList || [])
      if (!files.length) return
      const newImages = files.map(makeImage)
      setImages((prev) => [...prev, ...newImages])
      newImages.forEach((img) => {
        if (img.status === "error") return // oversized, already flagged in makeImage
        doUpload(img.clientId, img.file, {
          modality: img.modality,
          sensor: img.sensor,
          timestamp: img.timestamp,
          roleHint: img.roleHint,
        })
      })
    },
    [doUpload]
  )

  const removeImage = useCallback((clientId) => {
    setImages((prev) => {
      const target = prev.find((img) => img.clientId === clientId)
      if (target) URL.revokeObjectURL(target.previewUrl)
      return prev.filter((img) => img.clientId !== clientId)
    })
  }, [])

  const updateImageMeta = useCallback(
    (clientId, patch) => {
      setImages((prev) =>
        prev.map((img) => (img.clientId === clientId ? { ...img, ...patch } : img))
      )
      const target = images.find((img) => img.clientId === clientId)
      if (!target) return
      const merged = { ...target, ...patch }
      setImages((prev) =>
        prev.map((img) => (img.clientId === clientId ? { ...img, status: "uploading" } : img))
      )
      doUpload(clientId, target.file, {
        modality: merged.modality,
        sensor: merged.sensor,
        timestamp: merged.timestamp,
        roleHint: merged.roleHint,
      })
    },
    [images, doUpload]
  )

  const readyImages = useMemo(() => images.filter((img) => img.status === "ready"), [images])
  const hasPendingUploads = useMemo(
    () => images.some((img) => img.status === "uploading"),
    [images]
  )

  const resetResult = useCallback(() => {
    setPhase("idle")
    setResult(null)
    setConnectionError(null)
    setValidationError(null)
  }, [])

  const settleWithResponse = useCallback(async (response, settleDelayMs, token) => {
    setResult(response)
    setPhase("settling")
    await delay(settleDelayMs)
    if (runToken.current !== token) return
    setPhase("result")
  }, [])

  const run = useCallback(async () => {
    setValidationError(null)
    setConnectionError(null)

    const trimmedQuery = query.trim()
    if (!trimmedQuery) {
      setValidationError("Enter a query before running SatQuery.")
      return
    }
    if (images.length === 0) {
      setValidationError("Upload at least one satellite image first.")
      return
    }
    if (hasPendingUploads) {
      setValidationError("Wait for image uploads to finish.")
      return
    }
    if (readyImages.length !== images.length) {
      setValidationError("Remove or retry the failed image upload before running SatQuery.")
      return
    }

    const token = ++runToken.current
    setResult(null)
    setPhase("analyzing")

    if (demoMode) {
      await delay(900)
      if (runToken.current !== token) return
      const response = buildDemoResponse(trimmedQuery, images.length)
      await settleWithResponse(response, 700, token)
      return
    }

    try {
      const response = await apiRunQuery({
        query: trimmedQuery,
        imageIds: readyImages.map((img) => img.serverId),
        coRegistered: images.length >= 2 ? coRegistered : null,
        sameLocation: images.length >= 2 ? sameLocation : null,
      })
      if (runToken.current !== token) return
      await settleWithResponse(response, 550, token)
    } catch (err) {
      if (runToken.current !== token) return
      if (err instanceof ApiError && err.isNetworkError) {
        setConnectionError(err.message)
        setPhase("idle")
      } else {
        const message = err instanceof ApiError ? err.message : "Unexpected error."
        setResult({
          status: "error",
          error: message,
          worker: null,
          task_type: null,
          answer: "",
          confidence: null,
          router_confidence: null,
          validation: null,
          assumptions: [],
          audit_summary: "",
          execution_trace: [],
        })
        setPhase("result")
      }
    }
  }, [
    query,
    images,
    readyImages,
    hasPendingUploads,
    demoMode,
    coRegistered,
    sameLocation,
    settleWithResponse,
  ])

  return {
    images,
    addFiles,
    removeImage,
    updateImageMeta,
    query,
    setQuery,
    coRegistered,
    setCoRegistered,
    sameLocation,
    setSameLocation,
    demoMode,
    setDemoMode,
    phase,
    result,
    connectionError,
    validationError,
    run,
    resetResult,
    readyImages,
    hasPendingUploads,
  }
}

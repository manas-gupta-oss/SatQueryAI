// Thin fetch wrapper around the SatQueryAI backend. Nothing UI-specific lives
// here so components/hooks never construct requests by hand.
//
// Expected backend (backend/app.py in the SatqueryAI repo):
//   GET  /api/health
//   GET  /api/workers
//   POST /api/upload   multipart/form-data
//   POST /api/query    application/json

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(
  /\/+$/,
  ""
)

export class ApiError extends Error {
  constructor(message, { status = null, detail = null, cause } = {}) {
    super(message)
    this.name = "ApiError"
    this.status = status
    this.detail = detail
    this.isNetworkError = status === null
    if (cause) this.cause = cause
  }
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(`${API_BASE_URL}${path}`, options)
  } catch (err) {
    throw new ApiError("Unable to connect to SatQuery backend.", { cause: err })
  }

  if (!res.ok) {
    let detail = null
    try {
      const body = await res.json()
      detail = body.detail || body.error || JSON.stringify(body)
    } catch {
      detail = res.statusText
    }
    throw new ApiError(
      typeof detail === "string" ? detail : "The backend rejected the request.",
      { status: res.status, detail }
    )
  }

  if (res.status === 204) return null
  return res.json()
}

export function getHealth() {
  return request("/api/health")
}

export function getWorkers() {
  return request("/api/workers")
}

export function uploadImage(file, meta = {}) {
  const form = new FormData()
  form.append("file", file)
  if (meta.modality) form.append("modality", meta.modality)
  if (meta.sensor) form.append("sensor", meta.sensor)
  if (meta.timestamp) form.append("timestamp", meta.timestamp)
  if (meta.roleHint) form.append("role_hint", meta.roleHint)

  return request("/api/upload", { method: "POST", body: form })
}

export function runQuery({ query, imageIds, coRegistered = null, sameLocation = null }) {
  return request("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      image_ids: imageIds,
      co_registered: coRegistered,
      same_location: sameLocation,
    }),
  })
}

export function resolveAssetUrl(url) {
  if (!url) return url
  return url.startsWith("http") ? url : `${API_BASE_URL}${url}`
}

export { API_BASE_URL }

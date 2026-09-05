// Static fallback for /api/workers so the architecture panel still renders
// something truthful if the backend is unreachable. Mirrors
// orchestration/tool_schema.WORKER_REGISTRY at the time this UI was built;
// the live endpoint is always preferred when reachable (see useSatQuery).
export const FALLBACK_WORKERS = [
  {
    id: "worker1",
    display_name: "Worker 1 - Single-Image Specialist",
    short_name: "Single Image Worker",
    description:
      "Single optical / multispectral image understanding: visual question answering, scene captioning, and text-guided region grounding.",
    tuned_on: "VRSBench",
    tasks: ["captioning", "grounding", "single_vqa"],
    min_images: 1,
    max_images: 1,
  },
  {
    id: "worker2",
    display_name: "Worker 2 - Cross-Modal Optical+SAR Specialist",
    short_name: "Cross-Modal Worker",
    description:
      "Joint reasoning over a co-registered optical/multispectral and SAR pair, plus question answering on a standalone SAR image.",
    tuned_on: "BigthNet.txt",
    tasks: ["cross_modal_fusion", "single_vqa"],
    min_images: 1,
    max_images: 2,
  },
  {
    id: "worker3",
    display_name: "Worker 3 - Bi-Temporal Change Specialist",
    short_name: "Bi-Temporal Worker",
    description:
      "Change understanding over two spatially corresponding images of the same area acquired at different times.",
    tuned_on: "CDVQA",
    tasks: ["change_description", "change_vqa"],
    min_images: 2,
    max_images: 2,
  },
]

export const WORKER_SHORT_NAMES = {
  worker1: "Single Image Worker",
  worker2: "Cross-Modal Worker",
  worker3: "Bi-Temporal Worker",
}

export const WORKER_ACCENTS = {
  worker1: {
    text: "text-worker1",
    border: "border-worker1",
    bg: "bg-worker1-soft",
    glow: "shadow-[0_0_0_1px_var(--color-worker1),0_0_24px_-4px_var(--color-worker1)]",
    dot: "bg-worker1",
  },
  worker2: {
    text: "text-worker2",
    border: "border-worker2",
    bg: "bg-worker2-soft",
    glow: "shadow-[0_0_0_1px_var(--color-worker2),0_0_24px_-4px_var(--color-worker2)]",
    dot: "bg-worker2",
  },
  worker3: {
    text: "text-worker3",
    border: "border-worker3",
    bg: "bg-worker3-soft",
    glow: "shadow-[0_0_0_1px_var(--color-worker3),0_0_24px_-4px_var(--color-worker3)]",
    dot: "bg-worker3",
  },
}

export const TASK_TYPE_LABELS = {
  single_vqa: "Single-Image VQA",
  captioning: "Scene Captioning",
  grounding: "Text-Guided Grounding",
  cross_modal_fusion: "Cross-Modal Fusion",
  change_vqa: "Change VQA",
  change_description: "Change Description",
}

export const EXAMPLE_QUERIES = [
  { text: "What is visible in this image?", icon: "Search" },
  { text: "Compare these two satellite images.", icon: "GitCompareArrows" },
  { text: "What changed between these images?", icon: "RefreshCw" },
  { text: "Identify the land-use pattern.", icon: "Map" },
  { text: "Analyze the temporal change in this region.", icon: "Clock" },
]

export const MODALITY_OPTIONS = [
  { value: "optical", label: "Optical" },
  { value: "multispectral", label: "Multispectral" },
  { value: "sar", label: "SAR" },
  { value: "unknown", label: "Unknown" },
]

export const SYSTEM_PANEL = {
  orchestration: "LangGraph",
  router: "Qwen 2.5 - 3B (4-bit, function-calling)",
  engine: "Qwen2-VL-2B (shared vision-language backbone)",
}

export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024 // 50MB, enforced client-side

// Landing-page copy. Grounded in the real three-worker system, not invented
// marketing claims - each line maps to an actual worker or the finalize node.
export const HERO_FEATURES = [
  { icon: "ScanEye", label: "Multi-Modal Understanding" },
  { icon: "GitCompareArrows", label: "Change Detection & Analysis" },
  { icon: "MapPinned", label: "Geospatial Insights" },
]

export const CAPABILITIES = [
  {
    icon: "Image",
    color: "blue",
    title: "Image Understanding",
    description:
      "Extract meaningful information from single satellite images - captioning, VQA, and region grounding.",
  },
  {
    icon: "Layers",
    color: "purple",
    title: "Cross-Modal Analysis",
    description:
      "Joint reasoning over optical, multispectral, and SAR data for deeper insight than either alone.",
  },
  {
    icon: "Clock",
    color: "teal",
    title: "Temporal Analysis",
    description: "Detect and analyze changes across time for the same geographical region.",
  },
  {
    icon: "Boxes",
    color: "orange",
    title: "Comprehensive Insights",
    description: "Generate actionable insights with detailed explanations and an auditable execution trace.",
  },
]

export const STATS = [
  { icon: "Cpu", label: "AI Engine", value: "Router + vision-language model" },
  { icon: "ShieldCheck", label: "Data Handling", value: "In-memory, not persisted" },
  { icon: "Target", label: "Validation", value: "Deterministic compatibility gate" },
  { icon: "Zap", label: "Response", value: "Single-pass routing" },
]

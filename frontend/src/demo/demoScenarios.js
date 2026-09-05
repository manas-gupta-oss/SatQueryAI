// Demo Mode: a scripted, clearly-labeled simulation of the real flow, for
// presenting without a live backend/GPU. Never mixed into the real request
// path - see useSatQuery.runQuery vs useSatQuery.runDemoQuery.
//
// Each scenario is shaped like the real backend's final_response (see
// orchestration/nodes/terminal_nodes.finalize_node) so ResultPanel needs no
// special-casing beyond a "_demo: true" banner.

const CHANGE_WORDS = /\b(change[ds]?|differ|compar|between|increase|decrease)\b/i
const SAR_WORDS = /\b(sar|radar|fusion|multispectral)\b/i
const LOCATE_WORDS = /\b(highlight|locate|where is|find the)\b/i

function pickWorkerId(query, imageCount) {
  if (imageCount >= 2 && (CHANGE_WORDS.test(query) || imageCount === 2)) return "worker3"
  if (SAR_WORDS.test(query)) return "worker2"
  return "worker1"
}

const SCENARIOS = {
  worker1: {
    task_type: "captioning",
    build: (query) => ({
      status: "ok",
      task_type: LOCATE_WORDS.test(query) ? "grounding" : "captioning",
      worker: "worker1",
      answer:
        "The scene shows a mixed land-use area: dense built-up structures in the upper-left " +
        "quadrant, a network of paved roads, scattered vegetation patches, and an open water " +
        "body along the lower edge of the frame. Rooftop density suggests a suburban settlement " +
        "pattern bordering cultivated land.",
      confidence: 0.88,
      router_confidence: 0.91,
      validation: { status: "pass", code: "ok", reason: "" },
      assumptions: [],
      audit_summary:
        "Single optical image, descriptive query -> captioning routed to worker1 (VRSBench-tuned single-image specialist).",
      execution_trace: [
        "Requested task: captioning",
        "Selected tool: worker1 (Worker 1 - Single-Image Specialist, adapted on VRSBench)",
        "Validation: pass",
        "Router confidence: 0.91",
        "[boss] (mock) task=captioning worker=worker1 status=pass",
        "[worker1] (mock) scene described",
      ],
    }),
  },
  worker2: {
    build: () => ({
      status: "ok",
      task_type: "cross_modal_fusion",
      worker: "worker2",
      answer:
        "Fusing the optical and SAR channels resolves surfaces that are ambiguous in either " +
        "modality alone: the optical band confirms spectral greenness consistent with cropland, " +
        "while the SAR backscatter highlights a compact, high-return cluster consistent with " +
        "built-up structures near the frame center. Combined, built-up area covers roughly 18% " +
        "of the scene and water bodies roughly 6%.",
      confidence: 0.82,
      router_confidence: 0.89,
      validation: { status: "pass", code: "ok", reason: "" },
      assumptions: [],
      audit_summary:
        "Co-registered optical + SAR pair, joint-analysis query -> cross_modal_fusion routed to worker2 (BigEarthNet.txt-tuned).",
      execution_trace: [
        "Requested task: cross_modal_fusion",
        "Selected tool: worker2 (Worker 2 - Cross-Modal Optical+SAR Specialist, adapted on BigEarthNet.txt)",
        "Image roles: {'optical': 'demo_img_0', 'sar': 'demo_img_1'}",
        "Validation: pass",
        "Router confidence: 0.89",
        "[boss] (mock) task=cross_modal_fusion worker=worker2 status=pass",
        "[worker2] (mock) fusion complete",
      ],
    }),
  },
  worker3: {
    build: (query) => ({
      status: "ok",
      task_type: query.trim().endsWith("?") ? "change_vqa" : "change_description",
      worker: "worker3",
      answer:
        "Between the two acquisition dates, built-up area expanded outward from the existing " +
        "settlement core, primarily replacing agricultural land along the eastern edge of the " +
        "scene. Water extent along the southern boundary decreased slightly, consistent with a " +
        "seasonal drawdown rather than a permanent change.",
      confidence: 0.79,
      router_confidence: 0.85,
      validation: { status: "pass", code: "ok", reason: "" },
      assumptions: [
        "co-registration was not verified by the uploader; assumed aligned (demo simulation)",
      ],
      audit_summary:
        "Two co-registered optical images of the same area at different acquisition dates -> change analysis routed to worker3 (CDVQA-tuned bi-temporal specialist).",
      execution_trace: [
        "Requested task: change_description",
        "Selected tool: worker3 (Worker 3 - Bi-Temporal Change Specialist, adapted on CDVQA)",
        "Image roles: {'pre': 'demo_img_0', 'post': 'demo_img_1'}",
        "Validation: pass",
        "Router confidence: 0.85",
        "Assumption: co-registration was not verified by the uploader; assumed aligned (demo simulation)",
        "[boss] (mock) task=change_description worker=worker3 status=pass",
        "[worker3] (mock) change analysis complete",
      ],
    }),
  },
}

export function buildDemoResponse(query, imageCount) {
  const workerId = pickWorkerId(query || "Describe this scene.", imageCount)
  const scenario = SCENARIOS[workerId]
  const response = scenario.build(query || "Describe this scene.")
  return {
    ...response,
    request_id: `demo_${Date.now().toString(36)}`,
    query: query || "",
    error: null,
    visual_evidence: { boxes: [], box_labels: [], mask_path: null, overlay_path: null },
    _demo: true,
  }
}

export function demoWorkerFor(query, imageCount) {
  return pickWorkerId(query || "", imageCount)
}

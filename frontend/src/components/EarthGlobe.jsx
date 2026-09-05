// Decorative CSS/SVG rendition of an Earth limb at night, used only as hero
// background texture. No external image assets - built from gradients + a
// deterministically-seeded scatter of "city light" points along the rim.

const CX = 640
const CY = 220
const R = 430

// Small seeded PRNG so the point scatter is fixed across renders/builds
// without hand-typing a coordinate list.
function mulberry32(seed) {
  let a = seed
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function buildCityLights() {
  const rand = mulberry32(7)
  const points = []

  // Scattered lights along the visible arc.
  for (let i = 0; i < 70; i++) {
    const angle = (140 + rand() * 150) * (Math.PI / 180)
    const radius = R - 4 - rand() * 26
    const cx = CX + Math.cos(angle) * radius
    const cy = CY + Math.sin(angle) * radius
    points.push({
      cx,
      cy,
      r: 0.6 + rand() * 1.4,
      opacity: 0.25 + rand() * 0.45,
      color: rand() > 0.3 ? "#fbbf24" : "#f97316",
    })
  }

  // A brighter hotspot cluster (matches a dense metro region).
  const hotAngle = 205 * (Math.PI / 180)
  const hotX = CX + Math.cos(hotAngle) * (R - 14)
  const hotY = CY + Math.sin(hotAngle) * (R - 14)
  for (let i = 0; i < 26; i++) {
    points.push({
      cx: hotX + (rand() - 0.5) * 46,
      cy: hotY + (rand() - 0.5) * 34,
      r: 0.6 + rand() * 1.6,
      opacity: 0.5 + rand() * 0.5,
      color: rand() > 0.4 ? "#fbbf24" : "#fde68a",
    })
  }

  return points
}

const CITY_LIGHTS = buildCityLights()

export default function EarthGlobe({ className = "" }) {
  return (
    <svg
      viewBox="0 0 900 700"
      className={className}
      preserveAspectRatio="xMaxYMid slice"
      aria-hidden="true"
    >
      <defs>
        <radialGradient id="planetBody" cx="35%" cy="35%" r="75%">
          <stop offset="0%" stopColor="#0b1224" />
          <stop offset="55%" stopColor="#070b16" />
          <stop offset="100%" stopColor="#04060c" />
        </radialGradient>
        <linearGradient id="rimGlow" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.9" />
          <stop offset="50%" stopColor="#38bdf8" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#818cf8" stopOpacity="0.15" />
        </linearGradient>
      </defs>

      <circle
        cx={CX}
        cy={CY}
        r={R + 18}
        fill="none"
        stroke="#3b82f6"
        strokeWidth="34"
        opacity="0.12"
        style={{ filter: "blur(18px)" }}
      />

      <circle cx={CX} cy={CY} r={R} fill="url(#planetBody)" />
      <circle cx={CX} cy={CY} r={R} fill="none" stroke="url(#rimGlow)" strokeWidth="2.5" opacity="0.85" />

      <g>
        {CITY_LIGHTS.map((p, i) => (
          <circle key={i} cx={p.cx} cy={p.cy} r={p.r} fill={p.color} opacity={p.opacity} />
        ))}
      </g>
    </svg>
  )
}

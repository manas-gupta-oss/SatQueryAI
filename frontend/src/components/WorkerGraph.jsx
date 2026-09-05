import { Boxes } from "lucide-react"
import BossNode from "./BossNode"
import WorkerCard from "./WorkerCard"

function Connector({ flowing, colorClass = "bg-line-soft" }) {
  return (
    <div className="relative h-7 w-px shrink-0">
      <div className={`h-full w-full ${flowing ? colorClass : "bg-line-soft"}`} />
      {flowing && (
        <div
          className={`absolute left-1/2 top-0 h-2 w-2 -translate-x-1/2 rounded-full ${colorClass} animate-pulse-soft`}
        />
      )}
    </div>
  )
}

function statusForWorker(workerId, phase, result) {
  if (phase === "settling" && result?.worker === workerId) return "ACTIVE"
  if (phase === "result" && result?.status === "ok" && result?.worker === workerId) return "COMPLETED"
  return "IDLE"
}

export default function WorkerGraph({ workers, phase, result }) {
  const bossFlowing = phase === "analyzing"
  const routedOk = (phase === "settling" || phase === "result") && result?.status === "ok"

  return (
    <section className="flex flex-col items-center">
      <BossNode phase={phase} result={result} />

      <Connector flowing={bossFlowing} colorClass="bg-accent" />

      <div className="relative w-full max-w-4xl">
        <div className="absolute left-[16.6%] right-[16.6%] top-0 hidden h-px bg-line-soft sm:block" />
        <div className="grid grid-cols-1 gap-4 pt-4 sm:grid-cols-3 sm:gap-5">
          {workers.map((worker) => (
            <div key={worker.id} className="flex flex-col items-center gap-0">
              <span className="hidden h-4 w-px bg-line-soft sm:block" />
              <div className="w-full">
                <WorkerCard worker={worker} status={statusForWorker(worker.id, phase, result)} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <Connector flowing={routedOk} colorClass="bg-online" />

      <div
        className={`flex w-full max-w-md items-center gap-3 rounded-xl border bg-panel p-4 transition-colors ${
          routedOk ? "border-online/40" : "border-line"
        }`}
      >
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-line">
          <Boxes className="h-4.5 w-4.5 text-ink-soft" strokeWidth={1.75} />
        </div>
        <div className="min-w-0">
          <p className="text-[13px] font-semibold text-ink">Shared Vision-Language Engine</p>
          <p className="truncate text-[10.5px] font-mono text-ink-faint">Qwen2-VL-2B . common backbone for all workers</p>
        </div>
      </div>
    </section>
  )
}

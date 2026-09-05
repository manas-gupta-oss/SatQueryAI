import { useEffect, useState } from "react"
import Sidebar from "./components/Sidebar"
import HeroLanding from "./components/HeroLanding"
import WorkerGraph from "./components/WorkerGraph"
import ResultPanel from "./components/ResultPanel"
import ArchitecturePanel from "./components/ArchitecturePanel"
import ErrorBanner from "./components/ErrorBanner"
import { getWorkers } from "./api/satqueryApi"
import { FALLBACK_WORKERS } from "./lib/constants"
import { useSatQuery } from "./hooks/useSatQuery"

export default function App() {
  const sq = useSatQuery()
  const [workers, setWorkers] = useState(FALLBACK_WORKERS)

  useEffect(() => {
    let cancelled = false
    getWorkers()
      .then((res) => {
        if (!cancelled && Array.isArray(res) && res.length) setWorkers(res)
      })
      .catch(() => {
        // keep the static fallback; the UI still renders truthfully
      })
    return () => {
      cancelled = true
    }
  }, [])

  const showHero = sq.phase === "idle"

  return (
    <div className="flex min-h-screen flex-col bg-bg lg:flex-row">
      <Sidebar sq={sq} />

      <main className="bg-grid flex-1 overflow-y-auto">
        {showHero ? (
          <>
            {sq.connectionError && (
              <div className="px-5 pt-5 sm:px-8">
                <ErrorBanner message={sq.connectionError} />
              </div>
            )}
            <HeroLanding />
          </>
        ) : (
          <div className="mx-auto flex max-w-[1600px] flex-col items-center gap-6 px-5 py-8 sm:px-8">
            <WorkerGraph workers={workers} phase={sq.phase} result={sq.result} />

            <ErrorBanner message={sq.connectionError} />

            {sq.phase === "result" && <ResultPanel result={sq.result} images={sq.readyImages} />}

            <ArchitecturePanel workers={workers} />
          </div>
        )}
      </main>
    </div>
  )
}

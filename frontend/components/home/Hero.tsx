'use client'

import { useState, useEffect } from 'react'
import Button from '@/components/ui/Button'

const TAGLINE =
  'Search 145 years of Social Security data, chart the trends, and forecast where a name is headed next.'

const STATS = [
  { num: '1880—2024', label: 'Years of data' },
  { num: 'ARIMA', label: '5-year forecasts' },
  { num: 'AI', label: 'Natural-language chat' },
]

const HEADLINE_GRADIENT = {
  background:
    'linear-gradient(to bottom, #ffffff 0%, rgba(255,255,255,0.95) 40%, rgba(255,255,255,0.70) 100%)',
  WebkitBackgroundClip: 'text' as const,
  WebkitTextFillColor: 'transparent' as const,
  backgroundClip: 'text' as const,
}

export default function Hero() {
  const [typed, setTyped] = useState('')

  useEffect(() => {
    if (typed.length >= TAGLINE.length) return
    const t = setTimeout(() => setTyped(TAGLINE.slice(0, typed.length + 1)), 25)
    return () => clearTimeout(t)
  }, [typed])

  return (
    <section className="relative z-10 flex items-center overflow-hidden">
      <div className="relative mx-auto w-full max-w-7xl px-6 py-24 lg:py-32">
        <div className="max-w-3xl space-y-8">
          {/* Status badge */}
          <div className="flex items-center gap-3">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#0ea5e9] opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-[#0ea5e9]" />
            </span>
            <span className="font-mono text-[11px] tracking-widest text-[#8a8f98]">
              SSA dataset · updated yearly
            </span>
          </div>

          {/* Headline */}
          <div>
            <h1
              className="text-5xl leading-none font-semibold tracking-[-0.03em] md:text-7xl"
              style={HEADLINE_GRADIENT}
            >
              Baby Names Explorer
            </h1>
            <div className="mt-3 flex items-center gap-4">
              <span className="h-px w-12 bg-[#0ea5e9]/50" />
              <h2 className="font-mono text-lg tracking-widest text-[#0ea5e9] md:text-xl">
                Popularity, charted & forecast
              </h2>
            </div>
          </div>

          {/* Typewriter tagline */}
          <p className="min-h-[3.5rem] max-w-xl text-base leading-relaxed text-[#8a8f98] md:text-lg">
            <span className="sr-only">{TAGLINE}</span>
            <span aria-hidden="true">
              {typed}
              {typed.length < TAGLINE.length && (
                <span className="ml-0.5 inline-block h-4 w-0.5 translate-y-0.5 animate-[blink_1s_step-end_infinite] bg-[#0ea5e9] align-middle" />
              )}
            </span>
          </p>

          {/* CTAs */}
          <div className="flex flex-wrap items-center gap-4">
            <Button variant="primary" size="lg" href="/search">
              Search a Name
            </Button>
            <Button variant="secondary" size="lg" href="/chat">
              Ask the AI
            </Button>
          </div>

          {/* Stats */}
          <div className="flex flex-wrap gap-8 border-t border-white/[0.06] pt-6">
            {STATS.map(({ num, label }) => (
              <div key={label}>
                <div className="font-mono text-lg font-semibold text-[#ededef]">{num}</div>
                <div className="text-xs text-[#8a8f98]">{label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

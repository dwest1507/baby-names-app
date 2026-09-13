import { type ReactNode } from 'react'

const HEADLINE_GRADIENT = {
  background:
    'linear-gradient(to bottom, #ffffff 0%, rgba(255,255,255,0.95) 40%, rgba(255,255,255,0.70) 100%)',
  WebkitBackgroundClip: 'text' as const,
  WebkitTextFillColor: 'transparent' as const,
  backgroundClip: 'text' as const,
}

// A ledger rather than a row of headline figures: the value is the fact, the
// label says what it covers. The forecast is the pooled gradient-boosted model,
// not the per-name ARIMA fit it replaced — docs/adr/0010-*.md.
const LEDGER: { value: ReactNode; label: string }[] = [
  { value: '1880—2025', label: '145 years of SSA records' },
  { value: 'Pooled model', label: '5-year forecasts, conformal bands' },
  { value: 'Groq', label: 'Questions answered in plain English' },
]

export default function Hero() {
  return (
    <div className="space-y-7">
      {/* Eyebrow — the accent rule leads the line instead of splitting it */}
      <div className="flex items-center gap-3">
        <span className="h-px w-8 bg-[#0ea5e9]/50" />
        <span className="font-mono text-[11px] tracking-widest text-[#8a8f98]">
          Social Security Administration · updated yearly
        </span>
      </div>

      {/* Headline */}
      <h1
        className="text-4xl leading-[1.05] font-semibold tracking-[-0.03em] md:text-6xl"
        style={HEADLINE_GRADIENT}
      >
        Baby Names
        <br />
        Explorer
      </h1>

      <p className="max-w-md text-base leading-relaxed text-[#8a8f98]">
        A century and a half of naming, charted. Look up where a name has been, where the model
        thinks it is headed, or just ask the data a question.
      </p>

      {/* Ledger */}
      <dl className="max-w-md divide-y divide-white/[0.06] border-y border-white/[0.06]">
        {LEDGER.map(({ value, label }) => (
          <div key={label} className="flex items-baseline gap-4 py-2.5">
            <dt className="w-32 shrink-0 font-mono text-xs tracking-wider text-[#ededef]">
              {value}
            </dt>
            <dd className="text-xs text-[#8a8f98]">{label}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

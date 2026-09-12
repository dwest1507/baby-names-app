'use client'

import { useEffect, useState, type FormEvent } from 'react'
import Card from '@/components/ui/Card'
import Notice from '@/components/ui/Notice'
import Section from '@/components/layout/Section'
import SexToggle from '@/components/ui/SexToggle'
import Tag from '@/components/ui/Tag'
import TrendChart from '@/components/charts/TrendChart'
import {
  ApiError,
  getMeta,
  getNameForecast,
  getNameHistory,
  type ForecastPayload,
  type ForecastPoint,
  type NameRow,
} from '@/lib/api'
import { formatCount, formatPercent, formatRank } from '@/lib/format'

// Mirrors MIN_HISTORY_YEARS in backend/app/services/forecast.py: below this many
// recorded years the backend declines to fit a model.
const MIN_FORECAST_HISTORY_YEARS = 10

const inputClass =
  'h-11 w-full rounded-lg border border-white/[0.08] bg-white/[0.04] px-4 text-sm text-[#ededef] placeholder-[#8a8f98]/60 transition-all duration-150 focus:border-[#0ea5e9]/50 focus:bg-white/[0.06] focus:shadow-[0_0_0_3px_rgba(14,165,233,0.15)] focus:outline-none'

interface StatTileProps {
  label: string
  value: string
  delta?: number
  deltaLabel?: string
  comparisonYear?: number
  invertDelta?: boolean
}

function StatTile({
  label,
  value,
  delta,
  deltaLabel,
  comparisonYear,
  invertDelta = false,
}: StatTileProps) {
  const improving = delta !== undefined && (invertDelta ? delta < 0 : delta > 0)
  return (
    <Card variant="default" className="p-5">
      <div className="text-xs text-[#8a8f98]">{label}</div>
      <div className="mt-1 font-mono text-3xl font-semibold text-[#ededef]">{value}</div>
      {delta !== undefined && delta !== 0 && (
        <div
          className={`mt-1 inline-flex items-center gap-1 text-xs ${
            improving ? 'text-emerald-400' : 'text-red-400'
          }`}
        >
          <span aria-hidden="true">{improving ? '▲' : '▼'}</span>
          {deltaLabel}
        </div>
      )}
      {delta === 0 && comparisonYear !== undefined && (
        <div className="mt-1 text-xs text-[#8a8f98]">unchanged vs {comparisonYear}</div>
      )}
    </Card>
  )
}

// A pooled model has no per-name fit to report an order or residual
// diagnostics for. What *is* per-name is the stratum the name sits in, how
// wide the band it was given is, how well the model has actually done on it,
// and where it stands against its own peak — which is a feature the model
// reads. See docs/adr/0010-a-pooled-model-replaces-per-name-arima.md and
// docs/adr/0011-conformal-bands-keyed-by-strata.md.
const TIER_LABELS: Record<string, string> = {
  top100: 'Top 100',
  top1000: 'Top 1,000',
  top5000: 'Top 5,000',
  rest: 'Outside the top 5,000',
}

// The bins are tertiles of recent year-to-year log wobble, cut across the
// whole corpus at the origin — so "third" is literal.
const VOLATILITY_LABELS = ['Steadiest third', 'Middle third', 'Jumpiest third']

/** How much wider the top of the band is than its bottom, at the last year
 *  forecast — the horizon where a visitor's eye lands and the band is widest. */
function bandWidth(forecast: ForecastPoint[]): string | null {
  const last = forecast[forecast.length - 1]
  if (!last || last.lo95 <= 0) return null
  return `${(last.hi95 / last.lo95).toFixed(1)}\u00d7 at ${last.year}`
}

/** Where the latest recorded year stands against the name's own high-water
 *  mark. `below_peak` and `yrs_since_peak` are model features, so this says
 *  what the model is looking at rather than decorating the panel. */
function peakPosition(history: NameRow[]): string | null {
  const latest = history[history.length - 1]
  if (!latest) return null
  const peak = history.reduce((best, row) =>
    row.popularity_percent > best.popularity_percent ? row : best
  )
  if (peak.year === latest.year) return `At its peak (${latest.year})`
  const below = 1 - latest.popularity_percent / peak.popularity_percent
  return `${(below * 100).toFixed(0)}% below its ${peak.year} peak`
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-white/[0.04] py-2 text-sm last:border-0">
      <span className="text-[#8a8f98]">{label}</span>
      <span className="font-mono text-xs text-[#ededef]">{value}</span>
    </div>
  )
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [sex, setSex] = useState<'M' | 'F'>('F')
  const [history, setHistory] = useState<NameRow[] | null>(null)
  const [displayName, setDisplayName] = useState('')
  const [forecast, setForecast] = useState<ForecastPayload | null>(null)
  const [loading, setLoading] = useState(false)
  const [forecastLoading, setForecastLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latestDataYear, setLatestDataYear] = useState<number | null>(null)

  // The newest year present in the data decides whether a name is still in
  // current use; it is read from the data rather than hardcoded.
  useEffect(() => {
    getMeta()
      .then((meta) => setLatestDataYear(meta.max_year))
      .catch(() => setLatestDataYear(null))
  }, [])

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const name = query.trim()
    if (!name) return

    setLoading(true)
    setError(null)
    setHistory(null)
    setForecast(null)

    getNameHistory(name, sex)
      .then((data) => {
        setDisplayName(data.name)
        setHistory(data.history)
        setForecastLoading(true)
        return getNameForecast(name, sex)
          .then(setForecast)
          .finally(() => setForecastLoading(false))
      })
      .catch((e: unknown) => {
        if (e instanceof ApiError && e.status === 404) {
          setError(`No data found for “${name}” (${sex === 'F' ? 'female' : 'male'}).`)
        } else {
          setError(e instanceof ApiError ? e.message : 'Something went wrong loading the data.')
        }
      })
      .finally(() => setLoading(false))
  }

  const latest = history?.[history.length - 1]
  const previous = history?.[history.length - 2]
  const validation = forecast?.validation ?? null
  const model = forecast?.model ?? null
  const stratum = forecast?.stratum ?? null
  // Measured across the rolling backtest span, so a name the batch never
  // scored has none — see the `skill` field in lib/api.ts.
  const skill = validation?.skill
  const forecastBandWidth = forecast ? bandWidth(forecast.forecast) : null
  // The holdout is the window the batch withheld: its first year is the year
  // after the origin the model was trained through.
  const holdoutPoints = validation?.points ?? []
  const holdoutWindow = holdoutPoints.length
    ? {
        origin: holdoutPoints[0].year - 1,
        first: holdoutPoints[0].year,
        last: holdoutPoints[holdoutPoints.length - 1].year,
      }
    : null
  const peak = history ? peakPosition(history) : null
  const observedYears = history?.length ?? 0
  const forecastAbsent = forecast !== null && forecast.forecast.length === 0
  const notInCurrentUse =
    latest !== undefined && latestDataYear !== null && latest.year < latestDataYear

  // History is sparse, so "the previous year" can be years earlier. Say which
  // year the headline figures describe, and which year they are compared with.
  const statsCaption = latest
    ? previous
      ? `Most recent recorded year for ${displayName}: ${latest.year}, compared with ${previous.year}, the previous year it was recorded.`
      : `Most recent recorded year for ${displayName}: ${latest.year}.`
    : ''

  // A forecast can be missing for two quite different reasons, and saying which
  // one it is is the difference between an explanation and a silent gap.
  const forecastAbsenceReason = ((): string | null => {
    if (!forecastAbsent || !latest) return null
    if (notInCurrentUse) {
      return `No forecast: ${displayName} is not in current use. It was last recorded in ${latest.year}, and forecasts are only produced for names still recorded in ${latestDataYear}.`
    }
    if (observedYears < MIN_FORECAST_HISTORY_YEARS) {
      return `No forecast: there is not enough history for ${displayName}. Forecasting needs at least ${MIN_FORECAST_HISTORY_YEARS} recorded years, and ${displayName} has ${observedYears}.`
    }
    return `No forecast: a model could not be fitted for ${displayName}.`
  })()

  return (
    <Section>
      <div className="mb-8">
        <Tag variant="accent">FORECASTS</Tag>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight text-[#ededef] md:text-4xl">
          Name Search
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[#8a8f98]">
          Look up any name for the years it was actually recorded, plus a 5-year forecast with
          measured uncertainty bands and holdout validation. Forecasts are produced only for names
          still in use in the most recent year of data.
        </p>
      </div>

      {/* Search form */}
      <form onSubmit={handleSubmit} className="mb-10 flex flex-wrap items-end gap-4">
        <div className="min-w-64 flex-1">
          <label htmlFor="name" className="mb-1.5 block text-xs text-[#8a8f98]">
            Name
          </label>
          <input
            id="name"
            type="text"
            className={inputClass}
            placeholder="e.g. Emma, Liam"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoComplete="off"
            spellCheck="false"
          />
        </div>
        <div>
          <label className="mb-1.5 block text-xs text-[#8a8f98]">Sex</label>
          <SexToggle value={sex} onChange={setSex} />
        </div>
        <button
          type="submit"
          disabled={!query.trim() || loading}
          className="h-11 rounded-lg bg-[#0ea5e9] px-6 text-sm font-medium text-[#082f49] shadow-[0_0_0_1px_rgba(14,165,233,0.5),0_4px_12px_rgba(14,165,233,0.25)] transition-all duration-200 hover:bg-[#38bdf8] active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40"
        >
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>

      {error && <Notice variant="error">{error}</Notice>}

      {history && latest && (
        <div className="space-y-8">
          {/* Stat tiles */}
          <div className="space-y-3">
            <p className="text-xs text-[#8a8f98]">{statsCaption}</p>
            <div className="grid gap-4 sm:grid-cols-3">
              <StatTile
                label={`Rank in ${latest.year}`}
                value={formatRank(latest.popularity_rank)}
                delta={previous ? previous.popularity_rank - latest.popularity_rank : undefined}
                deltaLabel={
                  previous
                    ? `${Math.abs(latest.popularity_rank - previous.popularity_rank)} places vs ${previous.year}`
                    : undefined
                }
                comparisonYear={previous?.year}
              />
              <StatTile
                label={`Share of ${sex === 'F' ? 'female' : 'male'} births in ${latest.year}`}
                value={formatPercent(latest.popularity_percent)}
                delta={
                  previous ? latest.popularity_percent - previous.popularity_percent : undefined
                }
                deltaLabel={
                  previous
                    ? `${formatPercent(Math.abs(latest.popularity_percent - previous.popularity_percent), 4)} vs ${previous.year}`
                    : undefined
                }
                comparisonYear={previous?.year}
              />
              <StatTile
                label={`Babies named ${displayName} in ${latest.year}`}
                value={formatCount(latest.total_count)}
                delta={previous ? latest.total_count - previous.total_count : undefined}
                deltaLabel={
                  previous
                    ? `${formatCount(Math.abs(latest.total_count - previous.total_count))} vs ${previous.year}`
                    : undefined
                }
                comparisonYear={previous?.year}
              />
            </div>
          </div>

          {/* Trend + forecast chart */}
          <Card variant="glass" className="p-6">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-sm font-medium text-[#ededef]">
                Share of births for {displayName} — history & 5-year forecast
              </h2>
              {forecastLoading && (
                <span
                  className="animate-[pulse-dot_1.5s_ease-in-out_infinite] font-mono text-[11px] tracking-widest text-[#0ea5e9]"
                  role="status"
                >
                  LOADING FORECAST…
                </span>
              )}
            </div>
            {forecast ? (
              <TrendChart payload={forecast} />
            ) : (
              <TrendChart
                payload={{
                  name: displayName,
                  sex,
                  history: history.map((row) => ({
                    year: row.year,
                    value: row.popularity_percent,
                  })),
                  forecast: [],
                  validation: null,
                  model: null,
                  calibration: null,
                  stratum: null,
                }}
              />
            )}
            {forecastAbsenceReason && (
              <p className="mt-4 text-xs leading-relaxed text-[#8a8f98]">{forecastAbsenceReason}</p>
            )}
          </Card>

          {/* What this name's forecast rests on, beside what the model is */}
          {forecast && (stratum || validation) && (
            <div className="grid gap-6 lg:grid-cols-2">
              <Card variant="default" className="p-6">
                <h3 className="text-sm font-medium text-[#ededef]">What drives this forecast</h3>
                <p className="mt-1 text-xs leading-relaxed text-[#8a8f98]">
                  One pooled model forecasts every name, so there is no fit of its own to report for{' '}
                  {displayName}. These are the properties of {displayName} that decide how far the
                  forecast is trusted and how wide its band is.
                </p>
                <div className="mt-4">
                  {skill !== undefined && (
                    <Fact
                      label="Skill vs no change"
                      value={`${skill >= 0 ? '+' : '\u2212'}${formatPercent(Math.abs(skill), 1)}`}
                    />
                  )}
                  {stratum && (
                    <Fact
                      label="Popularity tier"
                      value={TIER_LABELS[stratum.tier] ?? stratum.tier}
                    />
                  )}
                  {stratum && (
                    <Fact
                      label="Volatility"
                      value={
                        VOLATILITY_LABELS[stratum.volatility_bin] ?? `Bin ${stratum.volatility_bin}`
                      }
                    />
                  )}
                  {forecastBandWidth && <Fact label="Band width (95%)" value={forecastBandWidth} />}
                  {peak && <Fact label="Against its peak" value={peak} />}
                </div>
                {skill !== undefined && (
                  <p className="mt-4 text-xs leading-relaxed text-[#8a8f98]">
                    Skill is averaged over {validation?.skill_windows} five-year window
                    {validation?.skill_windows === 1 ? '' : 's'} since 1995 — every window{' '}
                    {displayName} was eligible for — not the most recent one alone.
                  </p>
                )}
              </Card>
              {/* One model forecasts every name, so there is no per-name fit
                  to report an order or residual diagnostics for. What is
                  honestly sayable is what the model is and what it learned
                  from. See docs/adr/0010-a-pooled-model-replaces-per-name-arima.md. */}
              {model && (
                <Card variant="default" className="p-6">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium text-[#ededef]">How the forecast is made</h3>
                    <Tag variant="accent">POOLED</Tag>
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-[#8a8f98]">
                    Every name is forecast by one model, trained on how names in general have moved
                    — not by a model fitted to this name alone.
                  </p>
                  <div className="mt-4">
                    <Fact label="Model" value={model.model_name} />
                    <Fact
                      label="Trained on"
                      value={`${formatCount(model.training_rows)} name-years, ${model.training_origins} origins`}
                    />
                    <Fact label="Data through" value={String(model.trained_through)} />
                    <Fact label="Predicts" value={`${model.target}, h=1..${model.horizons}`} />
                    <Fact label="Row weighting" value={model.sample_weight} />
                  </div>
                  <div className="mt-4">
                    <div className="text-xs text-[#8a8f98]">Features it reads</div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {model.features.map((feature) => (
                        <span
                          key={feature}
                          className="rounded border border-white/[0.08] bg-white/[0.04] px-1.5 py-0.5 font-mono text-[11px] text-[#8a8f98]"
                        >
                          {feature}
                        </span>
                      ))}
                    </div>
                  </div>
                </Card>
              )}
            </div>
          )}

          {forecast && validation && (
            <Card variant="default" className="p-6">
              <h3 className="text-sm font-medium text-[#ededef]">Holdout validation</h3>
              <p className="mt-1 text-xs leading-relaxed text-[#8a8f98]">
                {holdoutWindow
                  ? `The model is trained through ${holdoutWindow.origin} and then scored on ${holdoutWindow.first}\u2013${holdoutWindow.last}, years it never saw, against what actually happened to ${displayName}.`
                  : `The model is retrained without the most recent years, then scored on ${displayName} against what actually happened.`}
              </p>
              <div className="mt-4 grid grid-cols-3 gap-4">
                <div>
                  <div className="text-xs text-[#8a8f98]">MAE</div>
                  <div className="mt-0.5 font-mono text-sm text-[#ededef]">
                    {formatPercent(validation.mae, 4)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-[#8a8f98]">RMSE</div>
                  <div className="mt-0.5 font-mono text-sm text-[#ededef]">
                    {formatPercent(validation.rmse, 4)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-[#8a8f98]">MAPE</div>
                  <div className="mt-0.5 font-mono text-sm text-[#ededef]">
                    {validation.mape.toFixed(1)}%
                  </div>
                </div>
              </div>
              {/* The three figures above are a claim about the window;
                    the window itself is the evidence for it. One row per
                    observed year the model did not see. */}
              {validation.points.length > 0 && (
                <div className="mt-4 overflow-x-auto">
                  <table
                    className="w-full text-left text-sm"
                    aria-label={`Holdout window: predicted against actual for ${displayName}`}
                  >
                    <thead>
                      <tr className="text-xs text-[#8a8f98]">
                        <th className="py-2 font-medium">Year</th>
                        <th className="py-2 text-right font-medium">Actual</th>
                        <th className="py-2 text-right font-medium">Predicted</th>
                        <th className="py-2 text-right font-medium">Error</th>
                      </tr>
                    </thead>
                    <tbody>
                      {validation.points.map((point) => (
                        <tr key={point.year} className="border-t border-white/[0.04]">
                          <td className="py-2 font-mono text-xs text-[#ededef]">{point.year}</td>
                          <td className="py-2 text-right font-mono text-xs text-[#ededef]">
                            {formatPercent(point.actual, 4)}
                          </td>
                          <td className="py-2 text-right font-mono text-xs text-[#8a8f98]">
                            {formatPercent(point.predicted, 4)}
                          </td>
                          <td className="py-2 text-right font-mono text-xs text-[#8a8f98]">
                            {point.predicted >= point.actual ? '+' : '\u2212'}
                            {formatPercent(Math.abs(point.predicted - point.actual), 4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {/* Skill compares this name's error against a naive baseline
                    that just repeats the last observed value, averaged over
                    every five-year window since 1995 rather than measured on
                    the holdout alone — see
                    docs/adr/0010-a-pooled-model-replaces-per-name-arima.md.
                    A forecast that loses to that baseline is flagged rather
                    than shown with equal confidence. */}
              <div className="mt-4">
                {skill === undefined ? null : skill >= 0 ? (
                  <p className="text-xs leading-relaxed text-emerald-400">
                    Beats the naive “no change” baseline by {formatPercent(skill, 1)}: averaged over{' '}
                    {validation.skill_windows} five-year window
                    {validation.skill_windows === 1 ? '' : 's'} since 1995, this model&apos;s error
                    was that much smaller than simply repeating the last recorded value.
                  </p>
                ) : (
                  <Notice variant="warning">
                    This forecast performs worse than simply assuming no change — across{' '}
                    {validation.skill_windows} five-year window
                    {validation.skill_windows === 1 ? '' : 's'} since 1995 its error was{' '}
                    {formatPercent(Math.abs(skill), 1)} higher than the naive baseline&apos;s. Treat
                    the forecast and its confidence bands with caution.
                  </Notice>
                )}
              </div>
            </Card>
          )}

          {/* Year-by-year table */}
          <Card variant="default" className="overflow-hidden">
            <h3 className="border-b border-white/[0.06] px-6 py-4 text-sm font-medium text-[#ededef]">
              Year-by-year data
            </h3>
            <div className="max-h-96 overflow-x-auto overflow-y-auto">
              <table
                className="w-full text-left text-sm"
                aria-label={`Year-by-year recorded data for ${displayName}`}
              >
                <thead className="sticky top-0 bg-[#0a0a0c]">
                  <tr className="text-xs text-[#8a8f98]">
                    <th className="px-6 py-3 font-medium">Year</th>
                    <th className="px-6 py-3 text-right font-medium">Babies</th>
                    <th className="px-6 py-3 text-right font-medium">Share of births</th>
                    <th className="px-6 py-3 text-right font-medium">Rank</th>
                  </tr>
                </thead>
                <tbody>
                  {[...history].reverse().map((row) => (
                    <tr
                      key={row.year}
                      className="border-b border-white/[0.04] transition-colors last:border-0 hover:bg-white/[0.03]"
                    >
                      <td className="px-6 py-2.5 font-mono text-xs text-[#ededef]">{row.year}</td>
                      <td className="px-6 py-2.5 text-right font-mono text-xs text-[#ededef]">
                        {formatCount(row.total_count)}
                      </td>
                      <td className="px-6 py-2.5 text-right font-mono text-xs text-[#8a8f98]">
                        {formatPercent(row.popularity_percent, 4)}
                      </td>
                      <td className="px-6 py-2.5 text-right font-mono text-xs text-[#8a8f98]">
                        {row.popularity_rank}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </Section>
  )
}

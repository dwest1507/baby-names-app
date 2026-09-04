'use client'

import { useEffect, useState } from 'react'
import Card from '@/components/ui/Card'
import Notice from '@/components/ui/Notice'
import Section from '@/components/layout/Section'
import SexToggle from '@/components/ui/SexToggle'
import Tag from '@/components/ui/Tag'
import TopNamesChart from '@/components/charts/TopNamesChart'
import { ApiError, getMeta, getTopNames, type NameRow } from '@/lib/api'
import { formatCount, formatPercent } from '@/lib/format'

const inputClass =
  'h-10 rounded-lg border border-white/[0.08] bg-white/[0.04] px-3 text-sm text-[#ededef] transition-all duration-150 focus:border-[#0ea5e9]/50 focus:bg-white/[0.06] focus:shadow-[0_0_0_3px_rgba(14,165,233,0.15)] focus:outline-none'

export default function ExplorePage() {
  const [sex, setSex] = useState<'M' | 'F'>('F')
  const [year, setYear] = useState(2024)
  const [limit, setLimit] = useState(20)
  const [yearRange, setYearRange] = useState({ min_year: 1880, max_year: 2024 })
  // The input keeps its own draft so a half-typed year ("19") does not get
  // clamped away mid-edit; `year` only advances to values the API accepts.
  const [yearDraft, setYearDraft] = useState('2024')

  // Result is tagged with the filter key it was fetched for; a mismatch means loading.
  const filterKey = `${sex}-${year}-${limit}`
  const [result, setResult] = useState<{
    key: string
    names?: NameRow[]
    error?: string
  } | null>(null)

  useEffect(() => {
    getMeta()
      .then((meta) => {
        setYearRange(meta)
        setYear(meta.max_year)
        setYearDraft(String(meta.max_year))
      })
      .catch(() => {
        // Fall back to defaults; the top-names request will surface real errors
      })
  }, [])

  useEffect(() => {
    let cancelled = false
    getTopNames(sex, year, limit)
      .then((data) => {
        if (!cancelled) setResult({ key: filterKey, names: data.names })
      })
      .catch((e: unknown) => {
        if (cancelled) return
        setResult({
          key: filterKey,
          error: e instanceof ApiError ? e.message : 'Something went wrong loading the data.',
        })
      })
    return () => {
      cancelled = true
    }
  }, [sex, year, limit, filterKey])

  const loading = result?.key !== filterKey
  const error = loading ? null : (result?.error ?? null)
  const names = loading ? null : (result?.names ?? null)

  return (
    <Section>
      <div className="mb-8">
        <Tag variant="accent">RANKINGS</Tag>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight text-[#ededef] md:text-4xl">
          Top Names
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[#8a8f98]">
          The most popular baby names for a given year and sex, ranked by the number of babies
          registered with the Social Security Administration.
        </p>
      </div>

      {/* Filters */}
      <div className="mb-8 flex flex-wrap items-end gap-4">
        <div>
          <label className="mb-1.5 block text-xs text-[#8a8f98]">Sex</label>
          <SexToggle value={sex} onChange={setSex} />
        </div>
        <div>
          <label htmlFor="year" className="mb-1.5 block text-xs text-[#8a8f98]">
            Year
          </label>
          <input
            id="year"
            type="number"
            className={`${inputClass} w-28`}
            min={yearRange.min_year}
            max={yearRange.max_year}
            value={yearDraft}
            onChange={(e) => {
              setYearDraft(e.target.value)
              const next = Number(e.target.value)
              if (e.target.value === '' || Number.isNaN(next)) return
              if (next < yearRange.min_year || next > yearRange.max_year) return
              setYear(next)
            }}
            onBlur={() => setYearDraft(String(year))}
          />
        </div>
        <div>
          <label htmlFor="limit" className="mb-1.5 block text-xs text-[#8a8f98]">
            Show top
          </label>
          <select
            id="limit"
            className={`${inputClass} w-24 appearance-none`}
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
          >
            {[10, 20, 30, 50, 100].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && <Notice variant="error">{error}</Notice>}

      {loading && (
        <p className="py-12 text-center text-sm text-[#8a8f98]" role="status">
          Loading top names…
        </p>
      )}

      {!loading && !error && names && names.length === 0 && (
        <Notice>
          No names found for {year}. Try a year between {yearRange.min_year} and{' '}
          {yearRange.max_year}.
        </Notice>
      )}

      {!loading && !error && names && names.length > 0 && (
        <div className="space-y-8">
          <Card variant="glass" className="p-6">
            <h2 className="mb-4 text-sm font-medium text-[#ededef]">
              Top {names.length} {sex === 'F' ? 'female' : 'male'} names of {year} — babies
              registered
            </h2>
            <TopNamesChart data={names} />
          </Card>

          <Card variant="default" className="overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-white/[0.06] text-xs text-[#8a8f98]">
                    <th className="px-6 py-3 font-medium">Rank</th>
                    <th className="px-6 py-3 font-medium">Name</th>
                    <th className="px-6 py-3 text-right font-medium">Babies</th>
                    <th className="px-6 py-3 text-right font-medium">Share of births</th>
                  </tr>
                </thead>
                <tbody>
                  {names.map((row) => (
                    <tr
                      key={row.name}
                      className="border-b border-white/[0.04] transition-colors last:border-0 hover:bg-white/[0.03]"
                    >
                      <td className="px-6 py-3 font-mono text-xs text-[#8a8f98]">
                        {row.popularity_rank}
                      </td>
                      <td className="px-6 py-3 text-[#ededef]">{row.name}</td>
                      <td className="px-6 py-3 text-right font-mono text-xs text-[#ededef]">
                        {formatCount(row.total_count)}
                      </td>
                      <td className="px-6 py-3 text-right font-mono text-xs text-[#8a8f98]">
                        {formatPercent(row.popularity_percent)}
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

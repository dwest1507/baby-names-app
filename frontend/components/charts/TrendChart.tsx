'use client'

import { useMemo } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  DefaultLegendContent,
  type DefaultLegendContentProps,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  type TooltipContentProps,
  XAxis,
  YAxis,
} from 'recharts'
import type { NameType, ValueType } from 'recharts/types/component/DefaultTooltipContent'
import type { ForecastPayload } from '@/lib/api'
import {
  AXIS_LINE,
  AXIS_TICK,
  CHART_COLORS,
  GRID_STROKE,
  TOOLTIP_LABEL_STYLE,
  TOOLTIP_STYLE,
} from './chartTheme'

interface TrendChartProps {
  payload: ForecastPayload
}

export interface Row {
  year: number
  history?: number
  forecast?: number
  ci80?: [number, number]
  ci95?: [number, number]
}

/** History, the forecast that continues it, then the inner and outer interval. */
const LEGEND_ORDER: (keyof Row)[] = ['history', 'forecast', 'ci80', 'ci95']

type BandKey = 'ci80' | 'ci95'

/**
 * How opaque each interval looks where a visitor reads it. The one place band
 * density is decided: the band fills and their legend swatches both derive
 * from it, so the two match by construction.
 */
const BAND_OPACITY: Record<BandKey, number> = { ci95: 0.16, ci80: 0.41 }

/**
 * The alpha each band is painted with. The inner band is painted over the
 * outer, so where it sits the two composite to 1 − (1 − outer)(1 − inner);
 * painted at its target it would render denser than intended. Its raw alpha is
 * solved for the composite instead.
 */
const BAND_FILL_OPACITY: Record<BandKey, number> = {
  ci95: BAND_OPACITY.ci95,
  ci80: 1 - (1 - BAND_OPACITY.ci80) / (1 - BAND_OPACITY.ci95),
}

function withAlpha(hex: string, alpha: number): string {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16))
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

const BAND_SWATCH: Record<BandKey, string> = {
  ci95: withAlpha(CHART_COLORS.forecast, BAND_OPACITY.ci95),
  ci80: withAlpha(CHART_COLORS.forecast, BAND_OPACITY.ci80),
}

/**
 * recharts draws a band's swatch in the band's own fill at full opacity. Give
 * each interval the colour its band renders at instead, keeping its label in
 * the series colour so the text does not fade with the swatch.
 */
function BandLegend(props: DefaultLegendContentProps) {
  const payload = props.payload?.map((entry) => {
    const swatch = BAND_SWATCH[entry.dataKey as BandKey]
    if (swatch === undefined) return entry
    return {
      ...entry,
      color: swatch,
      formatter: (value: string) => <span style={{ color: entry.color }}>{value}</span>,
    }
  })
  return <DefaultLegendContent {...props} payload={payload} />
}

const formatValue = (percent: number) => `${percent.toFixed(4)}%`
const formatRange = ([lo, hi]: [number, number]) => `${formatValue(lo)} – ${formatValue(hi)}`

interface TrendTooltipProps extends TooltipContentProps<ValueType, NameType> {
  label80: string
  label95: string
}

/**
 * What a hovered year says in numbers. A forecast year states its projected
 * value and both ranges, because the band is where the uncertainty is and a
 * shaded region is unreadable to anyone who cannot interpret one.
 */
function TrendTooltip({ active, payload, label80, label95 }: TrendTooltipProps) {
  const row = payload?.[0]?.payload as Row | undefined
  if (!active || row === undefined) return null

  // The last recorded year also carries a forecast value, only so the two lines
  // meet; what was recorded there is the whole story.
  const lines: [string, string][] = []
  if (row.history !== undefined) {
    lines.push(['Recorded', formatValue(row.history)])
  } else {
    if (row.forecast !== undefined) lines.push(['Forecast', formatValue(row.forecast)])
    if (row.ci80 !== undefined) lines.push([label80, formatRange(row.ci80)])
    if (row.ci95 !== undefined) lines.push([label95, formatRange(row.ci95)])
  }

  return (
    <div style={{ ...TOOLTIP_STYLE, padding: '8px 10px' }}>
      <p style={TOOLTIP_LABEL_STYLE}>{row.year}</p>
      {lines.map(([name, value]) => (
        <p key={name}>
          {name}: {value}
        </p>
      ))}
    </div>
  )
}

function toPercent(fraction: number): number {
  return fraction * 100
}

/**
 * One row per year the chart draws, history and forecast merged.
 *
 * Exported because the merge carries the rule that history outranks a
 * forecast, and that rule is worth pinning directly rather than inferring
 * from rendered SVG.
 */
export function buildChartRows(payload: ForecastPayload): {
  rows: Row[]
  forecastStart: number | undefined
} {
  const byYear = new Map<number, Row>()

  for (const point of payload.history) {
    byYear.set(point.year, { year: point.year, history: toPercent(point.value) })
  }

  // A forecast point for a year that has already been observed would overwrite
  // the record with a guess at it — the history line would end early and the
  // dashed line would cover a year the table below reports as recorded. That
  // is not hypothetical: the database is published independently of this code
  // (ADR 0006), so a deploy can meet an artifact built before the newest year
  // arrived, whose forecast starts on a year the history now covers. Recorded
  // history wins.
  const lastHistoryYear = payload.history[payload.history.length - 1]?.year
  for (const point of payload.forecast) {
    if (lastHistoryYear !== undefined && point.year <= lastHistoryYear) continue
    byYear.set(point.year, {
      year: point.year,
      forecast: toPercent(point.mean),
      ci80: [toPercent(point.lo80), toPercent(point.hi80)],
      ci95: [toPercent(point.lo95), toPercent(point.hi95)],
    })
  }

  // Connect the forecast line to the last historical point
  if (lastHistoryYear !== undefined && payload.forecast.length > 0) {
    const last = byYear.get(lastHistoryYear)
    if (last?.history !== undefined) last.forecast = last.history
  }

  // A year with no row is a year in which no births were recorded. Emit it as
  // an explicit empty row so the line breaks there instead of being drawn
  // straight across the gap.
  const ordered = [...byYear.values()].sort((a, b) => a.year - b.year)
  const rows: Row[] = []
  for (const row of ordered) {
    const previous = rows[rows.length - 1]
    if (previous) {
      for (let year = previous.year + 1; year < row.year; year++) rows.push({ year })
    }
    rows.push(row)
  }

  return { rows, forecastStart: lastHistoryYear }
}

export default function TrendChart({ payload }: TrendChartProps) {
  const { rows, forecastStart } = useMemo(() => buildChartRows(payload), [payload])

  const hasForecast = payload.forecast.length > 0

  // Bands must be labelled with the coverage they actually achieve — measured
  // by the precompute batch's holdout backtest across every eligible name —
  // not the nominal 80%/95% level, which the parent PRD found could overstate
  // coverage by 24 points. The row the payload carries is the one measured for
  // *this* name's popularity tier and volatility bin, so the label follows it
  // rather than any population figure. Falls back to the nominal label only if
  // calibration is missing entirely. See
  // docs/adr/0011-conformal-bands-keyed-by-strata.md.
  const intervalLabel = (nominal: '0.8' | '0.95'): string => {
    const measured = payload.calibration?.[nominal]?.empirical_coverage
    if (measured === undefined) return `${Math.round(Number(nominal) * 100)}% interval`
    return `${Math.round(measured * 100)}% interval`
  }
  const label80 = intervalLabel('0.8')
  const label95 = intervalLabel('0.95')

  return (
    <div
      className="h-[440px] w-full"
      role="img"
      aria-label={`Popularity trend and forecast for ${payload.name}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke={GRID_STROKE} vertical={false} />
          <XAxis
            dataKey="year"
            tick={AXIS_TICK}
            axisLine={AXIS_LINE}
            tickLine={false}
            type="number"
            domain={['dataMin', 'dataMax']}
            tickCount={10}
          />
          <YAxis
            tick={AXIS_TICK}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `${v.toFixed(2)}%`}
            width={64}
          />
          <Tooltip
            content={(props) => <TrendTooltip {...props} label80={label80} label95={label95} />}
          />
          {/* In recharts child order is paint order, and the bands have to be
              painted beneath the lines. The legend is sorted separately so it
              reads in the order the chart is understood instead. */}
          <Legend
            wrapperStyle={{ fontSize: 12, color: '#8a8f98' }}
            iconSize={10}
            content={BandLegend}
            itemSorter={(item) => LEGEND_ORDER.indexOf(item.dataKey as keyof Row)}
          />

          {hasForecast && (
            <Area
              dataKey="ci95"
              name={label95}
              stroke="none"
              fill={CHART_COLORS.forecast}
              fillOpacity={BAND_FILL_OPACITY.ci95}
              connectNulls={false}
              isAnimationActive={false}
              legendType="rect"
              tooltipType="none"
            />
          )}
          {hasForecast && (
            <Area
              dataKey="ci80"
              name={label80}
              stroke="none"
              fill={CHART_COLORS.forecast}
              fillOpacity={BAND_FILL_OPACITY.ci80}
              connectNulls={false}
              isAnimationActive={false}
              legendType="rect"
              tooltipType="none"
            />
          )}

          <Line
            dataKey="history"
            name="Historical"
            stroke={CHART_COLORS.history}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
            connectNulls={false}
            isAnimationActive={false}
          />
          {hasForecast && (
            /* The band is what the chart is really saying; the central line
               is one path through it. Drawn thinner than the history it
               continues and without the point markers that would read as
               five measured values — but not dimmer than its own fill, or it
               disappears into the band it sits in. See
               docs/adr/0011-conformal-bands-keyed-by-strata.md. */
            <Line
              dataKey="forecast"
              name="Forecast"
              stroke={CHART_COLORS.forecast}
              strokeWidth={1.25}
              strokeOpacity={0.92}
              strokeDasharray="6 4"
              dot={false}
              activeDot={{ r: 3 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          )}

          {hasForecast && forecastStart !== undefined && (
            <ReferenceLine
              x={forecastStart}
              stroke="rgba(255, 255, 255, 0.2)"
              strokeDasharray="3 3"
              label={{
                value: 'Forecast →',
                fill: '#8a8f98',
                fontSize: 11,
                position: 'insideTopRight',
              }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

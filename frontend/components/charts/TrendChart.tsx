'use client'

import { useMemo } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
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
  predicted?: number
  ci80?: [number, number]
  ci95?: [number, number]
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
  for (const point of payload.validation?.points ?? []) {
    const row = byYear.get(point.year) ?? { year: point.year }
    row.predicted = toPercent(point.predicted)
    byYear.set(point.year, row)
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
  const hasValidation = (payload.validation?.points.length ?? 0) > 0

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

  // The line carries how well the pooled model has actually done on *this*
  // name — its skill against a naive "no change" baseline, averaged over
  // every five-year window it was eligible for since 1995. A visitor reading
  // the dashed line should not have to look elsewhere to learn that for some
  // names it is worth less than the flat line they could have drawn
  // themselves. See docs/adr/0010-a-pooled-model-replaces-per-name-arima.md.
  const skill = payload.validation?.skill
  const forecastLabel =
    skill === undefined
      ? 'Pooled forecast'
      : skill >= 0
        ? `Pooled forecast · ${Math.round(skill * 100)}% better than no change`
        : `Pooled forecast · ${Math.round(-skill * 100)}% worse than no change`

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
            contentStyle={TOOLTIP_STYLE}
            labelStyle={TOOLTIP_LABEL_STYLE}
            formatter={(value, name) => {
              if (Array.isArray(value)) {
                return [`${value[0].toFixed(4)}% – ${value[1].toFixed(4)}%`, name]
              }
              return [`${Number(value).toFixed(4)}%`, name]
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: '#8a8f98' }} iconSize={10} />

          {hasForecast && (
            <Area
              dataKey="ci95"
              name={label95}
              stroke="none"
              fill={CHART_COLORS.forecast}
              fillOpacity={0.16}
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
              fillOpacity={0.3}
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
               is one path through it. Drawn thinner and dimmer than the
               history it continues, and without the point markers that would
               read as five measured values. See
               docs/adr/0011-conformal-bands-keyed-by-strata.md. */
            <Line
              dataKey="forecast"
              name={forecastLabel}
              stroke={CHART_COLORS.forecast}
              strokeWidth={1.25}
              strokeOpacity={0.65}
              strokeDasharray="6 4"
              dot={false}
              activeDot={{ r: 3 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          )}
          {hasValidation && (
            <Line
              dataKey="predicted"
              name="Validation predictions"
              stroke={CHART_COLORS.validation}
              strokeWidth={2}
              strokeDasharray="2 4"
              dot={{ r: 3, fill: CHART_COLORS.validation, strokeWidth: 0 }}
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

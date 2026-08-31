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

interface Row {
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

export default function TrendChart({ payload }: TrendChartProps) {
  const { rows, forecastStart } = useMemo(() => {
    const byYear = new Map<number, Row>()

    for (const point of payload.history) {
      byYear.set(point.year, { year: point.year, history: toPercent(point.value) })
    }
    for (const point of payload.validation?.points ?? []) {
      const row = byYear.get(point.year) ?? { year: point.year }
      row.predicted = toPercent(point.predicted)
      byYear.set(point.year, row)
    }

    const lastHistoryYear = payload.history[payload.history.length - 1]?.year
    for (const point of payload.forecast) {
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
  }, [payload])

  const hasForecast = payload.forecast.length > 0
  const hasValidation = (payload.validation?.points.length ?? 0) > 0

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
              name="95% interval"
              stroke="none"
              fill={CHART_COLORS.forecast}
              fillOpacity={0.1}
              connectNulls={false}
              isAnimationActive={false}
              legendType="none"
              tooltipType="none"
            />
          )}
          {hasForecast && (
            <Area
              dataKey="ci80"
              name="80% interval"
              stroke="none"
              fill={CHART_COLORS.forecast}
              fillOpacity={0.18}
              connectNulls={false}
              isAnimationActive={false}
              legendType="none"
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
            <Line
              dataKey="forecast"
              name="ARIMA forecast"
              stroke={CHART_COLORS.forecast}
              strokeWidth={2}
              strokeDasharray="6 4"
              dot={{ r: 3, fill: CHART_COLORS.forecast, strokeWidth: 0 }}
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

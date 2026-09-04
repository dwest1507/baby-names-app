'use client'

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { NameRow } from '@/lib/api'
import { formatCount } from '@/lib/format'
import {
  AXIS_LINE,
  AXIS_TICK,
  CHART_COLORS,
  GRID_STROKE,
  TOOLTIP_LABEL_STYLE,
  TOOLTIP_STYLE,
} from './chartTheme'

interface TopNamesChartProps {
  data: NameRow[]
}

export default function TopNamesChart({ data }: TopNamesChartProps) {
  return (
    <div className="h-[420px] w-full" role="img" aria-label="Bar chart of top names by count">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 8, right: 8, bottom: 48, left: 8 }}
          barCategoryGap="25%"
        >
          <CartesianGrid stroke={GRID_STROKE} vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ ...AXIS_TICK, angle: -45, textAnchor: 'end' }}
            axisLine={AXIS_LINE}
            tickLine={false}
            interval={0}
            height={70}
          />
          <YAxis
            tick={AXIS_TICK}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : `${v}`)}
            width={44}
          />
          <Tooltip
            cursor={{ fill: 'rgba(255, 255, 255, 0.04)' }}
            contentStyle={TOOLTIP_STYLE}
            labelStyle={TOOLTIP_LABEL_STYLE}
            formatter={(value) => [formatCount(Number(value)), 'Babies']}
          />
          <Bar
            dataKey="total_count"
            fill={CHART_COLORS.bar}
            fillOpacity={0.85}
            radius={[4, 4, 0, 0]}
            maxBarSize={40}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

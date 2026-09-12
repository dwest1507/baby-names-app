import { describe, expect, it } from 'vitest'
import { buildChartRows } from '@/components/charts/TrendChart'
import type { ForecastPayload } from '@/lib/api'

function payload(overrides: Partial<ForecastPayload> = {}): ForecastPayload {
  return {
    name: 'Emma',
    sex: 'F',
    history: [2022, 2023, 2024, 2025].map((year, i) => ({
      year,
      value: 0.001 + i * 0.0001,
    })),
    forecast: [2026, 2027].map((year) => ({
      year,
      mean: 0.002,
      lo80: 0.0015,
      hi80: 0.0025,
      lo95: 0.001,
      hi95: 0.003,
    })),
    validation: null,
    model: null,
    calibration: null,
    ...overrides,
  }
}

describe('buildChartRows', () => {
  it('draws the forecast after the last observed year', () => {
    const { rows, forecastStart } = buildChartRows(payload())

    expect(forecastStart).toBe(2025)
    expect(rows.filter((row) => row.ci80 !== undefined).map((row) => row.year)).toEqual([
      2026, 2027,
    ])
    // The forecast line reaches back to the last observed point so the two
    // lines meet rather than leaving a visible gap.
    expect(rows.find((row) => row.year === 2025)?.forecast).toBe(
      rows.find((row) => row.year === 2025)?.history
    )
  })

  it('keeps recorded history when a stale artifact forecasts a year that has happened', () => {
    // The database is published separately from this code (ADR 0006), so a
    // deploy can meet an artifact built a year earlier — one whose forecast
    // starts on a year the history now covers. Before the pooled batch moved
    // the origin to 2025, exactly this shipped: 2025 was an observed year in
    // `names` and a forecast year in `forecasts` at the same time.
    const stale = payload({
      forecast: [2025, 2026].map((year) => ({
        year,
        mean: 0.09,
        lo80: 0.08,
        hi80: 0.1,
        lo95: 0.07,
        hi95: 0.11,
      })),
    })

    const { rows } = buildChartRows(stale)
    const observed = rows.find((row) => row.year === 2025)

    expect(observed?.history).toBeCloseTo(0.13, 10)
    expect(observed?.ci80).toBeUndefined()
    expect(rows.filter((row) => row.ci80 !== undefined).map((row) => row.year)).toEqual([2026])
  })

  it('breaks the line across years with no recorded births', () => {
    const gapped = payload({
      history: [
        { year: 2019, value: 0.001 },
        { year: 2024, value: 0.002 },
        { year: 2025, value: 0.003 },
      ],
    })

    const { rows } = buildChartRows(gapped)

    expect(rows.map((row) => row.year)).toEqual([
      2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027,
    ])
    expect(rows.find((row) => row.year === 2021)?.history).toBeUndefined()
  })
})

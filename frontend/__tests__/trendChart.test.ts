import { describe, expect, it } from 'vitest'
import { buildChartRows, wheelZoom, yearTicks, zoomDomain } from '@/components/charts/TrendChart'
import type { ForecastPayload } from '@/lib/api'

function payload(overrides: Partial<ForecastPayload> = {}): ForecastPayload {
  return {
    name: 'Emma',
    sex: 'F',
    history: [2022, 2023, 2024, 2025].map((year, i) => ({
      year,
      value: 0.001 + i * 0.0001,
    })),
    forecast: [2026, 2027].map((year, i) => ({
      year,
      mean: 0.002,
      projected_rank: 12 + i,
      lo80: 0.0015,
      hi80: 0.0025,
      lo95: 0.001,
      hi95: 0.003,
    })),
    validation: null,
    model: null,
    calibration: null,
    stratum: null,
    track_record: {},
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
      forecast: [2025, 2026].map((year, i) => ({
        year,
        mean: 0.09,
        projected_rank: 12 + i,
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

  it('draws only recorded history and the forecast, not what past forecasts said', () => {
    // A third line competes with the story the chart tells and repeats what
    // the page reports in its historical table. See issue #59.
    const withTrackRecord = payload({
      track_record: {
        '5': [2023, 2024].map((year) => ({ year, projected_share: 0.0042, projected_rank: 42 })),
      },
    })

    const { rows } = buildChartRows(withTrackRecord)

    expect(rows.map((row) => row.year)).toEqual([2022, 2023, 2024, 2025, 2026, 2027])
    expect(JSON.stringify(rows)).not.toContain('0.42')
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

describe('zoomDomain', () => {
  it('restores the full range when there is no selection', () => {
    const { rows } = buildChartRows(payload())

    const domain = zoomDomain(rows, null)

    expect(domain?.x).toEqual([2022, 2027])
    // The full view keeps its zero baseline and reaches the top of the outer band.
    expect(domain?.y[0]).toBe(0)
    expect(domain?.y[1]).toBeGreaterThanOrEqual(0.3)
  })

  it('ends the full view on a round value, labelled at even steps', () => {
    // Peaking at 2.2012%, an axis ending on its raw maximum would label its top
    // "2.21%" and run the line into the frame.
    const { rows } = buildChartRows(
      payload({
        history: [
          { year: 2000, value: 0.022012 },
          { year: 2001, value: 0.02 },
        ],
        forecast: [],
      })
    )

    const domain = zoomDomain(rows, null)

    expect(domain?.y).toEqual([0, 2.5])
    expect(domain?.yTicks).toEqual([0, 0.5, 1, 1.5, 2, 2.5])
  })

  it('refits the vertical axis to the years selected, whichever way they were dragged', () => {
    // Recorded shares climb from 0.10% in 2022 to 0.13% in 2025 and the bands
    // reach 0.30% after. Zoomed to 2022–2023 the axis should frame 0.10–0.11%,
    // not magnify the whitespace up to the band.
    const { rows } = buildChartRows(payload())

    const domain = zoomDomain(rows, { from: 2023, to: 2022 })

    expect(domain?.x).toEqual([2022, 2023])
    const [low, high] = domain!.y
    expect(low).toBeGreaterThan(0.05)
    expect(low).toBeLessThanOrEqual(0.1)
    expect(high).toBeGreaterThanOrEqual(0.11)
    expect(high).toBeLessThan(0.12)
    // Labelled at a round step, not at whatever the padded extremes came to.
    expect(domain!.yTicks).toEqual([0.095, 0.1, 0.105, 0.11, 0.115])
  })

  it('rejects a selection narrower than one year', () => {
    // A click without a drag starts and ends on the same year; zooming to it
    // would leave nothing to draw.
    const { rows } = buildChartRows(payload())

    expect(zoomDomain(rows, { from: 2024, to: 2024 })).toBeNull()
    expect(zoomDomain(rows, { from: 2024, to: 2024.5 })).toBeNull()
    expect(zoomDomain(rows, { from: 2024, to: 2025 })).not.toBeNull()
  })

  it('rejects a selection with nothing drawn in it', () => {
    // Years with no recorded births are empty rows: there is no value to fit
    // the vertical axis to.
    const { rows } = buildChartRows(
      payload({
        history: [
          { year: 2010, value: 0.001 },
          { year: 2020, value: 0.002 },
          { year: 2025, value: 0.003 },
        ],
      })
    )

    expect(zoomDomain(rows, { from: 2012, to: 2016 })).toBeNull()
  })
})

describe('wheelZoom', () => {
  // 1990–2025 recorded, 2026–2027 forecast.
  const { rows } = buildChartRows(
    payload({
      history: Array.from({ length: 36 }, (_, i) => ({ year: 1990 + i, value: 0.001 })),
    })
  )

  it('zooms in about the year under the pointer', () => {
    const zoomed = wheelZoom(rows, null, 2020, 'in')!

    const [from, to] = [zoomed.from, zoomed.to]
    expect(to - from).toBeLessThan(2027 - 1990)
    // The year under the pointer stays where it was across the plot.
    expect((2020 - from) / (to - from)).toBeCloseTo((2020 - 1990) / (2027 - 1990), 6)
  })

  it('returns to the full range once zooming out reaches it', () => {
    let range = wheelZoom(rows, null, 2020, 'in')
    range = wheelZoom(rows, range, 2020, 'out')
    range = wheelZoom(rows, range, 2020, 'out')

    expect(range).toBeNull()
  })

  it('stops zooming in before the view is narrower than one year', () => {
    let range: ReturnType<typeof wheelZoom> = null
    for (let i = 0; i < 50; i++) range = wheelZoom(rows, range, 2020, 'in')

    expect(range!.to - range!.from).toBeGreaterThanOrEqual(1)
    expect(zoomDomain(rows, range)).not.toBeNull()
  })
})

describe('yearTicks', () => {
  it('labels round years rather than dividing the range evenly', () => {
    expect(yearTicks([1880, 2030])).toEqual([1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020])
    // A wheel zoom leaves fractional edges; the labels stay on round years inside them.
    expect(yearTicks([1904.3, 2024.1])).toEqual([1920, 1940, 1960, 1980, 2000, 2020])
    expect(yearTicks([1997, 2029])).toEqual([2000, 2005, 2010, 2015, 2020, 2025])
  })

  it('never labels a year that is not a whole one', () => {
    expect(yearTicks([2000, 2003])).toEqual([2000, 2001, 2002, 2003])
  })
})

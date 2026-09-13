import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { cloneElement, type ReactElement } from 'react'
import userEvent from '@testing-library/user-event'
import type { NameRow } from '@/lib/api'
import { CHART_COLORS } from '@/components/charts/chartTheme'
import { formatPercent } from '@/lib/format'

const getMeta = vi.fn()
const getNameHistory = vi.fn()
const getNameForecast = vi.fn()

vi.mock('@/lib/api', async () => {
  const actual = await vi.importActual<typeof import('@/lib/api')>('@/lib/api')
  return {
    ...actual,
    getMeta: () => getMeta(),
    getNameHistory: (...a: unknown[]) => getNameHistory(...a),
    getNameForecast: (...a: unknown[]) => getNameForecast(...a),
  }
})

// recharts measures its container, which jsdom reports as zero-sized; give the
// chart a fixed size so it actually renders its paths.
vi.mock('recharts', async () => {
  const actual = await vi.importActual<typeof import('recharts')>('recharts')
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: ReactElement }) =>
      cloneElement(children as ReactElement<{ width?: number; height?: number }>, {
        width: 800,
        height: 400,
      }),
  }
})

import SearchPage from '@/app/search/page'

const NEWEST_YEAR = 2025

/** Observed rows for the given years — nothing for the years in between. */
function historyFor(name: string, years: number[]): NameRow[] {
  return years.map((year, i) => ({
    name,
    sex: 'F' as const,
    year,
    total_count: 1000 + i * 10,
    popularity_percent: 0.001 + i * 0.0001,
    popularity_rank: 50 - i,
  }))
}

function emptyForecast(name: string, history: NameRow[]) {
  return {
    name,
    sex: 'F' as const,
    history: history.map((row) => ({ year: row.year, value: row.popularity_percent })),
    forecast: [],
    validation: null,
    model: null,
    calibration: null,
    stratum: null,
    track_record: {},
  }
}

/** The global model card: one pooled model produces every name's forecast, so
 * what the page can say about "the model" describes the batch rather than this
 * name. See docs/adr/0010-a-pooled-model-replaces-per-name-arima.md. */
const MODEL_CARD = {
  model_name: 'LightGBM Pooled Regressor (h=1..5)',
  model_class: 'gradient-boosted trees',
  target: 'log(y[t+h] / y[t])',
  features: ['g1', 'g5', 'accel', 'vol', 'level'],
  horizons: 5,
  trained_through: NEWEST_YEAR,
  training_origins: 91,
  training_rows: 638_412,
  sample_weight: 'share^0.5',
  seed: 0,
}

/** A forecast with validation, the model card and measured calibration —
 * everything the "Holdout validation" panel and the chart's interval labels
 * read from. */
function fullForecast(
  name: string,
  history: NameRow[],
  overrides: {
    skill?: number
    empirical80?: number
    empirical95?: number
    tier?: string
    volatilityBin?: number
    stratum?: { tier: string; volatility_bin: number } | null
    trackRecord?: Record<
      string,
      { year: number; projected_share: number; projected_rank: number }[]
    >
  } = {}
) {
  const {
    skill = 0.25,
    empirical80 = 0.44,
    empirical95 = 0.51,
    tier = 'top100',
    volatilityBin = 1,
    stratum = { tier, volatility_bin: volatilityBin },
    // What the fit h years earlier said about each year from 1995 + h on: 2h%
    // high in odd years and h% low in even ones, so one year ahead it misses
    // by +2% / −1% and five years ahead by +10% / −5%.
    trackRecord = Object.fromEntries(
      [1, 2, 3, 4, 5].map((h) => [
        String(h),
        history
          .filter((row) => row.year >= 1995 + h)
          .map((row) => ({
            year: row.year,
            projected_share: row.popularity_percent * (row.year % 2 ? 1 + 0.02 * h : 1 - 0.01 * h),
            // Missed by exactly the horizon, so a rank read off the page says
            // which horizon produced it.
            projected_rank: row.popularity_rank + h,
          })),
      ])
    ),
  } = overrides
  return {
    name,
    sex: 'F' as const,
    history: history.map((row) => ({ year: row.year, value: row.popularity_percent })),
    forecast: [2026, 2027, 2028, 2029, 2030].map((year, i) => ({
      year,
      mean: 0.002,
      // Ranked in the batch against the whole field observed at the origin,
      // so the page only prints it. See
      // docs/adr/0013-projected-rank-against-a-frozen-field.md.
      projected_rank: 12 + i,
      lo80: 0.0015,
      hi80: 0.0025,
      lo95: 0.001,
      hi95: 0.003,
    })),
    validation: {
      mae: 0.0000123,
      rmse: 0.0000234,
      mape: 12.3,
      skill,
      skill_windows: 26,
    },
    model: MODEL_CARD,
    // Coverage is measured per stratum, so the row a name carries is the one
    // for its own popularity tier and volatility bin — not a population
    // average. See docs/adr/0011-conformal-bands-keyed-by-strata.md.
    calibration: {
      '0.8': {
        nominal: 0.8,
        tier,
        volatility_bin: volatilityBin,
        empirical_coverage: empirical80,
        n: 45,
      },
      '0.95': {
        nominal: 0.95,
        tier,
        volatility_bin: volatilityBin,
        empirical_coverage: empirical95,
        n: 45,
      },
    },
    // The name's *own* stratum, which is not the stratum the calibration row
    // describes: a name whose cell was too thin to earn a band is served the
    // population's. See docs/adr/0011-conformal-bands-keyed-by-strata.md.
    stratum,
    track_record: trackRecord,
  }
}

async function search(name: string) {
  const user = userEvent.setup()
  render(<SearchPage />)
  await user.type(screen.getByLabelText('Name'), name)
  await user.click(screen.getByRole('button', { name: 'Search' }))
}

describe('SearchPage forecast absence', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  it('explains that a name no longer in use is not forecast', async () => {
    const years = Array.from({ length: 34 }, (_, i) => 1960 + i) // ends 1993
    const history = historyFor('Debra', years)
    getNameHistory.mockResolvedValue({ name: 'Debra', sex: 'F', history })
    getNameForecast.mockResolvedValue(emptyForecast('Debra', history))

    await search('Debra')

    const notice = await screen.findByText(/not in current use/i)
    expect(notice).toHaveTextContent('1993')
    expect(notice).toHaveTextContent(String(NEWEST_YEAR))
  })
})

describe('SearchPage forecast absence for a recent arrival', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  it('explains that a name recorded in the newest year has too little history', async () => {
    const years = Array.from({ length: 8 }, (_, i) => 2018 + i) // ends 2025
    const history = historyFor('Mateo', years)
    getNameHistory.mockResolvedValue({ name: 'Mateo', sex: 'F', history })
    getNameForecast.mockResolvedValue(emptyForecast('Mateo', history))

    await search('Mateo')

    const notice = await screen.findByText(/not enough history/i)
    expect(notice).toHaveTextContent('10 recorded years')
    expect(notice).not.toHaveTextContent(/not in current use/i)
  })
})

describe('SearchPage headline statistics', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  it('describes the most recent recorded year and names the comparison year', async () => {
    // Recorded in 1990 and then not again until 1993: the comparison is with
    // 1990, not "the prior year", and the figures are identical between them.
    const years = [...Array.from({ length: 10 }, (_, i) => 1980 + i), 1993]
    const history: NameRow[] = years.map((year) => ({
      name: 'Debra',
      sex: 'F',
      year,
      total_count: 1200,
      popularity_percent: 0.0012,
      popularity_rank: 40,
    }))
    getNameHistory.mockResolvedValue({ name: 'Debra', sex: 'F', history })
    getNameForecast.mockResolvedValue(emptyForecast('Debra', history))

    await search('Debra')

    const caption = await screen.findByText(/most recent recorded year/i)
    expect(caption).toHaveTextContent('1993')
    expect(caption).toHaveTextContent('1989')
    // "vs prior year" hides a nine-year gap; the year must be named.
    expect(screen.queryByText(/prior year/i)).toBeNull()
    expect(screen.getAllByText(/unchanged vs 1989/i).length).toBeGreaterThan(0)
  })
})

describe('SearchPage trend chart', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  /** Number of separate segments in the historical line's SVG path. */
  function historyLineSegments(): number {
    const path = document.querySelector('.recharts-line-curve')?.getAttribute('d') ?? ''
    return (path.match(/M/g) ?? []).length
  }

  it('breaks the line across years with no recorded births before the forecast arrives', async () => {
    const history = historyFor('Debra', [1980, 1981, 1982, 1990, 1991, 1992])
    getNameHistory.mockResolvedValue({ name: 'Debra', sex: 'F', history })
    getNameForecast.mockReturnValue(new Promise(() => {})) // never settles

    await search('Debra')

    await screen.findByLabelText(/trend and forecast for Debra/i)
    await waitFor(() => expect(historyLineSegments()).toBe(2))
  })
})

describe('SearchPage trend chart once the forecast has loaded', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  it('still breaks the line across years with no recorded births', async () => {
    const years = [2010, 2011, 2012, 2021, 2022, 2023, 2024, 2025]
    const history = historyFor('Luna', years)
    getNameHistory.mockResolvedValue({ name: 'Luna', sex: 'F', history })
    getNameForecast.mockResolvedValue({
      ...emptyForecast('Luna', history),
      forecast: [2025, 2026, 2027, 2028, 2029].map((year) => ({
        year,
        mean: 0.002,
        lo80: 0.0015,
        hi80: 0.0025,
        lo95: 0.001,
        hi95: 0.003,
      })),
    })

    await search('Luna')

    await screen.findByLabelText(/trend and forecast for Luna/i)
    await waitFor(() => {
      const path = document.querySelector('.recharts-line-curve')?.getAttribute('d') ?? ''
      expect((path.match(/M/g) ?? []).length).toBe(2)
    })
  })
})

describe('SearchPage validation panel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  it('shows validation figures in readable units rather than scientific notation', async () => {
    const years = Array.from({ length: 15 }, (_, i) => 2010 + i)
    const history = historyFor('Emma', years)
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    getNameForecast.mockResolvedValue(fullForecast('Emma', history))

    await search('Emma')

    const heading = await screen.findByText('Holdout validation')
    const card = heading.parentElement as HTMLElement
    expect(card).toBeTruthy()
    expect(card.textContent).not.toMatch(/e[+-]\d/i)
  })

  it("reports the model's skill against the naive no-change baseline", async () => {
    const years = Array.from({ length: 15 }, (_, i) => 2010 + i)
    const history = historyFor('Emma', years)
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    getNameForecast.mockResolvedValue(fullForecast('Emma', history, { skill: 0.25 }))

    await search('Emma')

    // Scoped to the validation panel: the per-name panel states skill too.
    const panel = (await screen.findByText('Holdout validation')).closest('div')!
    expect(within(panel).getByText(/no change/i)).toHaveTextContent('25.0%')
  })

  it('flags a forecast that performs worse than the naive baseline', async () => {
    const years = Array.from({ length: 15 }, (_, i) => 2010 + i)
    const history = historyFor('Emma', years)
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    getNameForecast.mockResolvedValue(fullForecast('Emma', history, { skill: -0.1 }))

    await search('Emma')

    const panel = (await screen.findByText('Holdout validation')).closest('div')!
    expect(within(panel).getByText(/worse than.*no change/i)).toHaveTextContent('10.0%')
  })
})

describe('SearchPage track record', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  async function searchEmma(overrides: Parameters<typeof fullForecast>[2] = {}) {
    const years = Array.from({ length: 40 }, (_, i) => 1986 + i) // ends 2025
    const history = historyFor('Emma', years)
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    getNameForecast.mockResolvedValue(fullForecast('Emma', history, overrides))
    await search('Emma')
    const table = await screen.findByRole('table', { name: /year-by-year/i })
    await within(table).findByRole('columnheader', { name: /error/i })
    return { history, table }
  }

  function rowFor(table: HTMLElement, year: number): HTMLElement {
    return within(table).getByText(String(year)).closest('tr')!
  }

  /** The error column as numbers, for every row that has one. */
  function visibleErrors(table: HTMLElement): number[] {
    return within(table)
      .getAllByRole('row')
      .slice(1)
      .map((row) => within(row).getAllByRole('cell').slice(-1)[0].textContent ?? '')
      .filter((text) => text !== '')
      .map((text) => Number(text.replace('\u2212', '-').replace('%', '')))
  }

  it('opens on what the model said one year ahead', async () => {
    // One year out is the question most visitors are asking, and the one a
    // reader can interpret first. See
    // docs/adr/0012-a-track-record-replaces-the-holdout-on-the-page.md.
    const { table } = await searchEmma()

    const selector = screen.getByRole('radiogroup', { name: /years ahead/i })
    expect(within(selector).getByRole('radio', { name: '1 year' })).toBeChecked()
    expect(rowFor(table, 2011)).toHaveTextContent('+2.0%')
  })

  it('shows, for each year the model can be checked on, what it said and how wrong it was', async () => {
    // Error is relative to what happened, so a miss on a common name and a
    // miss on a rare one read on the same scale: (model − actual) / actual.
    const { history, table } = await searchEmma()

    const odd = history.find((row) => row.year === 2011)!
    expect(rowFor(table, 2011)).toHaveTextContent(formatPercent(odd.popularity_percent * 1.02, 4))
    expect(rowFor(table, 2011)).toHaveTextContent('+2.0%')

    const even = history.find((row) => row.year === 2012)!
    expect(rowFor(table, 2012)).toHaveTextContent(formatPercent(even.popularity_percent * 0.99, 4))
    expect(rowFor(table, 2012)).toHaveTextContent('\u22121.0%')
  })

  it('colours each error by the direction it missed in, and defines the column', async () => {
    // "−1.0%" beside a share column that is also a percentage is ambiguous
    // without saying what it is a percentage of.
    const { table } = await searchEmma()

    const high = within(rowFor(table, 2011)).getByText('+2.0%')
    const alsoHigh = within(rowFor(table, 2013)).getByText('+2.0%')
    const low = within(rowFor(table, 2012)).getByText('\u22121.0%')
    expect(high.className).toBe(alsoHigh.className)
    expect(low.className).not.toBe(high.className)

    expect(screen.getByText(/as a share of what actually happened/i)).toBeInTheDocument()
  })

  it('prints the projected rank next to the rank the year actually recorded', async () => {
    // Adjacent columns, because that is the comparison a reader is making:
    // "it said 47th, it was 46th". Which is also why the rank the model is
    // credited with has to be ranked against the whole field rather than
    // against the forecastable names — a systematic gap between two adjacent
    // columns reads as the model being wrong, not as a ranking artifact. See
    // docs/adr/0013-projected-rank-against-a-frozen-field.md.
    const { history, table } = await searchEmma()

    const headers = within(table)
      .getAllByRole('columnheader')
      .map((header) => header.textContent)
    expect(headers.indexOf('Projected rank')).toBe(headers.indexOf('Rank') + 1)

    const row = history.find((entry) => entry.year === 2011)!
    const [rank, projectedRank] = within(rowFor(table, 2011)).getAllByRole('cell').slice(3, 5)
    expect(rank).toHaveTextContent(String(row.popularity_rank))
    expect(projectedRank).toHaveTextContent(String(row.popularity_rank + 1))
  })

  it('moves the projected rank with the horizon selector', async () => {
    // The fixture misses the rank by exactly the horizon, so the column says
    // which horizon produced it.
    const { history, table } = await searchEmma()
    const user = userEvent.setup()

    await user.click(screen.getByRole('radio', { name: '5 years' }))

    const row = history.find((entry) => entry.year === 2011)!
    const projectedRank = within(rowFor(table, 2011)).getAllByRole('cell')[4]
    expect(projectedRank).toHaveTextContent(String(row.popularity_rank + 5))
  })

  it('leaves a year the model was never checked on blank rather than zero', async () => {
    // "Not measured" and "measured as nothing" are different facts. 1995 is
    // before the first origin could be scored; 2005 is a gap inside the record,
    // an origin at which this name was not eligible.
    const years = Array.from({ length: 40 }, (_, i) => 1986 + i)
    const record = historyFor('Emma', years)
      .filter((row) => row.year >= 2000 && row.year !== 2005)
      .map((row) => ({
        year: row.year,
        projected_share: row.popularity_percent,
        projected_rank: row.popularity_rank,
      }))
    const { table } = await searchEmma({ trackRecord: { '1': record } })

    for (const year of [1995, 2005]) {
      const cells = within(rowFor(table, year)).getAllByRole('cell')
      const [projectedRank, projected, error] = cells.slice(-3)
      expect(projectedRank).toHaveTextContent(/^$/)
      expect(projected).toHaveTextContent(/^$/)
      expect(error).toHaveTextContent(/^$/)
    }
    const checked = within(rowFor(table, 2006)).getAllByRole('cell').slice(-1)[0]
    expect(checked).toHaveTextContent('+0.0%')
  })

  it('summarises accuracy with figures a reader can check against the error column', async () => {
    // Derived from the rows on screen, never stored, so averaging the column
    // by hand gives exactly what the summary says. One year ahead, 30 years
    // from 1996: 15 odd years 2% high and 15 even years 1% low, so the typical
    // miss is 1.5% and on average it ran 0.5% high.
    const { table } = await searchEmma()

    const errors = visibleErrors(table)
    expect(errors).toHaveLength(30)
    const meanMiss = errors.reduce((sum, e) => sum + Math.abs(e), 0) / errors.length
    const meanError = errors.reduce((sum, e) => sum + e, 0) / errors.length

    const summary = screen.getByRole('group', { name: /accuracy/i })
    expect(summary).toHaveTextContent('30 years')
    expect(within(summary).getByText(`${meanMiss.toFixed(1)}%`)).toBeInTheDocument()
    expect(within(summary).getByText(`+${meanError.toFixed(1)}%`)).toBeInTheDocument()
  })

  it('moves the projected share, the error column and the summary together when the horizon changes', async () => {
    const { history, table } = await searchEmma()
    const user = userEvent.setup()

    await user.click(screen.getByRole('radio', { name: '5 years' }))

    const odd = history.find((row) => row.year === 2011)!
    expect(rowFor(table, 2011)).toHaveTextContent(formatPercent(odd.popularity_percent * 1.1, 4))
    expect(rowFor(table, 2011)).toHaveTextContent('+10.0%')
    // Checked one year ahead, but nothing was predicted five years before 1998.
    const [projected, error] = within(rowFor(table, 1998)).getAllByRole('cell').slice(-2)
    expect(projected).toHaveTextContent(/^$/)
    expect(error).toHaveTextContent(/^$/)
    expect(screen.getByRole('group', { name: /5 years ahead/i })).toHaveTextContent('26 years')
    expect(screen.getByText(/for each year 5 years before it/i)).toBeInTheDocument()
  })

  it('summarises exactly the visible error column at every horizon', async () => {
    const { table } = await searchEmma()
    const user = userEvent.setup()

    for (const h of [1, 2, 3, 4, 5]) {
      await user.click(screen.getByRole('radio', { name: h === 1 ? '1 year' : `${h} years` }))

      const errors = visibleErrors(table)
      expect(errors).toHaveLength(31 - h)
      const meanMiss = errors.reduce((sum, e) => sum + Math.abs(e), 0) / errors.length
      const meanError = errors.reduce((sum, e) => sum + e, 0) / errors.length

      const summary = screen.getByRole('group', { name: /accuracy/i })
      expect(summary).toHaveTextContent(`${errors.length} years`)
      expect(within(summary).getByText(`${meanMiss.toFixed(1)}%`)).toBeInTheDocument()
      expect(within(summary).getByText(`+${meanError.toFixed(1)}%`)).toBeInTheDocument()
    }
  })

  it('lets the horizon be chosen from the keyboard', async () => {
    await searchEmma()
    const user = userEvent.setup()

    screen.getByRole('radio', { name: '1 year' }).focus()
    await user.keyboard('{ArrowRight}')

    expect(screen.getByRole('radio', { name: '2 years' })).toBeChecked()
    expect(screen.getByRole('group', { name: /2 years ahead/i })).toBeInTheDocument()
  })

  it('shows a short record as short, and says so when a horizon was never checked', async () => {
    // First eligible at origin 2021, after the scored span: it can be checked
    // one to four years ahead, and never five. No row is made up to fill that.
    const years = Array.from({ length: 40 }, (_, i) => 1986 + i)
    const record = Object.fromEntries(
      [1, 2, 3, 4].map((h) => [
        String(h),
        historyFor('Emma', years)
          .filter((row) => row.year >= 2021 + h)
          .map((row) => ({
            year: row.year,
            projected_share: row.popularity_percent,
            projected_rank: row.popularity_rank,
          })),
      ])
    )
    const { table } = await searchEmma({ trackRecord: record })
    const user = userEvent.setup()

    expect(visibleErrors(table)).toHaveLength(4)

    await user.click(screen.getByRole('radio', { name: '5 years' }))

    expect(visibleErrors(table)).toHaveLength(0)
    expect(screen.queryByRole('group', { name: /accuracy/i })).not.toBeInTheDocument()
    expect(screen.getByText(/not yet been checked 5 years ahead/i)).toBeInTheDocument()
  })
})

describe('SearchPage interval labels', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  it('labels shaded bands with their measured coverage, not the nominal level', async () => {
    const years = Array.from({ length: 15 }, (_, i) => 2010 + i)
    const history = historyFor('Emma', years)
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    getNameForecast.mockResolvedValue(
      fullForecast('Emma', history, { empirical80: 0.44, empirical95: 0.51 })
    )

    await search('Emma')

    await screen.findByLabelText(/trend and forecast for Emma/i)
    expect(screen.queryByText(/95% interval/i)).toBeNull()
    expect(screen.queryByText(/80% interval/i)).toBeNull()
    expect(await screen.findByText(/51% interval/i)).toBeTruthy()
    expect(await screen.findByText(/44% interval/i)).toBeTruthy()
  })

  it('reports the coverage measured for this name\u2019s own stratum', async () => {
    // Two names at the same popularity tier but in different volatility bins
    // carry different calibration rows, because the bands they were given
    // were built from different residuals. The legend has to follow the row
    // it was handed rather than any single figure, or the whole point of
    // keying calibration by stratum is lost between the API and the chart.
    const years = Array.from({ length: 15 }, (_, i) => 2010 + i)
    const history = historyFor('Olivia', years)
    getNameHistory.mockResolvedValue({ name: 'Olivia', sex: 'F', history })
    getNameForecast.mockResolvedValue(
      fullForecast('Olivia', history, {
        tier: 'top100',
        volatilityBin: 2,
        empirical80: 0.79,
        empirical95: 0.94,
      })
    )

    await search('Olivia')

    await screen.findByLabelText(/trend and forecast for Olivia/i)
    expect(await screen.findByText(/79% interval/i)).toBeTruthy()
    expect(await screen.findByText(/94% interval/i)).toBeTruthy()
    expect(screen.queryByText(/44% interval/i)).toBeNull()
    expect(screen.queryByText(/51% interval/i)).toBeNull()
  })
})

describe('SearchPage forecast presentation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  async function searchEmma(overrides: Parameters<typeof fullForecast>[2] = {}) {
    const years = Array.from({ length: 40 }, (_, i) => 1986 + i) // ends 2025
    const history = historyFor('Emma', years)
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    getNameForecast.mockResolvedValue(fullForecast('Emma', history, overrides))
    await search('Emma')
    await screen.findByLabelText(/trend and forecast for Emma/i)
    return history
  }

  /** recharts labels each drawn curve with its series name. */
  function curve(selector: string): SVGElement {
    const element = document.querySelector(selector)
    if (!element) throw new Error(`nothing drawn for ${selector}`)
    return element as SVGElement
  }

  const attr = (element: SVGElement, name: string) => Number(element.getAttribute(name))

  it('draws the band as the primary object and the forecast line beneath it', async () => {
    // The band is the honest object: the point forecast is one path through
    // it rather than a trajectory. Reading that line as exact is the mistake
    // the parent PRD is trying to stop, so it is thinner and dimmer than the
    // history it continues, and the band is more present than it is.
    await searchEmma()

    const forecast = curve('.recharts-line-curve[name="Forecast"]')
    const history = curve('.recharts-line-curve[name="Historical"]')
    const bands = [...document.querySelectorAll('.recharts-area-area')] as SVGElement[]

    expect(attr(forecast, 'stroke-width')).toBeLessThan(attr(history, 'stroke-width'))
    expect(attr(forecast, 'stroke-opacity')).toBeLessThan(1)
    // The two bands read as one shaded mass with the likelier interval
    // denser inside the wider one, rather than as two faint outlines.
    const [outer, inner] = bands.map((band) => attr(band, 'fill-opacity'))
    expect(inner).toBeGreaterThan(outer)
    expect(inner).toBeGreaterThanOrEqual(0.25)
  })

  it('paints the bands beneath the lines, whatever order the legend reads in', async () => {
    // SVG has no z-index: whatever is painted later covers what came before.
    await searchEmma()

    const bands = [...document.querySelectorAll('.recharts-area-area')]
    const lines = [...document.querySelectorAll('.recharts-line-curve')]
    expect(bands).toHaveLength(2)
    for (const band of bands) {
      for (const line of lines) {
        expect(band.compareDocumentPosition(line) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
      }
    }
  })

  it('names the forecast line in the legend without grading it', async () => {
    // The legend names things; how well the model has done on this name is
    // reported on the page, where a losing skill is still stated plainly.
    await searchEmma({ skill: -0.1 })

    expect(legendLabels()).toContain('Forecast')
    const legend = document.querySelector('.recharts-legend-wrapper')!
    expect(legend.textContent).not.toMatch(/no change|better|worse|skill/i)
  })

  const legendLabels = () =>
    [...document.querySelectorAll('.recharts-legend-item-text')].map((item) => item.textContent)

  it('reads the legend in the order the chart is understood', async () => {
    // History, then the line that continues it, then the likelier interval
    // before the wider one — whatever order the series are painted in.
    await searchEmma({ empirical80: 0.44, empirical95: 0.51 })

    expect(legendLabels()).toEqual(['Historical', 'Forecast', '44% interval', '51% interval'])
  })

  /** How opaque an element's fill renders: its colour's alpha times fill-opacity. */
  function fillAlpha(element: Element): number {
    const fill = element.getAttribute('fill') ?? ''
    const rgba = fill.match(/rgba\([^)]*,\s*([\d.]+)\)/)
    const colourAlpha = rgba ? Number(rgba[1]) : 1
    const opacity = element.getAttribute('fill-opacity')
    return colourAlpha * (opacity === null ? 1 : Number(opacity))
  }

  it("gives each interval's swatch the opacity its band actually renders at", async () => {
    // The inner band is painted over the outer one, so where it sits the two
    // composite: what a visitor sees there is not the inner band's own alpha.
    await searchEmma({ empirical80: 0.44, empirical95: 0.51 })

    const outerBand = fillAlpha(curve('.recharts-area-area[name="51% interval"]'))
    const innerBand = fillAlpha(curve('.recharts-area-area[name="44% interval"]'))
    const rendered = { outer: outerBand, inner: 1 - (1 - outerBand) * (1 - innerBand) }

    const swatch = (label: string) =>
      fillAlpha(
        screen.getByLabelText(`${label} legend icon`).querySelector('.recharts-legend-icon')!
      )
    expect(swatch('51% interval')).toBeCloseTo(rendered.outer, 3)
    expect(swatch('44% interval')).toBeCloseTo(rendered.inner, 3)
  })

  /**
   * Hover the plot above a year and return the tooltip it opens. The chart is
   * 800 wide; the y-axis takes the first 72px and the right margin the last
   * 16, so years spread evenly across the rest.
   */
  async function hoverYear(year: number, [first, last]: [number, number]) {
    const plotLeft = 72
    const plotRight = 784
    const x = plotLeft + ((year - first) / (last - first)) * (plotRight - plotLeft)
    fireEvent.mouseMove(document.querySelector('.recharts-wrapper')!, { clientX: x, clientY: 200 })
    return waitFor(() => {
      const tooltip = document.querySelector('.recharts-tooltip-wrapper')
      if (!tooltip?.textContent?.includes(String(year))) throw new Error(`no tooltip for ${year}`)
      return tooltip as HTMLElement
    })
  }

  it('states the projected value and both ranges when a forecast year is hovered', async () => {
    // The band carries the uncertainty, and a shaded region is unreadable to
    // anyone who cannot interpret one: hovering says it in numbers.
    await searchEmma()

    const tooltip = await hoverYear(2028, [1986, 2030])

    expect(tooltip).toHaveTextContent('0.2000%')
    expect(tooltip).toHaveTextContent('0.1500% – 0.2500%')
    expect(tooltip).toHaveTextContent('0.1000% – 0.3000%')
  })

  it("names each range by the coverage measured for this name's stratum", async () => {
    // Without its measured coverage the two ranges differ only by width, and
    // the reader supplies a confidence level of their own. See
    // docs/adr/0011-conformal-bands-keyed-by-strata.md.
    await searchEmma({ empirical80: 0.79, empirical95: 0.94 })

    const tooltip = await hoverYear(2028, [1986, 2030])

    expect(tooltip).toHaveTextContent('79% interval: 0.1500% – 0.2500%')
    expect(tooltip).toHaveTextContent('94% interval: 0.1000% – 0.3000%')
    expect(tooltip).not.toHaveTextContent(/80%|95%/)
  })

  it('makes no accuracy claim when a forecast year is hovered', async () => {
    // Skill is reported on the page; the tooltip identifies, it does not grade.
    await searchEmma({ skill: -0.1 })

    const tooltip = await hoverYear(2028, [1986, 2030])

    expect(tooltip).not.toHaveTextContent(/no change|better|worse|skill|accura/i)
  })

  it('states the recorded value when a recorded year is hovered', async () => {
    const history = await searchEmma()

    const tooltip = await hoverYear(2000, [1986, 2030])

    const recorded = history.find((row) => row.year === 2000)!.popularity_percent
    expect(tooltip).toHaveTextContent(formatPercent(recorded, 4))
  })

  it('reports the last recorded year as recorded, not as the start of the forecast', async () => {
    // The forecast line is drawn back to the last observed point so the two
    // lines meet; that join is not a projection of a year already recorded.
    const history = await searchEmma()

    const tooltip = await hoverYear(2025, [1986, 2030])

    expect(tooltip).toHaveTextContent(formatPercent(history.at(-1)!.popularity_percent, 4))
    expect(tooltip).not.toHaveTextContent(/forecast|interval/i)
  })

  it('writes a rare name’s share in fixed decimals, never scientific notation', async () => {
    const years = Array.from({ length: 40 }, (_, i) => 1986 + i)
    const history = historyFor('Emma', years).map((row) => ({ ...row, popularity_percent: 3e-8 }))
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    getNameForecast.mockResolvedValue(fullForecast('Emma', history))
    await search('Emma')
    await screen.findByLabelText(/trend and forecast for Emma/i)

    const tooltip = await hoverYear(2000, [1986, 2030])

    expect(tooltip).toHaveTextContent('0.0000%')
    expect(tooltip.textContent).not.toMatch(/\de[-+]?\d/i)
  })

  /** Years printed along the horizontal axis. */
  function xAxisYears(): number[] {
    return [
      ...document.querySelectorAll(
        '.recharts-xAxis-tick-labels .recharts-cartesian-axis-tick-value'
      ),
    ].map((tick) => Number(tick.textContent))
  }

  /** Press on one year, move to another and release, as a visitor dragging across the plot. */
  async function dragAcross(from: number, to: number, range: [number, number]) {
    const plot = document.querySelector('.recharts-wrapper')!
    await hoverYear(from, range)
    fireEvent.mouseDown(plot)
    await hoverYear(to, range)
    fireEvent.mouseUp(plot)
  }

  it('offers no reset control until the chart is zoomed', async () => {
    await searchEmma()

    expect(screen.queryByRole('button', { name: /reset zoom/i })).not.toBeInTheDocument()
  })

  it('zooms to the years dragged across, and resets to the full range', async () => {
    // 145 years in 440 pixels leave the recent decades a few pixels wide.
    await searchEmma()

    await dragAcross(2000, 2010, [1986, 2030])

    await waitFor(() => {
      const years = xAxisYears()
      expect(years.length).toBeGreaterThan(0)
      expect(Math.min(...years)).toBeGreaterThanOrEqual(2000)
      expect(Math.max(...years)).toBeLessThanOrEqual(2010)
    })

    await userEvent.click(screen.getByRole('button', { name: /reset zoom/i }))

    await waitFor(() => expect(Math.min(...xAxisYears())).toBeLessThan(2000))
    expect(screen.queryByRole('button', { name: /reset zoom/i })).not.toBeInTheDocument()
  })

  it('zooms in when the wheel turns over the plot', async () => {
    await searchEmma()

    await hoverYear(2020, [1986, 2030])
    fireEvent.wheel(screen.getByLabelText(/trend and forecast for Emma/i), { deltaY: -100 })

    await waitFor(() => expect(Math.min(...xAxisYears())).toBeGreaterThan(1986))
    expect(screen.getByRole('button', { name: /reset zoom/i })).toBeInTheDocument()
  })

  it('labels a zoomed vertical axis finely enough to tell its ticks apart', async () => {
    // Zoomed into a decade the shares differ in the third decimal place; at
    // two decimals every tick would read the same.
    await searchEmma()

    await dragAcross(2000, 2003, [1986, 2030])

    await waitFor(() => {
      const labels = [
        ...document.querySelectorAll(
          '.recharts-yAxis-tick-labels .recharts-cartesian-axis-tick-value'
        ),
      ].map((tick) => tick.textContent)
      expect(labels.length).toBeGreaterThan(1)
      expect(new Set(labels).size).toBe(labels.length)
    })
  })

  it('draws no point markers on the forecast line', async () => {
    // A dot per year reads as five measurements. They are not measurements.
    // recharts hoists dots out of their series' layer, so they are identified
    // by the colour the forecast series is drawn in.
    await searchEmma()

    const dots = [...document.querySelectorAll('.recharts-dot')]
    expect(dots.filter((dot) => dot.getAttribute('fill') === CHART_COLORS.forecast)).toHaveLength(0)
  })
})

describe('SearchPage forecast table', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  async function searchEmma(overrides: Parameters<typeof fullForecast>[2] = {}) {
    const years = Array.from({ length: 40 }, (_, i) => 1986 + i) // ends 2025
    const history = historyFor('Emma', years)
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    const payload = fullForecast('Emma', history, overrides)
    getNameForecast.mockResolvedValue(payload)
    await search('Emma')
    return payload
  }

  it('lists each forecast year with its projected share and likely range', async () => {
    // The chart made legible to someone who cannot read a shaded region: the
    // range is the narrower band, because one range per row is all a reader
    // needs and the wider one is what the chart is for.
    const payload = await searchEmma()

    const table = await screen.findByRole('table', { name: /forecast/i })
    const rows = within(table).getAllByRole('row').slice(1) // header row first
    expect(rows).toHaveLength(payload.forecast.length)
    payload.forecast.forEach((point, i) => {
      expect(rows[i]).toHaveTextContent(String(point.year))
      expect(rows[i]).toHaveTextContent(formatPercent(point.mean, 4))
      expect(rows[i]).toHaveTextContent(formatPercent(point.lo80, 4))
      expect(rows[i]).toHaveTextContent(formatPercent(point.hi80, 4))
      expect(rows[i]).not.toHaveTextContent(formatPercent(point.lo95, 4))
    })
  })

  it('answers "will it still be in the top ten?" with a projected rank per year', async () => {
    // The question a parent actually has. The model predicts a share of
    // births, so the rank beside it is computed in the batch against the whole
    // field observed at the origin — the page only prints what it was given.
    // See docs/adr/0013-projected-rank-against-a-frozen-field.md.
    const payload = await searchEmma()

    const table = await screen.findByRole('table', { name: /forecast/i })
    expect(within(table).getByRole('columnheader', { name: /projected rank/i })).toBeInTheDocument()
    const rows = within(table).getAllByRole('row').slice(1)
    payload.forecast.forEach((point, i) => {
      expect(rows[i]).toHaveTextContent(String(point.projected_rank))
    })
  })

  it('heads the range with the coverage measured for this name, never the nominal level', async () => {
    // A name whose own cell was too thin is served the population's band, and
    // its calibration row names `*`. The heading follows the row it was
    // handed, whichever stratum that row describes. See
    // docs/adr/0011-conformal-bands-keyed-by-strata.md.
    await searchEmma({
      stratum: { tier: 'top1000', volatility_bin: 2 },
      tier: '*',
      volatilityBin: -1,
      empirical80: 0.776,
    })

    const table = await screen.findByRole('table', { name: /forecast/i })
    const heading = within(table).getByRole('columnheader', { name: /likely range/i })
    expect(heading).toHaveTextContent('78%')
    expect(heading).not.toHaveTextContent('80%')
  })

  it('renders no forecast table for a name with no forecast', async () => {
    const years = Array.from({ length: 34 }, (_, i) => 1960 + i) // ends 1993
    const history = historyFor('Debra', years)
    getNameHistory.mockResolvedValue({ name: 'Debra', sex: 'F', history })
    getNameForecast.mockResolvedValue(emptyForecast('Debra', history))

    await search('Debra')

    await screen.findByText(/not in current use/i)
    expect(screen.getByRole('table', { name: /year-by-year/i })).toBeInTheDocument()
    expect(screen.queryByRole('table', { name: /forecast/i })).toBeNull()
  })

  it('scrolls within its own container rather than widening the page', async () => {
    // jsdom lays nothing out, so the observable part of "readable on a narrow
    // screen" is that the table sits in a box that scrolls horizontally.
    await searchEmma()

    const table = await screen.findByRole('table', { name: /forecast/i })
    expect(table.parentElement).toHaveClass('overflow-x-auto')
  })
})

describe('SearchPage per-name forecast attributes', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  /** History that peaked mid-way and has since halved, so "below its peak"
   *  has an answer worth printing. */
  function peakedHistory(): NameRow[] {
    const years = Array.from({ length: 40 }, (_, i) => 1986 + i) // ends 2025
    return years.map((year, i) => ({
      name: 'Emma',
      sex: 'F' as const,
      year,
      total_count: 1000,
      popularity_percent: year <= 2005 ? 0.0001 * (i + 1) : 0.002 - 0.00005 * (i - 19),
      popularity_rank: 50,
    }))
  }

  async function searchWith(history: NameRow[], overrides = {}) {
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    getNameForecast.mockResolvedValue(fullForecast('Emma', history, overrides))
    await search('Emma')
    return (await screen.findByText('What drives this forecast')).closest('div') as HTMLElement
  }

  it("reports this name's measured skill and how many windows stand behind it", async () => {
    const panel = await searchWith(peakedHistory(), { skill: 0.25 })

    expect(within(panel).getByText(/25\.0%/)).toBeInTheDocument()
    expect(panel.textContent).toContain('26')
  })

  it("names this name's popularity tier and volatility bin, not the band's", async () => {
    // A name whose own cell was too thin is served the population's band, and
    // the calibration row then says `*`. Printing that as the name's tier
    // would be false, so the panel reads the name's own stratum.
    const panel = await searchWith(peakedHistory(), {
      stratum: { tier: 'top1000', volatility_bin: 2 },
      tier: '*',
      volatilityBin: -1,
    })

    expect(panel.textContent).toMatch(/top 1,?000/i)
    expect(panel.textContent).toMatch(/jumpiest/i)
    expect(panel.textContent).not.toContain('*')
  })

  it('reports how wide the band it was given actually is', async () => {
    // lo95 0.001 to hi95 0.003 at the last horizon: a factor of three.
    const panel = await searchWith(peakedHistory())

    expect(within(panel).getByText(/3\.0×/)).toBeInTheDocument()
  })

  it('places the name against its own historical peak', async () => {
    // The pooled model reads distance-below-peak and years-since-peak as
    // features, so the panel says what those features see.
    const history = peakedHistory()
    const panel = await searchWith(history)

    const peak = history.reduce((a, b) => (b.popularity_percent > a.popularity_percent ? b : a))
    expect(panel.textContent).toContain(String(peak.year))
  })

  it('sits alongside the global model card', async () => {
    await searchWith(peakedHistory())

    expect(screen.getByText('How the forecast is made')).toBeInTheDocument()
    expect(screen.getByText('What drives this forecast')).toBeInTheDocument()
  })
})

describe('SearchPage pooled model', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  async function searchWithForecast() {
    const years = Array.from({ length: 40 }, (_, i) => 1986 + i) // ends 2025
    const history = historyFor('Emma', years)
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    getNameForecast.mockResolvedValue(fullForecast('Emma', history))
    await search('Emma')
    await waitFor(() => expect(getNameForecast).toHaveBeenCalled())
    return history
  }

  it('names the model that actually produced the line, not ARIMA', async () => {
    await searchWithForecast()

    await waitFor(() => expect(screen.getByText(MODEL_CARD.model_name)).toBeInTheDocument())
    expect(document.body.textContent).not.toMatch(/ARIMA/)
  })

  it('describes the pooled model instead of a per-name fit', async () => {
    await searchWithForecast()

    await waitFor(() => expect(screen.getByText('How the forecast is made')).toBeInTheDocument())
    expect(screen.getByText(MODEL_CARD.model_name)).toBeInTheDocument()
    // Trained across every name's history, not this one's thirty points.
    expect(screen.getByText(/638,412/)).toBeInTheDocument()
    expect(screen.getByText(/91 origins/)).toBeInTheDocument()
    for (const feature of MODEL_CARD.features) {
      expect(screen.getByText(feature)).toBeInTheDocument()
    }
    expect(screen.queryByText(/Ljung/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Jarque/)).not.toBeInTheDocument()
  })

  it('reports the newest year as recorded history, not as a forecast', async () => {
    const history = await searchWithForecast()

    // The year-by-year table is the record; 2025 has to appear in it with the
    // share that was actually observed.
    const newest = history[history.length - 1]
    await waitFor(() => expect(screen.getByText('Forecast →')).toBeInTheDocument())
    const table = screen.getByRole('table', { name: /year-by-year/i })
    const row = within(table).getByText('2025').closest('tr')
    expect(row).not.toBeNull()
    expect(row!.textContent).toContain(formatPercent(newest.popularity_percent, 4))
  })
})

describe('SearchPage submission', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  // A bare <button> in a form submits it, and pressing Enter in a text field
  // activates the form's *first* submit button rather than the one the visitor
  // would have clicked. With the sex toggle sitting between the name field and
  // Search, that meant Enter picked "Female": the toggle flipped and the
  // results shown were for the sex that was no longer selected.
  it('searches the sex that is selected, whichever way the form is submitted', async () => {
    const years = Array.from({ length: 40 }, (_, i) => 1986 + i)
    const history = historyFor('Liam', years)
    getNameHistory.mockResolvedValue({ name: 'Liam', sex: 'M', history })
    getNameForecast.mockResolvedValue(emptyForecast('Liam', history))

    const user = userEvent.setup()
    render(<SearchPage />)
    await user.click(screen.getByRole('radio', { name: 'Male' }))
    await user.type(screen.getByLabelText('Name'), 'Liam{Enter}')

    await waitFor(() => expect(getNameHistory).toHaveBeenCalledWith('Liam', 'M'))
    expect(getNameForecast).toHaveBeenCalledWith('Liam', 'M')
    expect(screen.getByRole('radio', { name: 'Male' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('radio', { name: 'Female' })).toHaveAttribute('aria-checked', 'false')
  })
})

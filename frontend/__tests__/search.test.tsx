import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import { cloneElement, type ReactElement } from 'react'
import userEvent from '@testing-library/user-event'
import type { NameRow } from '@/lib/api'
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
  } = {}
) {
  const {
    skill = 0.25,
    empirical80 = 0.44,
    empirical95 = 0.51,
    tier = 'top100',
    volatilityBin = 1,
  } = overrides
  return {
    name,
    sex: 'F' as const,
    history: history.map((row) => ({ year: row.year, value: row.popularity_percent })),
    forecast: [2026, 2027, 2028, 2029, 2030].map((year) => ({
      year,
      mean: 0.002,
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
      points: [{ year: 2021, actual: 0.002, predicted: 0.0021 }],
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

    const notice = await screen.findByText(/no change/i)
    expect(notice).toHaveTextContent('25.0%')
  })

  it('flags a forecast that performs worse than the naive baseline', async () => {
    const years = Array.from({ length: 15 }, (_, i) => 2010 + i)
    const history = historyFor('Emma', years)
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    getNameForecast.mockResolvedValue(fullForecast('Emma', history, { skill: -0.1 }))

    await search('Emma')

    const notice = await screen.findByText(/worse than.*no change/i)
    expect(notice).toHaveTextContent('10.0%')
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

    await waitFor(() => expect(screen.getByText(/Pooled model forecast/)).toBeInTheDocument())
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
    const row = within(screen.getByRole('table')).getByText('2025').closest('tr')
    expect(row).not.toBeNull()
    expect(row!.textContent).toContain(formatPercent(newest.popularity_percent, 4))
  })
})

import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { cloneElement, type ReactElement } from 'react'
import userEvent from '@testing-library/user-event'
import type { NameRow } from '@/lib/api'

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

const NEWEST_YEAR = 2024

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

/** A forecast with validation, model diagnostics and measured calibration —
 * everything the "Holdout validation" panel and the chart's interval labels
 * read from. */
function fullForecast(
  name: string,
  history: NameRow[],
  overrides: { skill?: number; empirical80?: number; empirical95?: number } = {}
) {
  const { skill = 0.25, empirical80 = 0.44, empirical95 = 0.51 } = overrides
  return {
    name,
    sex: 'F' as const,
    history: history.map((row) => ({ year: row.year, value: row.popularity_percent })),
    forecast: [2025, 2026, 2027, 2028, 2029].map((year) => ({
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
      points: [{ year: 2020, actual: 0.002, predicted: 0.0021 }],
    },
    model: {
      order: [1, 1, 1],
      aic: 100,
      bic: 105,
      log_applied: false,
      diagnostics: {
        ljung_box: { p_value: 0.5, is_white_noise: true },
        normality: { p_value: 0.5, is_normal: true },
        heteroscedasticity: { p_value: 0.5, is_homoscedastic: true },
        overall_quality: true,
      },
      stationarity: { is_stationary: true, adf_pvalue: 0.01, kpss_pvalue: 0.5 },
    },
    calibration: {
      '0.8': { nominal: 0.8, empirical_coverage: empirical80, n: 45 },
      '0.95': { nominal: 0.95, empirical_coverage: empirical95, n: 45 },
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
    const years = Array.from({ length: 8 }, (_, i) => 2017 + i) // ends 2024
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
    const years = [2010, 2011, 2012, 2020, 2021, 2022, 2023, 2024]
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
})

import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
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
  } = {}
) {
  const {
    skill = 0.25,
    empirical80 = 0.44,
    empirical95 = 0.51,
    tier = 'top100',
    volatilityBin = 1,
    stratum = { tier, volatility_bin: volatilityBin },
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
      skill_windows: 26,
      points: [2021, 2022, 2023, 2024, 2025].map((year, i) => ({
        year,
        actual: 0.002 + i * 0.0001,
        predicted: 0.0021 + i * 0.0001,
      })),
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

    // Scoped to the validation panel: the chart legend now carries the same
    // comparison, in the shorter form the line is labelled with.
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

describe('SearchPage holdout validation table', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1960, max_year: NEWEST_YEAR })
  })

  it('shows what was predicted against what happened, year by year', async () => {
    // Three summary error figures are a claim; the window itself is the
    // evidence for it. Origin 2020, so the rows are the five observed years
    // 2021-2025 the model did not see.
    const years = Array.from({ length: 40 }, (_, i) => 1986 + i)
    const history = historyFor('Emma', years)
    getNameHistory.mockResolvedValue({ name: 'Emma', sex: 'F', history })
    const payload = fullForecast('Emma', history)
    getNameForecast.mockResolvedValue(payload)

    await search('Emma')

    // The window is 2021-2025, forecast from origin 2020 — say so, rather
    // than leaving "the 5 most recent years" to be counted off the rows.
    const panel = (await screen.findByText('Holdout validation')).closest('div')!
    expect(panel.textContent).toContain('2021')
    expect(panel.textContent).toContain('2025')
    expect(panel.textContent).toContain('2020')

    const table = await screen.findByRole('table', { name: /holdout/i })
    for (const point of payload.validation.points) {
      const row = within(table).getByText(String(point.year)).closest('tr')!
      expect(row.textContent).toContain(formatPercent(point.actual, 4))
      expect(row.textContent).toContain(formatPercent(point.predicted, 4))
    }
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

    const forecast = curve('.recharts-line-curve[name^="Pooled forecast"]')
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

  it("labels the forecast line with this name's measured skill", async () => {
    // The legend is where a visitor reads what the dashed line is. Naming the
    // model without saying how well it has done on *this* name invites the
    // line to be trusted uniformly, which is exactly what the measurements
    // say it should not be.
    await searchEmma({ skill: 0.25 })

    const label = curve('.recharts-line-curve[name^="Pooled forecast"]').getAttribute('name')!
    expect(label).toMatch(/25%/)
    expect(screen.getByText(label)).toBeInTheDocument()
  })

  it('says plainly on the line when the model loses to no-change on this name', async () => {
    await searchEmma({ skill: -0.1 })

    const label = curve('.recharts-line-curve[name^="Pooled forecast"]').getAttribute('name')!
    expect(label).toMatch(/worse than no change/i)
    expect(screen.getByText(label)).toBeInTheDocument()
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

    await waitFor(() => expect(screen.getByText(/^Pooled forecast/)).toBeInTheDocument())
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

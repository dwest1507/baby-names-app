export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api/${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
  } catch {
    throw new ApiError('Could not reach the server. Please check your connection.', 0)
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`
    try {
      const body = await response.json()
      if (typeof body?.detail === 'string') detail = body.detail
    } catch {
      // response body wasn't JSON; fall back to the generic message
    }
    throw new ApiError(detail, response.status)
  }

  return response.json() as Promise<T>
}

export interface YearRange {
  min_year: number
  max_year: number
}

export interface NameRow {
  name: string
  sex: 'M' | 'F'
  year: number
  total_count: number
  popularity_percent: number
  popularity_rank: number
}

export interface ValidationPoint {
  year: number
  actual: number
  predicted: number
}

export interface Validation {
  mae: number
  rmse: number
  mape: number
  // How much smaller this name's error was than a naive "no change"
  // baseline's — the last observed value repeated: 1 - model_mae / naive_mae.
  // 0 means no better than assuming nothing changed; negative means worse.
  //
  // Unlike the three figures above, it is not this holdout window's: it is
  // averaged over every five-year window the name was eligible for since
  // 1995 (26 of them on the 2025 database). One window would mostly measure
  // that window — the 2021-25 one contains the birth-rate shock — rather than
  // how predictable the name is.
  skill: number
  // How many of those windows stand behind `skill`. A name recorded since
  // 1995 has all 26; a recent arrival has a handful.
  skill_windows: number
  points: ValidationPoint[]
}

export interface CalibrationLevel {
  nominal: number
  // The stratum this coverage was measured over: the popularity tier and
  // volatility bin of the names it describes. `'*'` / -1 is the
  // whole-population fallback, served only for a name whose own stratum the
  // backtest never populated.
  tier: string
  volatility_bin: number
  empirical_coverage: number
  n: number
}

// Keyed by nominal level as a string ("0.8", "0.95"). Each row is the coverage
// the batch measured for names in *this* name's stratum, across every eligible
// name's holdout backtest — not a population average, which is exactly what
// conceals a badly calibrated tail. See
// docs/adr/0011-conformal-bands-keyed-by-strata.md. The shaded bands must be
// labelled with `empirical_coverage`, not `nominal`.
export type Calibration = Record<string, CalibrationLevel>

// One pooled model produces every name's forecast, so this describes the
// batch rather than the name being looked at: there is no per-name fit left to
// report an order or residual diagnostics for. See
// docs/adr/0010-a-pooled-model-replaces-per-name-arima.md.
export interface Model {
  model_name: string
  model_class: string
  // The quantity the model predicts, as an expression.
  target: string
  features: string[]
  horizons: number
  trained_through: number
  training_origins: number
  training_rows: number
  sample_weight: string
  seed: number
}

export interface ForecastPoint {
  year: number
  mean: number
  lo80: number
  hi80: number
  lo95: number
  hi95: number
}

export interface ForecastPayload {
  name: string
  sex: 'M' | 'F'
  history: { year: number; value: number }[]
  forecast: ForecastPoint[]
  validation: Validation | null
  model: Model | null
  calibration: Calibration | null
}

export interface ChatEntry {
  role: 'user' | 'assistant'
  content: string
  sql?: string | null
}

export function getMeta(): Promise<YearRange> {
  return request<YearRange>('meta')
}

export function getTopNames(
  sex: 'M' | 'F',
  year: number,
  limit: number
): Promise<{ names: NameRow[] }> {
  const params = new URLSearchParams({ sex, year: String(year), limit: String(limit) })
  return request<{ names: NameRow[] }>(`top-names?${params}`)
}

export function getNameHistory(
  name: string,
  sex: 'M' | 'F'
): Promise<{ name: string; sex: 'M' | 'F'; history: NameRow[] }> {
  const params = new URLSearchParams({ sex })
  return request(`names/${encodeURIComponent(name)}?${params}`)
}

export function getNameForecast(name: string, sex: 'M' | 'F'): Promise<ForecastPayload> {
  const params = new URLSearchParams({ sex })
  return request<ForecastPayload>(`names/${encodeURIComponent(name)}/forecast?${params}`)
}

export function postChat(
  message: string,
  history: ChatEntry[]
): Promise<{ answer: string; sql: string | null }> {
  return request('chat', {
    method: 'POST',
    body: JSON.stringify({ message, history }),
  })
}

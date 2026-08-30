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
  points: ValidationPoint[]
}

export interface DiagnosticTest {
  p_value: number
  is_white_noise?: boolean
  is_normal?: boolean
  is_homoscedastic?: boolean
}

export interface Model {
  order: number[]
  aic: number
  bic: number
  log_applied: boolean
  diagnostics: {
    ljung_box: { p_value: number; is_white_noise: boolean }
    normality: { p_value: number; is_normal: boolean }
    heteroscedasticity: { p_value: number; is_homoscedastic: boolean }
    overall_quality: boolean
  }
  stationarity: {
    is_stationary: boolean
    adf_pvalue: number
    kpss_pvalue: number
  }
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

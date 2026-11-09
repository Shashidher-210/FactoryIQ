export type Health = {
  status: 'ready' | 'degraded'
  model_ready: boolean
  model_version: string | null
  detail: string | null
}

export type VehicleListItem = {
  vehicle_id: number
  latest_time_step: number | null
  reading_count: number | null
  repair_probability: number
  risk_level: 'Low' | 'Medium' | 'High'
  predicted_class: 'no_repair' | 'repair'
  prediction_window: string
}

export type VehiclesResponse = {
  total: number
  offset: number
  limit: number
  classification_threshold: number
  risk_band_thresholds: { low_max: number; medium_max: number }
  risk_thresholds_note: string
  items: VehicleListItem[]
}

export type VehicleDetail = {
  vehicle_id: number
  specifications: Record<string, string | null>
  feature_values: Record<string, number | null>
  prediction: Prediction
}

export type Prediction = {
  vehicle_id: number | null
  repair_probability: number
  risk_level: 'Low' | 'Medium' | 'High'
  predicted_class: 'no_repair' | 'repair'
  classification_threshold: number
  risk_band_thresholds: { low_max: number; medium_max: number }
  prediction_window: string
  model_version: string
  prediction_timestamp: string
}

export type PredictionSummary = {
  evaluated_vehicle_count: number
  minimum_probability: number
  maximum_probability: number
  mean_probability: number
  median_probability: number
  percentiles: Record<string, number>
  at_or_above_classification_threshold: number
  risk_counts: Record<'Low' | 'Medium' | 'High', number>
}

export type FleetRisk = {
  population_scope: string
  classification_threshold: number
  risk_band_thresholds: { low_max: number; medium_max: number }
  risk_thresholds_note: string
  summary: PredictionSummary
  split_summaries: Record<'train' | 'validation' | 'test', PredictionSummary>
  top_vehicles: Prediction[]
}

export type ImportantFeature = {
  feature_name: string
  contribution: number
  contribution_scale: 'xgboost_raw_margin'
}

export type Investigation = {
  vehicle_id: number
  prediction: Prediction
  investigation: {
    summary: string
    summary_source: 'rule_based' | 'llm'
    important_features: ImportantFeature[]
    recommended_actions: string[]
  }
  disclaimer: string
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/+$/, '')

export class ApiError extends Error {
  readonly status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError(`Cannot reach FactoryIQ API at ${API_BASE_URL}. Check that the backend is running.`)
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body: unknown = await response.json()
      if (typeof body === 'object' && body !== null && 'detail' in body) {
        const value = body.detail
        if (typeof value === 'string') detail = value
        else if (Array.isArray(value)) detail = value.map((item) => typeof item === 'object' && item !== null && 'msg' in item ? String(item.msg) : 'Invalid input').join('; ')
      }
    } catch {
      detail = response.statusText || detail
    }
    throw new ApiError(detail, response.status)
  }

  return response.json() as Promise<T>
}

export const api = {
  health: (signal?: AbortSignal) => request<Health>('/health', { signal }),
  vehicles: (offset: number, limit: number, signal?: AbortSignal) =>
    request<VehiclesResponse>(`/vehicles?offset=${offset}&limit=${limit}`, { signal }),
  fleetRisk: (signal?: AbortSignal) => request<FleetRisk>('/fleet/risk', { signal }),
  vehicle: (vehicleId: number, signal?: AbortSignal) => request<VehicleDetail>(`/vehicles/${vehicleId}`, { signal }),
  predict: (vehicleId: number, signal?: AbortSignal) =>
    request<Prediction>('/predict', { method: 'POST', body: JSON.stringify({ vehicle_id: vehicleId }), signal }),
  investigate: (vehicleId: number, signal?: AbortSignal) =>
    request<Investigation>('/agent/investigate', { method: 'POST', body: JSON.stringify({ vehicle_id: vehicleId }), signal }),
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'An unexpected error occurred. Please try again.'
}

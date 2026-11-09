import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const vehicles = {
  total: 14809,
  offset: 0,
  limit: 25,
  classification_threshold: 0.19462941586971283,
  risk_band_thresholds: { low_max: 0.2, medium_max: 0.5 },
  risk_thresholds_note: 'Demonstration thresholds; not calibrated for operational use.',
  items: [
    { vehicle_id: 17, latest_time_step: 100, reading_count: 81, repair_probability: 0.42, risk_level: 'Medium', predicted_class: 'repair', prediction_window: 'time steps 100 to 200 (repair time >100 and <=200)' },
    { vehicle_id: 42, latest_time_step: 97, reading_count: 78, repair_probability: 0.08, risk_level: 'Low', predicted_class: 'no_repair', prediction_window: 'time steps 100 to 200 (repair time >100 and <=200)' },
  ],
}

const fleetRisk = {
  population_scope: 'Eligible 14,809-vehicle modeling cohort.',
  classification_threshold: 0.19462941586971283,
  risk_band_thresholds: { low_max: 0.2, medium_max: 0.5 },
  risk_thresholds_note: 'Demonstration thresholds; not calibrated for operational use.',
  summary: {
    evaluated_vehicle_count: 14809,
    minimum_probability: 0.0007,
    maximum_probability: 0.9056,
    mean_probability: 0.0592,
    median_probability: 0.0239,
    percentiles: { p1: 0.0023, p5: 0.0041, p10: 0.0056, p25: 0.0108, p50: 0.0239, p75: 0.0559, p90: 0.1355, p95: 0.2559, p99: 0.5874 },
    at_or_above_classification_threshold: 1009,
    risk_counts: { Low: 13831, Medium: 734, High: 244 },
  },
  split_summaries: {
    train: { evaluated_vehicle_count: 10366, minimum_probability: 0.0007, maximum_probability: 0.9056, mean_probability: 0.0612, median_probability: 0.0234, percentiles: {}, at_or_above_classification_threshold: 743, risk_counts: { Low: 9643, Medium: 501, High: 222 } },
    validation: { evaluated_vehicle_count: 2221, minimum_probability: 0.0008, maximum_probability: 0.7490, mean_probability: 0.0545, median_probability: 0.0243, percentiles: {}, at_or_above_classification_threshold: 131, risk_counts: { Low: 2094, Medium: 114, High: 13 } },
    test: { evaluated_vehicle_count: 2222, minimum_probability: 0.0011, maximum_probability: 0.7745, mean_probability: 0.0546, median_probability: 0.0253, percentiles: {}, at_or_above_classification_threshold: 135, risk_counts: { Low: 2094, Medium: 119, High: 9 } },
  },
  top_vehicles: [
    { vehicle_id: 10218, repair_probability: 0.90562755, risk_level: 'High', predicted_class: 'repair', classification_threshold: 0.19462941586971283, risk_band_thresholds: { low_max: 0.2, medium_max: 0.5 }, prediction_window: 'time steps 100 to 200', model_version: 'factoryiq-xgb-1.0.0', prediction_timestamp: '2026-11-08T00:00:00Z' },
  ],
}

const vehiclePrediction = {
  vehicle_id: 17,
  repair_probability: 0.42,
  risk_level: 'Medium' as const,
  predicted_class: 'repair' as const,
  classification_threshold: 0.19462941586971283,
  risk_band_thresholds: { low_max: 0.2, medium_max: 0.5 },
  prediction_window: 'steps 101-200',
  model_version: 'factoryiq-xgb-1.0.0',
  prediction_timestamp: '2025-01-02T03:04:05Z',
}

function renderApp(initialEntry = '/') {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={queryClient}><MemoryRouter initialEntries={[initialEntry]}><App /></MemoryRouter></QueryClientProvider>)
}

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }))
}

describe('FactoryIQ dashboard', () => {
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  beforeEach(() => {
    vi.restoreAllMocks()
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/health')) return jsonResponse({ status: 'ready', model_ready: true, model_version: 'factoryiq-xgb-1.0.0', detail: null })
      if (url.endsWith('/fleet/risk')) return jsonResponse(fleetRisk)
      if (url.includes('/vehicles?')) return jsonResponse(vehicles)
      if (url.endsWith('/vehicles/17')) return jsonResponse({ vehicle_id: 17, specifications: { Spec_0: 'category-a' }, feature_values: { '171_0__mean': 12.5 }, prediction: vehiclePrediction })
      return jsonResponse({ detail: 'Not found' }, 404)
    }))
  })

  it('renders the overview and navigates to the paginated vehicle table', async () => {
    renderApp()
    expect(await screen.findByRole('heading', { name: 'Overview' })).toBeInTheDocument()
    expect(await screen.findByText('14,809')).toBeInTheDocument()
    expect(await screen.findAllByText('244')).not.toHaveLength(0)
    expect(await screen.findByText('Vehicle 10218')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('link', { name: /Vehicle explorer/i }))
    expect(await screen.findByRole('heading', { name: 'Vehicle explorer' })).toBeInTheDocument()
    expect(await screen.findByText('Vehicle 17')).toBeInTheDocument()
    expect(screen.getByText('81')).toBeInTheDocument()
    expect(screen.getByText('42.0%')).toBeInTheDocument()
  })

  it('filters vehicles on the loaded server page and opens a selected vehicle', async () => {
    renderApp('/vehicles')
    expect(await screen.findByText('Vehicle 17')).toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox', { name: 'Search current page by vehicle ID' }), { target: { value: '42' } })
    expect(screen.getByText('Vehicle 42')).toBeInTheDocument()
    expect(screen.queryByText('Vehicle 17')).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox', { name: 'Search current page by vehicle ID' }), { target: { value: '' } })
    fireEvent.click(screen.getByRole('link', { name: 'View vehicle 17' }))
    expect(await screen.findByRole('heading', { name: 'Vehicle 17' })).toBeInTheDocument()
    fireEvent.click(screen.getByText('Advanced Details'))
    expect(screen.getByText('171_0__mean')).toBeInTheDocument()
    expect(screen.getByText('42.0')).toBeInTheDocument()
  })

  it('does not turn a missing prediction into a Low risk label', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/health')) return jsonResponse({ status: 'ready', model_ready: true, model_version: 'factoryiq-xgb-1.0.0', detail: null })
      if (url.includes('/vehicles?')) return jsonResponse({
        ...vehicles,
        items: [{ vehicle_id: 17, latest_time_step: 100, reading_count: 81 }],
      })
      return jsonResponse({ detail: 'Not found' }, 404)
    }))
    renderApp('/vehicles')
    expect(await screen.findByText('Prediction unavailable')).toBeInTheDocument()
    expect(screen.queryByText('Low risk')).not.toBeInTheDocument()
  })

  it('shows actual prediction fields returned by the API', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/health')) return jsonResponse({ status: 'ready', model_ready: true, model_version: 'factoryiq-xgb-1.0.0', detail: null })
      if (url.endsWith('/predict')) {
        expect(JSON.parse(String(init?.body))).toEqual({ vehicle_id: 17 })
        return jsonResponse(vehiclePrediction)
      }
      return jsonResponse(vehicles)
    }))
    renderApp('/prediction?vehicleId=17')
    fireEvent.click(await screen.findByRole('button', { name: /Run prediction/i }))
    expect(await screen.findByText('42.0')).toBeInTheDocument()
    expect(screen.getByText('Medium risk')).toBeInTheDocument()
    expect(screen.getByText('steps 101-200')).toBeInTheDocument()
    expect(screen.getByText(/saved binary classification cutoff is 0.1946/i)).toBeInTheDocument()
  })

  it('shows investigation source, contributions, recommendations and disclaimer', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/health')) return jsonResponse({ status: 'ready', model_ready: true, model_version: 'factoryiq-xgb-1.0.0', detail: null })
      if (url.endsWith('/agent/investigate')) return jsonResponse({
        vehicle_id: 17,
        prediction: vehiclePrediction,
        investigation: { summary: 'Deterministic model assessment.', summary_source: 'rule_based', important_features: [{ feature_name: '171_0__mean', contribution: 0.72, contribution_scale: 'xgboost_raw_margin' }], recommended_actions: ['Review anonymized history.'] },
        disclaimer: 'Decision-support prototype; not validated.',
      })
      return jsonResponse(vehicles)
    }))
    renderApp('/investigation?vehicleId=17')
    fireEvent.click(await screen.findByRole('button', { name: /Investigate vehicle/i }))
    expect(await screen.findByText('Deterministic model assessment.')).toBeInTheDocument()
    expect(screen.getByText('DETERMINISTIC')).toBeInTheDocument()
    expect(screen.getByText('171_0__mean')).toBeInTheDocument()
    expect(screen.getByText('Review anonymized history.')).toBeInTheDocument()
    expect(screen.getByText('Decision-support prototype; not validated.')).toBeInTheDocument()
  })

  it('renders useful backend error text when the vehicle API fails', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      if (String(input).endsWith('/health')) return jsonResponse({ status: 'degraded', model_ready: false, model_version: null, detail: 'Model is not loaded' })
      return jsonResponse({ detail: 'Vehicle service unavailable' }, 503)
    }))
    renderApp('/vehicles')
    const alert = await screen.findByRole('alert')
    expect(within(alert).getByText('Vehicle service unavailable')).toBeInTheDocument()
  })
})

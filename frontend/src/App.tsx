import { createContext, useContext, useMemo, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { Link, NavLink, Route, Routes, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  Activity, ArrowDownRight, ArrowRight, ArrowUpRight, BarChart3, Bot, ChevronLeft, ChevronRight,
  CircleAlert, ClipboardList, Gauge, Layers3, Menu, Search, ShieldCheck, Truck,
} from 'lucide-react'
import {
  api, errorMessage, type Health, type Investigation, type Prediction, type VehicleDetail,
  type VehicleListItem,
} from './lib/api'

type SessionState = {
  investigations: Investigation[]
  recordInvestigation: (investigation: Investigation) => void
}

const SessionContext = createContext<SessionState | null>(null)
const useSession = () => {
  const context = useContext(SessionContext)
  if (!context) throw new Error('Session context is unavailable')
  return context
}

const navItems = [
  { to: '/', label: 'Overview', icon: Gauge, end: true },
  { to: '/vehicles', label: 'Vehicle explorer', icon: Truck },
  { to: '/prediction', label: 'Risk prediction', icon: Activity },
  { to: '/investigation', label: 'AI investigation', icon: Bot },
  { to: '/insights', label: 'Model insights', icon: BarChart3 },
]

export default function App() {
  const [investigations, setInvestigations] = useState<Investigation[]>([])
  const [mobileOpen, setMobileOpen] = useState(false)
  const session = useMemo<SessionState>(() => ({
    investigations,
    recordInvestigation: (result) => {
      setInvestigations((current) => [result, ...current.filter((item) => item.vehicle_id !== result.vehicle_id)].slice(0, 8))
    },
  }), [investigations])

  return (
    <SessionContext.Provider value={session}>
      <AppShell mobileOpen={mobileOpen} setMobileOpen={setMobileOpen}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/vehicles" element={<VehicleExplorer />} />
          <Route path="/vehicles/:vehicleId" element={<VehicleDetails />} />
          <Route path="/prediction" element={<ActionPage mode="prediction" />} />
          <Route path="/investigation" element={<ActionPage mode="investigation" />} />
          <Route path="/insights" element={<ModelInsights />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </AppShell>
    </SessionContext.Provider>
  )
}

function AppShell({ children, mobileOpen, setMobileOpen }: { children: ReactNode; mobileOpen: boolean; setMobileOpen: (open: boolean) => void }) {
  return (
    <div className="app-shell">
      {mobileOpen && <button className="scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
      <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
        <Link className="brand" to="/" onClick={() => setMobileOpen(false)}>
          <span className="brand-mark"><Activity size={21} strokeWidth={2.2} /></span>
          <span><strong>Factory<span>IQ</span></strong><small>MAINTENANCE INTELLIGENCE</small></span>
        </Link>
        <div className="nav-caption">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          {navItems.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} onClick={() => setMobileOpen(false)} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Icon size={18} strokeWidth={1.8} /><span>{label}</span>
              {label === 'AI investigation' && <span className="nav-tag">AI</span>}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-note"><ShieldCheck size={17} /><span>Decision support<br /><small>Not a validated maintenance system</small></span></div>
          <div className="sidebar-footer">FactoryIQ <span>·</span> Local prototype</div>
        </div>
      </aside>
      <main className="main-shell">
        <header className="topbar">
          <button className="mobile-menu icon-button" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={20} /></button>
          <div className="topbar-title">Predictive Intelligence <span>/</span> SCANIA Component X</div>
          <BackendStatus />
        </header>
        <div className="page-content">{children}</div>
      </main>
    </div>
  )
}

function BackendStatus() {
  const { data, isError, isFetching } = useQuery<Health>({
    queryKey: ['health'], queryFn: ({ signal }) => api.health(signal), refetchInterval: 30000, retry: 1,
  })
  const ready = data?.model_ready
  return (
    <div className={`backend-status ${ready ? 'online' : 'offline'}`} title={data?.detail ?? (isError ? 'API unreachable' : undefined)}>
      <span className={`status-dot ${isFetching && !data ? 'pulse' : ''}`} />
      {ready ? 'Backend connected' : isError ? 'Backend unavailable' : data ? 'Model not ready' : 'Connecting'}
    </div>
  )
}

function PageHeading({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action && <div className="heading-action">{action}</div>}</div>
}

function Overview() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['fleet-risk'], queryFn: ({ signal }) => api.fleetRisk(signal),
  })
  const { investigations } = useSession()
  const summary = data?.summary
  const distribution = (['Low', 'Medium', 'High'] as const).map((name) => ({
    name,
    count: summary?.risk_counts[name] ?? 0,
  }))

  return (
    <>
      <PageHeading eyebrow="FLEET INTELLIGENCE" title="Overview" description="Cached model predictions for the complete eligible modeling cohort—not just vehicles scored in this browser session." action={<Link className="button button-primary" to="/vehicles"><Truck size={16} /> Explore vehicles <ArrowRight size={16} /></Link>} />
      <section className="hero-panel">
        <div className="hero-copy"><div className="hero-chip"><span className="chip-dot" /> PREDICTIVE MAINTENANCE WORKSPACE</div><h2>Make better-informed<br /><span>maintenance decisions.</span></h2><p>Explore cutoff-safe vehicle summaries, inspect model-estimated repair risk, and investigate contributing signals.</p><Link to="/vehicles" className="hero-link">Select a vehicle <ArrowRight size={16} /></Link></div>
        <div className="hero-art" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="hero-core"><Activity size={44} /></div><div className="orbit-node node-a"><Layers3 size={17} /></div><div className="orbit-node node-b"><BarChart3 size={17} /></div><div className="orbit-node node-c"><ShieldCheck size={17} /></div></div>
        <div className="hero-footnote">Historical SCANIA Component X dataset <span>·</span> Not live factory telemetry</div>
      </section>
      {error && <ErrorBanner error={error} />}
      <section className="stat-grid">
        <StatCard label="Vehicles evaluated" value={isLoading ? '—' : summary?.evaluated_vehicle_count.toLocaleString() ?? '—'} icon={<Truck size={18} />} tone="teal" caption="All eligible train, validation, and test records" />
        <StatCard label="At/above binary threshold" value={isLoading ? '—' : summary?.at_or_above_classification_threshold.toLocaleString() ?? '—'} icon={<Activity size={18} />} tone="blue" caption={`Classification cutoff: ${data?.classification_threshold.toFixed(4) ?? '—'}`} />
        <StatCard label="High demonstration band" value={isLoading ? '—' : summary?.risk_counts.High.toLocaleString() ?? '—'} icon={<CircleAlert size={18} />} tone="amber" caption="Probability ≥ 0.50; not a validated operational band" />
        <StatCard label="Prediction window" value="100 steps" icon={<Layers3 size={18} />} tone="violet" caption="Observed repair in steps 101–200" />
      </section>
      <div className="section-grid">
        <section className="panel">
          <div className="panel-heading"><div><h2>Fleet risk distribution</h2><p>{summary ? `${summary.evaluated_vehicle_count.toLocaleString()} eligible vehicles scored` : 'Loading cached fleet predictions'}</p></div><span className="soft-icon"><BarChart3 size={17} /></span></div>
          {error ? null : isLoading ? <LoadingState label="Loading fleet prediction summary…" /> : summary && <div className="risk-distribution">{distribution.map((item) => <div className="risk-row" key={item.name}><span className={`risk-dot risk-${item.name.toLowerCase()}`} /><span className="risk-name">{item.name}</span><div className="risk-track"><span className={`risk-fill fill-${item.name.toLowerCase()}`} style={{ width: `${item.count === 0 ? 0 : Math.max(3, (item.count / summary.evaluated_vehicle_count) * 100)}%` }} /></div><strong>{item.count.toLocaleString()}</strong></div>)}</div>}
          {data && <div className="panel-footnote">Population: {data.population_scope}<br />Low &lt; {data.risk_band_thresholds.low_max.toFixed(2)} · Medium ≥ {data.risk_band_thresholds.low_max.toFixed(2)} and &lt; {data.risk_band_thresholds.medium_max.toFixed(2)} · High ≥ {data.risk_band_thresholds.medium_max.toFixed(2)}. {data.risk_thresholds_note} Binary classification uses a separate cutoff of {data.classification_threshold.toFixed(4)}.</div>}
        </section>
        <section className="panel">
          <div className="panel-heading"><div><h2>Highest predicted probabilities</h2><p>Top vehicles in the scored eligible cohort</p></div><span className="soft-icon"><ArrowUpRight size={17} /></span></div>
          {error ? null : isLoading ? <LoadingState label="Loading highest-risk vehicles…" /> : data?.top_vehicles.slice(0, 5).map((prediction) => (
            <Link className="activity-item" key={prediction.vehicle_id} to={`/vehicles/${prediction.vehicle_id}`}>
              <span className="activity-icon"><Truck size={15} /></span>
              <span className="activity-copy"><strong>Vehicle {prediction.vehicle_id}</strong><small>{(prediction.repair_probability * 100).toFixed(1)}% estimated probability</small></span>
              <RiskPill risk={prediction.risk_level} />
              <ChevronRight size={15} />
            </Link>
          ))}
          {data && <div className="panel-footnote">Predictions are model estimates, not repair guarantees. They are scored from step-100 features for the stated 100-step window.</div>}
        </section>
        <section className="panel">
          <div className="panel-heading"><div><h2>Recent investigations</h2><p>Activity recorded in this browser session</p></div><span className="soft-icon"><Bot size={17} /></span></div>
          {investigations.length ? <div className="activity-list">{investigations.slice(0, 4).map((item) => <Link className="activity-item" key={item.vehicle_id} to={`/vehicles/${item.vehicle_id}`}><span className="activity-icon"><Bot size={15} /></span><span className="activity-copy"><strong>Vehicle {item.vehicle_id}</strong><small>{item.investigation.summary_source === 'llm' ? 'LLM-assisted summary' : 'Rule-based summary'}</small></span><RiskPill risk={item.prediction.risk_level} /><ChevronRight size={15} /></Link>)}</div> : <EmptyState icon={<ClipboardList />} title="Nothing investigated yet" message="Investigations you run will appear here for this session." />}
        </section>
      </div>
      <div className="callout-note"><ShieldCheck size={17} /><p><strong>Decision-support prototype.</strong> Predictions estimate observed Component X repair risk within the stated window; they are not failure guarantees or safety/maintenance instructions.</p></div>
    </>
  )
}

function StatCard({ label, value, icon, tone, caption }: { label: string; value: string; icon: ReactNode; tone: string; caption: string }) {
  return <div className="stat-card"><span className={`stat-icon ${tone}`}>{icon}</span><span className="stat-label">{label}</span><strong className="stat-value">{value}</strong><span className="stat-caption">{caption}</span></div>
}

function VehicleExplorer() {
  const [offset, setOffset] = useState(0)
  const [search, setSearch] = useState('')
  const limit = 25
  const { data, isLoading, error, isFetching } = useQuery({
    queryKey: ['vehicles', offset, limit], queryFn: ({ signal }) => api.vehicles(offset, limit, signal),
  })
  const filtered = (data?.items ?? []).filter((item) => String(item.vehicle_id).includes(search.trim()))

  return (
    <>
      <PageHeading eyebrow="FLEET DATA" title="Vehicle explorer" description="Browse cached predictions for the eligible 14,809-vehicle modeling cohort. All displayed probabilities use the saved model." />
      <section className="panel table-panel">
        <div className="table-toolbar"><div><h2>Eligible vehicles and predictions</h2><p>{data ? `${data.total.toLocaleString()} eligible records · showing ${offset + 1}–${Math.min(offset + limit, data.total)}` : 'Loading vehicle records'}</p></div><label className="search-box"><Search size={16} /><input aria-label="Search current page by vehicle ID" placeholder="Search this page…" value={search} onChange={(event) => setSearch(event.target.value)} /><span>Current page</span></label></div>
        <p className="search-hint">The API does not provide search; this filter applies to the current page only.</p>
        {data && <p className="search-hint">Binary repair classification uses ≥ {data.classification_threshold.toFixed(4)}. Demonstration bands: Low &lt; {data.risk_band_thresholds.low_max.toFixed(2)}, Medium &lt; {data.risk_band_thresholds.medium_max.toFixed(2)} after the Low band, High ≥ {data.risk_band_thresholds.medium_max.toFixed(2)}. {data.risk_thresholds_note}</p>}
        {error ? <ErrorBanner error={error} /> : isLoading ? <LoadingState label="Loading vehicle records…" /> : filtered.length ? <div className="table-scroll"><table><thead><tr><th>VEHICLE</th><th>AVAILABLE READINGS</th><th>LATEST TIME STEP</th><th>MODEL-ESTIMATED REPAIR PROBABILITY</th><th>RISK BAND / THRESHOLD CLASS</th><th>PREDICTION WINDOW</th><th /></tr></thead><tbody>{filtered.map((vehicle) => <VehicleRow key={vehicle.vehicle_id} vehicle={vehicle} />)}</tbody></table></div> : <EmptyState icon={<Search />} title="No vehicles found" message={search ? 'No matching vehicle ID on this page. Try another ID or move to a different page.' : 'The API returned no vehicle records for this page.'} />}
        <div className="pagination"><span>Page {Math.floor(offset / limit) + 1} of {data ? Math.max(1, Math.ceil(data.total / limit)).toLocaleString() : '—'}</span><div><button className="button button-secondary button-small" disabled={offset === 0 || isFetching} onClick={() => setOffset(Math.max(0, offset - limit))}><ChevronLeft size={15} /> Previous</button><button className="button button-secondary button-small" disabled={!data || offset + limit >= data.total || isFetching} onClick={() => setOffset(offset + limit)}>Next <ChevronRight size={15} /></button></div></div>
      </section>
      <div className="callout-note"><Layers3 size={17} /><p>Details contain cutoff-safe historical feature summaries. Anonymized identifiers are displayed as recorded and are not interpreted as physical sensor names.</p></div>
    </>
  )
}

function VehicleRow({ vehicle }: { vehicle: VehicleListItem }) {
  const hasPrediction = Number.isFinite(vehicle.repair_probability) && typeof vehicle.risk_level === 'string'
  return <tr><td><Link className="vehicle-link" to={`/vehicles/${vehicle.vehicle_id}`}><span className="vehicle-avatar"><Truck size={16} /></span><span><strong>Vehicle {vehicle.vehicle_id}</strong><small>SCANIA Component X</small></span></Link></td><td>{vehicle.reading_count?.toLocaleString() ?? '—'}</td><td>{vehicle.latest_time_step ?? '—'}</td><td>{hasPrediction ? <><strong>{(vehicle.repair_probability * 100).toFixed(1)}%</strong><small>Estimated repair probability</small></> : <span className="muted-status">Prediction unavailable</span>}</td><td>{hasPrediction ? <><RiskPill risk={vehicle.risk_level} /><small>{vehicle.predicted_class === 'repair' ? 'At/above binary cutoff' : 'Below binary cutoff'}</small></> : '—'}</td><td>{hasPrediction ? vehicle.prediction_window : '—'}</td><td><Link className="row-action" to={`/vehicles/${vehicle.vehicle_id}`} aria-label={`View vehicle ${vehicle.vehicle_id}`}><ArrowRight size={17} /></Link></td></tr>
}

function VehicleDetails() {
  const { vehicleId: vehicleIdParam } = useParams()
  const vehicleId = Number(vehicleIdParam)
  const validId = Number.isInteger(vehicleId) && vehicleId >= 0
  const { data, isLoading, error } = useQuery({
    queryKey: ['vehicle', vehicleId], queryFn: ({ signal }) => api.vehicle(vehicleId, signal), enabled: validId,
  })
  if (!validId) return <ErrorBanner error="Vehicle ID must be a non-negative integer." />
  if (isLoading) return <LoadingState label={`Loading vehicle ${vehicleId}…`} />
  if (error) return <><PageHeading eyebrow="VEHICLE RECORD" title={`Vehicle ${vehicleId}`} description="Unable to load this vehicle record." /><ErrorBanner error={error} /></>
  if (!data) return null
  return <VehicleDetailContent vehicle={data} />
}

function VehicleDetailContent({ vehicle }: { vehicle: VehicleDetail }) {
  const [featureSearch, setFeatureSearch] = useState('')
  const specifications = Object.entries(vehicle.specifications)
  const features = Object.entries(vehicle.feature_values).filter(([name]) => name.toLowerCase().includes(featureSearch.toLowerCase())).slice(0, 28)
  return (
    <>
      <PageHeading eyebrow="VEHICLE RECORD" title={`Vehicle ${vehicle.vehicle_id}`} description="Historical model inputs available at the step-100 prediction cutoff." action={<Link className="button button-secondary" to="/vehicles"><ChevronLeft size={16} /> Back to vehicles</Link>} />
      <div className="detail-action-row"><Link className="button button-primary" to={`/prediction?vehicleId=${vehicle.vehicle_id}`}><Activity size={16} /> Predict repair risk</Link><Link className="button button-secondary" to={`/investigation?vehicleId=${vehicle.vehicle_id}`}><Bot size={16} /> Investigate vehicle</Link></div>
      <section className="panel vehicle-risk-panel">
        <div className="panel-heading"><div><h2>Model-estimated repair probability</h2><p>Cached prediction for this vehicle from the saved model</p></div><span className="soft-icon"><Activity size={17} /></span></div>
        <div className="probability-display"><strong>{(vehicle.prediction.repair_probability * 100).toFixed(1)}<small>%</small></strong><RiskPill risk={vehicle.prediction.risk_level} /></div>
        <RiskExplanation prediction={vehicle.prediction} />
        <div className="result-details"><DetailLine label="Prediction window" value={vehicle.prediction.prediction_window} /><DetailLine label="Model version" value={vehicle.prediction.model_version} /></div>
      </section>
      <div className="detail-grid">
        <section className="panel"><div className="panel-heading"><div><h2>Vehicle specifications</h2><p>Specification values returned by the API</p></div><span className="soft-icon"><Truck size={17} /></span></div><div className="spec-grid">{specifications.map(([name, value]) => <div className="spec-item" key={name}><span>{name}</span><strong>{value ?? 'Not available'}</strong></div>)}</div></section>
        <section className="panel cutoff-panel"><span className="stat-icon teal"><Layers3 size={18} /></span><h2>Prediction cutoff &amp; horizon</h2><p>The model uses available feature summaries through <strong>time step 100</strong> to estimate observed Component X repair risk during the subsequent <strong>100-step window</strong>.</p><div className="cutoff-bar"><span>Historical feature cutoff</span><strong>Step 100</strong></div><p className="small-note">This is a risk ranking, not a failure-date estimate. No raw sensor timeline is provided by this endpoint.</p></section>
      </div>
      <details className="advanced-details">
        <summary>Advanced Details</summary>
        <section className="panel feature-panel"><div className="panel-heading"><div><h2>Anonymized operational feature summaries</h2><p>{Object.keys(vehicle.feature_values).length} feature fields · values are not interpreted as physical measurements</p></div><label className="search-box feature-search"><Search size={16} /><input aria-label="Filter feature identifiers" placeholder="Filter identifiers…" value={featureSearch} onChange={(event) => setFeatureSearch(event.target.value)} /></label></div>
          {features.length ? <div className="feature-grid">{features.map(([name, value]) => <div className="feature-item" key={name}><span title={name}>{name}</span><strong>{value === null ? '—' : Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 })}</strong></div>)}</div> : <EmptyState icon={<Search />} title="No matching identifiers" message="Try a different feature name." />}
          {features.length === 28 && <p className="panel-footnote">Showing first 28 matching identifiers. Filter above to inspect other fields.</p>}
        </section>
      </details>
    </>
  )
}

function ActionPage({ mode }: { mode: 'prediction' | 'investigation' }) {
  const [params] = useSearchParams()
  const [vehicleId, setVehicleId] = useState(params.get('vehicleId') ?? '')
  const [predictionResult, setPredictionResult] = useState<Prediction | null>(null)
  const [investigationResult, setInvestigationResult] = useState<Investigation | null>(null)
  const { recordInvestigation } = useSession()
  const navigate = useNavigate()
  const isInvestigation = mode === 'investigation'
  const predictionMutation = useMutation({ mutationFn: (id: number) => api.predict(id), onSuccess: setPredictionResult })
  const investigationMutation = useMutation({ mutationFn: (id: number) => api.investigate(id), onSuccess: (result) => { setInvestigationResult(result); setPredictionResult(result.prediction); recordInvestigation(result) } })
  const pending = predictionMutation.isPending || investigationMutation.isPending
  const error = predictionMutation.error ?? investigationMutation.error

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const id = Number(vehicleId)
    if (!Number.isInteger(id) || id < 0) return
    setPredictionResult(null)
    setInvestigationResult(null)
    if (isInvestigation) investigationMutation.mutate(id)
    else predictionMutation.mutate(id)
  }

  const valid = vehicleId.trim() !== '' && Number.isInteger(Number(vehicleId)) && Number(vehicleId) >= 0
  return (
    <>
      <PageHeading eyebrow={isInvestigation ? 'LANGGRAPH WORKFLOW' : 'MODEL INFERENCE'} title={isInvestigation ? 'AI investigation center' : 'Repair-risk prediction'} description={isInvestigation ? 'Run the backend investigation workflow for one selected vehicle.' : 'Request an XGBoost estimate using a vehicle’s cutoff-safe model inputs.'} />
      <div className="action-layout">
        <section className="panel action-form-panel"><div className="action-icon">{isInvestigation ? <Bot size={21} /> : <Activity size={21} />}</div><h2>{isInvestigation ? 'Investigate a vehicle' : 'Estimate repair risk'}</h2><p>{isInvestigation ? 'The workflow retrieves the vehicle, predicts risk, computes feature contributions, and returns a summary with recommended next steps.' : 'The backend loads the vehicle’s model input record and returns a repair probability and demonstration risk category.'}</p>
          <form onSubmit={submit}><label htmlFor="vehicle-id">Vehicle identifier</label><input id="vehicle-id" type="number" min="0" step="1" required value={vehicleId} onChange={(event) => setVehicleId(event.target.value)} placeholder="e.g. 12" /><button className="button button-primary button-wide" type="submit" disabled={!valid || pending}>{pending ? <><span className="spinner" /> {isInvestigation ? 'Investigating…' : 'Predicting…'}</> : <>{isInvestigation ? <Bot size={16} /> : <Activity size={16} />}{isInvestigation ? 'Investigate vehicle' : 'Run prediction'}<ArrowRight size={16} /></>}</button></form>
          {valid && <Link className="inline-link" to={`/vehicles/${Number(vehicleId)}`}>View vehicle details <ArrowRight size={14} /></Link>}
          {error && <ErrorBanner error={error} />}
        </section>
        <section className="panel result-panel">
          {investigationResult ? <InvestigationResult result={investigationResult} /> : predictionResult ? <PredictionResult prediction={predictionResult} /> : <EmptyState icon={isInvestigation ? <Bot /> : <Activity />} title="Results will appear here" message={isInvestigation ? 'Choose a vehicle identifier and start an investigation.' : 'Choose a vehicle identifier and run a prediction.'} />}
        </section>
      </div>
      <div className="callout-note"><ShieldCheck size={17} /><p><strong>Use as decision support only.</strong> Probabilities are estimates, not guaranteed failures. Low/Medium/High categories use demonstration thresholds and are not validated maintenance decision thresholds.</p></div>
      {predictionResult && !isInvestigation && <div className="result-followup"><span>Want more context for vehicle {predictionResult.vehicle_id}?</span><button className="button button-secondary button-small" onClick={() => { setInvestigationResult(null); navigate(`/investigation?vehicleId=${predictionResult.vehicle_id}`) }}>Investigate vehicle <ArrowRight size={14} /></button></div>}
    </>
  )
}

function PredictionResult({ prediction }: { prediction: Prediction }) {
  const percent = prediction.repair_probability * 100
  return <div className="prediction-result"><div className="result-kicker"><span className="result-check"><ShieldCheck size={16} /></span> MODEL ESTIMATE</div><h2>Repair risk estimate</h2><div className="probability-display"><strong>{percent.toFixed(1)}<small>%</small></strong><RiskPill risk={prediction.risk_level} /></div><div className={`probability-track ${prediction.risk_level ? '' : 'probability-unavailable'}`}><span className={`probability-fill fill-${prediction.risk_level.toLowerCase()}`} style={{ width: `${percent}%` }} /></div><RiskExplanation prediction={prediction} /><div className="result-details"><DetailLine label="Vehicle" value={prediction.vehicle_id === null ? 'Feature payload' : String(prediction.vehicle_id)} /><DetailLine label="Prediction window" value={prediction.prediction_window} /><DetailLine label="Model version" value={prediction.model_version} /><DetailLine label="Generated" value={new Date(prediction.prediction_timestamp).toLocaleString()} /></div></div>
}

function RiskExplanation({ prediction }: { prediction: Prediction }) {
  const { low_max: lowMax, medium_max: mediumMax } = prediction.risk_band_thresholds
  const bandDescription = {
    Low: `Low demonstration band (probability below ${lowMax.toFixed(2)}).`,
    Medium: `Medium demonstration band (probability from ${lowMax.toFixed(2)} to below ${mediumMax.toFixed(2)}).`,
    High: `High demonstration band (probability at least ${mediumMax.toFixed(2)}).`,
  }[prediction.risk_level]
  const nextStep = prediction.risk_level === 'High' || prediction.predicted_class === 'repair'
    ? 'Review the vehicle history and corroborate this signal with established diagnostic procedures before taking action.'
    : 'Continue routine monitoring and review other available maintenance evidence; a low estimate does not rule out a repair.'
  return <div className="risk-explanation">
    <p>{bandDescription} The saved binary classification cutoff is {prediction.classification_threshold.toFixed(4)}; this vehicle is {prediction.predicted_class === 'repair' ? 'at or above' : 'below'} that separate cutoff.</p>
    <p><strong>Suggested next investigation:</strong> {nextStep}</p>
    <p className="small-note">This estimates an observed repair in the stated window, not failure certainty or timing. Bands are demonstration-only, not validated operational thresholds; anonymized features do not establish cause.</p>
  </div>
}

function InvestigationResult({ result }: { result: Investigation }) {
  const features = result.investigation.important_features
  const maxContribution = Math.max(0.001, ...features.map((item) => Math.abs(item.contribution)))
  return <div className="investigation-result"><div className="result-kicker"><span className="result-check"><Bot size={16} /></span> INVESTIGATION COMPLETE</div><div className="investigation-title"><div><h2>Vehicle {result.vehicle_id}</h2><p>{(result.prediction.repair_probability * 100).toFixed(1)}% estimated repair probability</p></div><RiskPill risk={result.prediction.risk_level} /></div>
    <RiskExplanation prediction={result.prediction} />
    <div className="investigation-summary"><span className="summary-label">SUMMARY <span className="source-tag">{result.investigation.summary_source === 'rule_based' ? 'DETERMINISTIC' : 'LLM-ASSISTED'}</span></span><p>{result.investigation.summary}</p></div>
    <div className="contribution-section"><h3>Important contributing features</h3><p className="small-note">XGBoost raw-margin contribution values indicate model influence, not causation. Identifiers remain anonymized.</p>{features.length ? <div className="contribution-list">{features.slice(0, 8).map((item) => <div className="contribution-row" key={item.feature_name}><span title={item.feature_name}>{item.feature_name}</span><div className="contribution-track"><span className={item.contribution >= 0 ? 'contribution-positive' : 'contribution-negative'} style={{ width: `${Math.max(2, Math.abs(item.contribution) / maxContribution * 100)}%` }} /></div><strong>{item.contribution >= 0 ? '+' : ''}{item.contribution.toFixed(3)}</strong></div>)}</div> : <p className="small-note">No contribution values returned.</p>}</div>
    <div className="next-actions"><h3>Recommended next actions</h3><ul>{result.investigation.recommended_actions.map((action, index) => <li key={`${index}-${action}`}><span>{String(index + 1).padStart(2, '0')}</span>{action}</li>)}</ul></div><div className="disclaimer"><ShieldCheck size={16} />{result.disclaimer}</div>
  </div>
}

function DetailLine({ label, value }: { label: string; value: string }) { return <div className="detail-line"><span>{label}</span><strong>{value}</strong></div> }

function ModelInsights() {
  const metrics = [
    { label: 'ROC-AUC', value: '0.793', description: 'Ranking performance across thresholds', icon: <Activity size={18} /> },
    { label: 'Average precision', value: '0.261', description: 'Precision-recall summary score', icon: <BarChart3 size={18} /> },
    { label: 'Precision', value: '0.326', description: 'Share of positive alerts that were repairs', icon: <ArrowUpRight size={18} /> },
    { label: 'Recall', value: '0.321', description: 'Share of repairs detected by the model', icon: <ArrowDownRight size={18} /> },
  ]
  return <>
    <PageHeading eyebrow="MODEL EVALUATION" title="Model insights" description="Held-out test results from the saved evaluation artifact; understand the trade-offs before using risk estimates." />
    <div className="metrics-grid">{metrics.map((metric) => <div className="metric-card" key={metric.label}><span className="metric-icon">{metric.icon}</span><span className="stat-label">{metric.label}</span><strong>{metric.value}</strong><small>{metric.description}</small></div>)}</div>
    <div className="insights-grid">
      <section className="panel confusion-panel"><div className="panel-heading"><div><h2>Confusion matrix</h2><p>Held-out test set · validation-selected threshold 0.1946</p></div></div><div className="matrix-axis"><span>Predicted class</span><span>NO REPAIR</span><span>REPAIR</span></div><div className="matrix-row"><strong>Actual<br />no repair</strong><div className="matrix-cell true-cell"><span>TRUE NEGATIVES</span><strong>1,994</strong></div><div className="matrix-cell false-cell"><span>FALSE POSITIVES</span><strong>91</strong></div></div><div className="matrix-row"><strong>Actual<br />repair</strong><div className="matrix-cell false-cell"><span>FALSE NEGATIVES</span><strong>93</strong></div><div className="matrix-cell true-cell"><span>TRUE POSITIVES</span><strong>44</strong></div></div><div className="panel-footnote">Source: models/evaluation_metrics.json · 2,222 held-out test vehicles</div></section>
      <section className="panel metric-explainer"><div className="panel-heading"><div><h2>Reading the results</h2><p>Plain-language context</p></div><span className="soft-icon"><ClipboardList size={17} /></span></div><div className="explanation-item"><span className="explain-bullet teal-bullet" /><div><strong>Precision · 32.6%</strong><p>About 1 in 3 model-positive cases were repairs in this test set. Many positive predictions were false alarms.</p></div></div><div className="explanation-item"><span className="explain-bullet blue-bullet" /><div><strong>Recall · 32.1%</strong><p>The model identified about 1 in 3 repairs in the held-out set; most repair events were missed at this threshold.</p></div></div><div className="explanation-item"><span className="explain-bullet amber-bullet" /><div><strong>Class imbalance</strong><p>Only 137 of 2,222 test records (6.17%) were positive. Accuracy alone would hide weak repair detection.</p></div></div><div className="threshold-box"><span>Validation-selected threshold</span><strong>0.1946</strong><small>Selected on validation data using maximum F1; Low/Medium/High dashboard categories use separate demonstration thresholds.</small></div></section>
    </div>
    <div className="callout-note warning-note"><CircleAlert size={17} /><p><strong>Limitations.</strong> Metrics reflect one held-out split and observed repair labels. Contributions are not causal explanations; anonymized fields have no known physical interpretation. This prototype is not production-ready or validated for maintenance/safety decisions.</p></div>
  </>
}

function RiskPill({ risk }: { risk: Prediction['risk_level'] }) { return <span className={`risk-pill pill-${risk.toLowerCase()}`}><span />{risk} risk</span> }
function ErrorBanner({ error }: { error: unknown }) { return <div className="error-banner" role="alert"><CircleAlert size={18} /><div><strong>Unable to complete request</strong><p>{errorMessage(error)}</p></div></div> }
function LoadingState({ label }: { label: string }) { return <div className="loading-state"><span className="spinner" />{label}</div> }
function EmptyState({ icon, title, message }: { icon: ReactNode; title: string; message: string }) { return <div className="empty-state"><span className="empty-icon">{icon}</span><strong>{title}</strong><p>{message}</p></div> }
function NotFound() { return <div className="not-found"><h1>Page not found</h1><p>The page you requested does not exist.</p><Link className="button button-primary" to="/">Return to overview</Link></div> }

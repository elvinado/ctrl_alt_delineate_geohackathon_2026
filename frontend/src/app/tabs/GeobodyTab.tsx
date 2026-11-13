import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
const ProspectVoxelCanvas = lazy(() => import('./ProspectVoxelCanvas'));
import {
    AlertTriangle, Box, Download, Image as ImageIcon,
    RefreshCw, Search, SlidersHorizontal,
} from 'lucide-react';
import {
    geobodyAssetUrl, getGeobody, getGeobodyManifest, getGeobodySlice, getGeobodyVoxelPreview, listGeobodies,
    listProcessingJobs,
    type GeobodyDetailResponse, type GeobodyManifest, type GeobodyRecord, type GeobodySliceResponse, type GeobodyVoxelPreview,
} from '../../api/backend';
import './GeobodyTab.css';

type Props = { backendConnected: boolean };

const API_ROOT = '/api/v1/geobodies/jobs/';
const value = (row: GeobodyRecord | undefined, key: string): string | number | boolean | null =>
    row?.[key] ?? null;
const fmt = (n: unknown, digits = 1) => typeof n === 'number' && Number.isFinite(n)
    ? n.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';
const statusText = (status: unknown) => String(status ?? 'UNKNOWN').replaceAll('_', ' ');
const maskPath = (mask: number[][]): string => {
    const segments: string[] = [];
    mask.forEach((row, y) => {
        let start = -1;
        row.forEach((cell, x) => {
            if (cell && start < 0) start = x;
            if ((!cell || x === row.length - 1) && start >= 0) {
                const end = cell && x === row.length - 1 ? x + 1 : x;
                segments.push(`M${start} ${y}h${end - start}v1h-${end - start}z`);
                start = -1;
            }
        });
    });
    return segments.join('');
};

export default function GeobodyTab({ backendConnected }: Props) {
    const [jobId, setJobId] = useState('');
    const [jobInput, setJobInput] = useState('');
    const [jobOptions, setJobOptions] = useState<{ id: string; label: string }[]>([]);
    const [manifest, setManifest] = useState<GeobodyManifest | null>(null);
    const [bodies, setBodies] = useState<GeobodyRecord[]>([]);
    const [selectedId, setSelectedId] = useState<number | null>(null);
    const [detail, setDetail] = useState<GeobodyDetailResponse | null>(null);
    const [slice, setSlice] = useState<GeobodySliceResponse | null>(null);
    const [voxelPreview, setVoxelPreview] = useState<GeobodyVoxelPreview | null>(null);
    const [voxelLoading, setVoxelLoading] = useState(false);
    const [voxelError, setVoxelError] = useState('');
    const [plane, setPlane] = useState<'inline' | 'crossline' | 'time'>('inline');
    const [coordinate, setCoordinate] = useState('');
    const [search, setSearch] = useState('');
    const [loading, setLoading] = useState(false);
    const [sliceLoading, setSliceLoading] = useState(false);
    const [error, setError] = useState('');
    const [jobLoading, setJobLoading] = useState(false);
    const discoveryStarted = useRef(false);
    const [viewerIndex, setViewerIndex] = useState<number | null>(null);
    const touchStartX = useRef<number | null>(null);

    const loadJob = useCallback(async (id: string) => {
        if (!id.trim()) return;
        setLoading(true);
        setError('');
        setJobId(id.trim());
        setJobInput(id.trim());
        setDetail(null);
        setSlice(null);
        setSelectedId(null);
        try {
            const [nextManifest, nextList] = await Promise.all([
                getGeobodyManifest(id.trim()),
                listGeobodies(id.trim(), 200, 0),
            ]);
            setManifest(nextManifest);
            setBodies(nextList.geobodies);
            if (nextList.geobodies.length) setSelectedId(Number(nextList.geobodies[0].geobody_id));
            if (!nextList.geobodies.length) setError('The Stage 12 catalogue is available, but contains no retained geobodies.');
        } catch (e) {
            setManifest(null);
            setBodies([]);
            setError(e instanceof Error ? e.message : 'Could not load geobody products.');
        } finally { setLoading(false); }
    }, []);

    const discoverJobs = useCallback(async () => {
        if (!backendConnected) return;
        setJobLoading(true);
        setError('');
        try {
            const result = await listProcessingJobs(100, 0);
            const candidates = result.items.filter(job => {
                // Worker entry points were renamed in backend v155. Accept the
                // current name and the legacy name so jobs created before the
                // rename remain discoverable. API metadata may contain either
                // a basename or a path, so normalize it before comparison.
                const rawName = String(job.python_script ?? job.operation ?? '')
                    .replaceAll('\\', '/');
                const scriptName = rawName.split('/').pop()?.toLowerCase() ?? '';
                return [
                    'process_d_geobody.py',
                    'process_geogg_geobody.py',
                    'process_d_geobody',
                    'process_geogg_geobody',
                ].includes(scriptName);
            });
            const options: { id: string; label: string }[] = [];
            for (const job of candidates) {
                const id = String(job.job_id ?? job.id ?? '');
                if (!id) continue;
                try {
                    const summary = await getGeobodyManifest(id);
                    if (summary.stage === '12_geobody') {
                        options.push({
                            id,
                            label: `${id} · ${String(job.status ?? summary.status ?? 'job')} · ${job.created_at ? new Date(job.created_at).toLocaleString() : 'date unavailable'}`,
                        });
                    }
                } catch { /* Ignore jobs without a published Stage 12 manifest. */ }
            }
            setJobOptions(options);
            if (!jobId && options.length) await loadJob(options[0].id);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Could not discover geobody jobs.');
        } finally { setJobLoading(false); }
    }, [backendConnected, jobId, loadJob]);

    useEffect(() => {
        if (backendConnected && !discoveryStarted.current) {
            discoveryStarted.current = true;
            void discoverJobs();
        }
    }, [backendConnected, discoverJobs]);

    useEffect(() => {
        if (!jobId || selectedId === null) return;
        let cancelled = false;
        setSlice(null);
        setDetail(null);
        void getGeobody(jobId, selectedId).then(result => {
            if (!cancelled) setDetail(result);
        }).catch(e => {
            if (!cancelled) setError(e instanceof Error ? e.message : 'Could not load geobody details.');
        });
        return () => { cancelled = true; };
    }, [jobId, selectedId]);

    useEffect(() => {
        if (!jobId || selectedId === null) { setVoxelPreview(null); return; }
        let active = true; setVoxelLoading(true); setVoxelError('');
        getGeobodyVoxelPreview(jobId, selectedId).then(result => { if (active) setVoxelPreview(result); })
            .catch(e => { if (active) { setVoxelPreview(null); setVoxelError(e instanceof Error ? e.message : 'Could not load geobody voxels.'); } })
            .finally(() => { if (active) setVoxelLoading(false); });
        return () => { active = false; };
    }, [jobId, selectedId]);

    const selectedBody = useMemo(
        () => bodies.find(body => Number(body.geobody_id) === selectedId),
        [bodies, selectedId],
    );

    const loadSlice = async (nextPlane = plane, nextCoordinate?: number) => {
        if (!jobId || selectedId === null) return;
        setSliceLoading(true);
        setError('');
        try {
            const result = await getGeobodySlice(jobId, selectedId, nextPlane, nextCoordinate);
            setSlice(result);
            setPlane(nextPlane);
            setCoordinate(String(result.coordinate));
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Could not retrieve geobody slice.');
        } finally { setSliceLoading(false); }
    };

    useEffect(() => {
        if (!jobId || selectedId === null || !selectedBody) return;
        const field = plane === 'inline' ? 'centroid_inline' : plane === 'crossline' ? 'centroid_crossline' : 'centroid_twt_ms';
        const coord = Number(selectedBody[field]);
        if (Number.isFinite(coord)) void loadSlice(plane, coord);
        // Load the default slice when the selected body changes.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [jobId, selectedId]);

    const filteredBodies = useMemo(() => {
        const q = search.trim().toLowerCase();
        if (!q) return bodies;
        return bodies.filter(body => [body.geobody_id, body.voxel_count, body.closure_status, body.interpretation_status]
            .some(item => String(item ?? '').toLowerCase().includes(q)));
    }, [bodies, search]);

    const figures = (manifest?.exports?.visualisations ?? []).filter(item => typeof item.name === 'string') as { name: string; title?: string; geobody_id?: number; [key: string]: unknown }[];
    const figureItems = useMemo(() => {
        const items = [
            ...figures.map(figure => ({
                name: figure.name,
                title: String(figure.title ?? figure.name),
                url: geobodyAssetUrl(jobId, `/figures/${encodeURIComponent(figure.name)}`),
            })),
            ...(detail?.figures ?? []).map(figure => ({
                name: figure.name,
                title: String(figure.title ?? figure.name),
                url: figure.url,
            })),
        ];
        return items.filter((item, index) => items.findIndex(candidate => candidate.name === item.name) === index);
    }, [figures, detail?.figures, jobId]);
    useEffect(() => {
        if (viewerIndex === null) return;
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') setViewerIndex(null);
            if (event.key === 'ArrowRight') setViewerIndex(index => index === null ? null : (index + 1) % figureItems.length);
            if (event.key === 'ArrowLeft') setViewerIndex(index => index === null ? null : (index - 1 + figureItems.length) % figureItems.length);
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [viewerIndex, figureItems.length]);
    const meshes = manifest?.exports?.meshes ?? [];
    const segyProducts = manifest?.exports?.segy ?? [];
    const segyReady = (product: string) => segyProducts.some(item => String(item.product ?? '').includes(product) && String(item.status ?? '').toUpperCase() === 'PASS');

    return <section className="page geobody-page">
        <div className="page-title geobody-page-title">
            <div>
                <span className="eyebrow">STAGE 12 · GEOBODY PRODUCTS</span>
                <h1>Geobody viewer</h1>
                <p>Explore extracted 3D bodies, inspect their slices, and retrieve meshes and SEG-Y exports.</p>
            </div>
            <div className="geobody-title-actions">
                <button className="btn btn-secondary" onClick={() => void discoverJobs()} disabled={!backendConnected || jobLoading}>
                    <RefreshCw className={jobLoading ? 'spin' : ''}/> Refresh jobs
                </button>
            </div>
        </div>

        <div className="geobody-jobbar theme-panel">
            <label>
                <span>Stage 12 processing job</span>
                <select value={jobId} onChange={event => { const id = event.target.value; if (id) void loadJob(id); }}>
                    <option value="">Select a completed geobody job…</option>
                    {jobOptions.map(option => <option key={option.id} value={option.id}>{option.label}</option>)}
                </select>
            </label>
            <span className="geobody-job-divider">or</span>
            <form onSubmit={event => { event.preventDefault(); void loadJob(jobInput); }}>
                <label><span>Job ID</span><input value={jobInput} onChange={event => setJobInput(event.target.value)} placeholder="Paste Stage 12 job ID"/></label>
                <button className="btn btn-primary" type="submit" disabled={!jobInput.trim() || loading}><Search/> Load</button>
            </form>
        </div>

        {!backendConnected && <div className="geobody-empty"><AlertTriangle/> Backend is offline. Connect to the backend to load geobody products.</div>}
        {loading && <div className="geobody-empty"><RefreshCw className="spin"/> Loading Stage 12 products…</div>}
        {error && <div className="geobody-error"><AlertTriangle/><span>{error}</span></div>}

        {manifest && <>
            <div className="geobody-status-strip">
                <div><span>JOB</span><strong>{jobId}</strong></div>
                <div><span>STAGE STATUS</span><strong className="geobody-status-ok">{statusText(manifest.status)}</strong></div>
                <div><span>RETAINED BODIES</span><strong>{fmt(manifest.counts?.retained_geobodies, 0)}</strong></div>
                <div><span>CANDIDATE VOXELS</span><strong>{fmt(manifest.counts?.candidate_voxels, 0)}</strong></div>
                <div><span>INTERPRETATION</span><strong className="geobody-status-warn">{statusText(manifest.interpretation_status)}</strong></div>
            </div>

            <div className="geobody-export-row theme-panel">
                <div className="geobody-section-title"><div><span className="eyebrow">DELIVERABLES</span><h2>Exports and visualisations</h2></div></div>
                <div className="geobody-export-links">
                    <a className={segyReady('geobody_mask') ? '' : 'disabled'} href={segyReady('geobody_mask') ? `${API_ROOT}${encodeURIComponent(jobId)}/exports/geobody_mask` : undefined} target="_blank" rel="noreferrer"><Download/> SEG-Y mask</a>
                    <a className={segyReady('geobody_id') ? '' : 'disabled'} href={segyReady('geobody_id') ? `${API_ROOT}${encodeURIComponent(jobId)}/exports/geobody_id` : undefined} target="_blank" rel="noreferrer"><Download/> SEG-Y geobody IDs</a>
                    <span className="geobody-export-count"><Box/> {meshes.filter(item => String(item.status).toUpperCase() === 'PASS').length} PLY meshes</span>
                    <button type="button" className="geobody-export-count geobody-open-gallery" onClick={() => setViewerIndex(0)} disabled={!figureItems.length}><ImageIcon/> View figures <b>{figureItems.length}</b></button>
                </div>
            </div>

            <div className="geobody-workspace">
                <section className="geobody-catalog theme-panel">
                    <div className="geobody-section-title"><div><span className="eyebrow">CATALOGUE</span><h2>Geobodies <small>{bodies.length}</small></h2></div></div>
                    <div className="geobody-search"><Search/><input value={search} onChange={event => setSearch(event.target.value)} placeholder="Filter ID, size, status…"/></div>
                    <div className="geobody-list">
                        {filteredBodies.map(body => {
                            const id = Number(body.geobody_id);
                            return <button key={id} className={`geobody-list-item${selectedId === id ? ' selected' : ''}`} onClick={() => setSelectedId(id)}>
                                <span className="geobody-id">GB {String(id).padStart(3, '0')}</span>
                                <strong>{fmt(value(body, 'voxel_count'), 0)} <small>voxels</small></strong>
                                <span className={String(body.closure_status).startsWith('TRUNCATED') ? 'geobody-truncated' : 'geobody-closed'}>{statusText(body.closure_status)}</span>
                            </button>;
                        })}
                        {!filteredBodies.length && <div className="geobody-empty">No geobodies match this filter.</div>}
                    </div>
                </section>

                <section className="geobody-detail theme-panel">
                    {selectedBody ? <>
                        <div className="geobody-section-title">
                            <div><span className="eyebrow">SELECTED BODY</span><h2>Geobody {selectedId}</h2></div>
                            {detail?.mesh && <a className="btn btn-secondary" href={geobodyAssetUrl(jobId, `/items/${selectedId}/mesh`)} target="_blank" rel="noreferrer"><Download/> Download PLY</a>}
                        </div>
                        <div className="geobody-metrics">
                            <div><span>VOXELS</span><strong>{fmt(value(selectedBody, 'voxel_count'), 0)}</strong></div>
                            <div><span>MEAN SCORE</span><strong>{fmt(value(selectedBody, 'mean_score'), 3)}</strong></div>
                            <div><span>SEED VOXELS</span><strong>{fmt(value(selectedBody, 'seed_voxel_count'), 0)}</strong></div>
                            <div><span>COLUMNS</span><strong>{fmt(value(selectedBody, 'distinct_columns'), 0)}</strong></div>
                        </div>
                        <div className="geobody-coordinates">
                            <span>Centroid IL <b>{fmt(value(selectedBody, 'centroid_inline'))}</b></span>
                            <span>XL <b>{fmt(value(selectedBody, 'centroid_crossline'))}</b></span>
                            <span>TWT <b>{fmt(value(selectedBody, 'centroid_twt_ms'))} ms</b></span>
                            <span>Closure <b>{statusText(value(selectedBody, 'closure_status'))}</b></span>
                        </div>
                        <div className="geobody-slice-panel">
                            <div className="geobody-slice-toolbar">
                                <div><span className="eyebrow">LIVE API SLICE</span><h3>{plane.toUpperCase()} SECTION</h3></div>
                                <div className="geobody-plane-buttons">
                                    {(['inline', 'crossline', 'time'] as const).map(p => <button key={p} className={plane === p ? 'active' : ''} onClick={() => {
                                        const field = p === 'inline' ? 'centroid_inline' : p === 'crossline' ? 'centroid_crossline' : 'centroid_twt_ms';
                                        const coord = Number(selectedBody[field]);
                                        setPlane(p);
                                        setCoordinate(String(coord));
                                        void loadSlice(p, coord);
                                    }}>{p === 'time' ? 'TWT' : p.toUpperCase()}</button>)}
                                </div>
                            </div>
                            <form className="geobody-coordinate-form" onSubmit={event => { event.preventDefault(); void loadSlice(plane, Number(coordinate)); }}>
                                <label>Coordinate <input type="number" step="any" value={coordinate} onChange={event => setCoordinate(event.target.value)}/></label>
                                <button className="btn btn-secondary" type="submit" disabled={sliceLoading || coordinate.trim() === ''}><SlidersHorizontal/> Update slice</button>
                            </form>
                            <div className="geobody-slice-canvas">
                                {sliceLoading && <div className="geobody-slice-overlay"><RefreshCw className="spin"/> Loading slice…</div>}
                                {slice ? <svg viewBox={`0 0 ${slice.shape[1]} ${slice.shape[0]}`} role="img" aria-label={`Geobody ${selectedId} ${plane} slice`}>
                                    <path d={maskPath(slice.mask)} className="geobody-slice-cell"/>
                                </svg> : !sliceLoading && <div className="geobody-empty">Choose a body to retrieve its slice.</div>}
                            </div>
                            {slice && <div className="geobody-slice-caption">{slice.body_cells.toLocaleString()} body cells · stride {slice.downsample_stride} · coordinate {fmt(slice.coordinate, 2)}</div>}
                        </div>
                        {detail?.mesh === null && <p className="geobody-note">No mesh was exported for this body. The backend may have skipped it due to configured mesh-size limits.</p>}
                        <div className="geobody-voxel-panel">
                            <div className="geobody-slice-toolbar"><div><span className="eyebrow">3D SPATIAL VIEW</span><h3>Voxel geometry</h3></div><span className="geobody-slice-caption">Authoritative Stage 12 label volume</span></div>
                            {voxelLoading && <div className="geobody-empty">Loading geobody voxels…</div>}
                            {voxelError && <div className="geobody-error"><AlertTriangle/><span>{voxelError}</span></div>}
                            {voxelPreview && <Suspense fallback={<div className="geobody-empty">Loading 3D renderer…</div>}><ProspectVoxelCanvas preview={voxelPreview}/></Suspense>}
                        </div>
                    </> : <div className="geobody-empty">Select a geobody from the catalogue.</div>}
                </section>
            </div>

            {manifest.limitations?.length ? <details className="geobody-limitations theme-panel"><summary>Interpretation limitations</summary><ul>{manifest.limitations.map(item => <li key={item}>{item}</li>)}</ul></details> : null}
        </>}
        {viewerIndex !== null && figureItems[viewerIndex] && <div className="geobody-lightbox" role="presentation" onClick={() => setViewerIndex(null)}>
            <section className="geobody-lightbox-dialog" role="dialog" aria-modal="true" aria-label="Geobody figure viewer" onClick={event => event.stopPropagation()}
                onTouchStart={event => { touchStartX.current = event.changedTouches[0]?.clientX ?? null; }}
                onTouchEnd={event => {
                    const start = touchStartX.current;
                    const end = event.changedTouches[0]?.clientX;
                    touchStartX.current = null;
                    if (start === null || end === undefined || Math.abs(end - start) < 45 || figureItems.length < 2) return;
                    setViewerIndex(index => index === null ? null : (index + (end < start ? 1 : -1) + figureItems.length) % figureItems.length);
                }}>
                <header className="geobody-lightbox-head">
                    <div><span className="eyebrow">STAGE 12 FIGURES</span><strong>{figureItems[viewerIndex].title}</strong></div>
                    <button type="button" className="geobody-lightbox-close" onClick={() => setViewerIndex(null)} aria-label="Close figure viewer">×</button>
                </header>
                <div className="geobody-lightbox-image-wrap">
                    {figureItems.length > 1 && <button type="button" className="geobody-lightbox-nav previous" onClick={() => setViewerIndex((viewerIndex - 1 + figureItems.length) % figureItems.length)} aria-label="Previous figure">‹</button>}
                    <img src={figureItems[viewerIndex].url} alt={figureItems[viewerIndex].title}/>
                    {figureItems.length > 1 && <button type="button" className="geobody-lightbox-nav next" onClick={() => setViewerIndex((viewerIndex + 1) % figureItems.length)} aria-label="Next figure">›</button>}
                </div>
                <footer className="geobody-lightbox-foot">
                    <button type="button" onClick={() => setViewerIndex((viewerIndex - 1 + figureItems.length) % figureItems.length)} disabled={figureItems.length < 2}>Previous</button>
                    <span>{viewerIndex + 1} / {figureItems.length}</span>
                    <button type="button" onClick={() => setViewerIndex((viewerIndex + 1) % figureItems.length)} disabled={figureItems.length < 2}>Next</button>
                </footer>
            </section>
        </div>}
    </section>;
}

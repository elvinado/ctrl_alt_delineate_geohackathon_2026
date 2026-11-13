import { useEffect, useMemo, useRef, useState } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import { getDatasetTable, getSeismicGeometry, getSeismicSection } from '../../api/backend';
import type { BackendSeismicGeometry, BackendSeismicSection } from '../../api/backend';
import type { BackendDatasetTable } from '../../api/backend';
import type { SeismicData, TableData, WellData } from '../data';

export function MiniSeismic({ large = false, data, datasetId, zoomX = 1, zoomY = 1 }: {
    large?: boolean;
    data?: SeismicData;
    datasetId?: string;
    zoomX?: number;
    zoomY?: number;
}) {
    const [cursor, setCursor] = useState<{
        x: number;
        y: number;
    } | null>(null);
    const [localZoomX, setLocalZoomX] = useState(zoomX);
    const [localZoomY, setLocalZoomY] = useState(zoomY);
    const [display, setDisplay] = useState<'Grayscale' | 'Red-blue' | 'Wiggle'>('Grayscale');
    const [gain, setGain] = useState(1.4);
    const [geometry, setGeometry] = useState<BackendSeismicGeometry | null>(null);
    const [remoteSection, setRemoteSection] = useState<BackendSeismicSection | null>(null);
    const [sectionLoading, setSectionLoading] = useState(false);
    const [sectionError, setSectionError] = useState('');
    const [sectionMode, setSectionMode] = useState<'Inline' | 'Crossline' | 'Time slice'>('Inline');
    const [sectionPos, setSectionPos] = useState(0);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const inlineValues = useMemo(() => geometry?.inlines ?? [...new Set((data?.traceHeaders ?? []).map(h => h.inline).filter(Boolean))].sort((a, b) => a - b), [data, geometry]);
    const crosslineValues = useMemo(() => geometry?.crosslines ?? [...new Set((data?.traceHeaders ?? []).map(h => h.crossline).filter(Boolean))].sort((a, b) => a - b), [data, geometry]);
    const previewSamples = data?.traces[0]?.length ?? 1;
    const sectionMax = sectionMode === 'Inline' ? Math.max(0, inlineValues.length - 1) : sectionMode === 'Crossline' ? Math.max(0, crosslineValues.length - 1) : Math.max(0, (geometry?.samples_per_trace ?? previewSamples) - 1);
    useEffect(() => { let cancelled = false; setGeometry(null); setRemoteSection(null); setSectionError(''); setSectionPos(0); if (datasetId) getSeismicGeometry(datasetId).then(result => { if (!cancelled) { setGeometry(result); setSectionPos(0); } }).catch(error => { if (!cancelled) setSectionError(error instanceof Error ? error.message : String(error)); }); return () => { cancelled = true; }; }, [datasetId]);
    useEffect(() => { setSectionPos(0); setRemoteSection(null); }, [data, sectionMode, geometry?.active_inline_byte, geometry?.active_crossline_byte]);
    useEffect(() => { if (!datasetId || !geometry) return; let cancelled = false; const mode = sectionMode === 'Inline' ? 'inline' : sectionMode === 'Crossline' ? 'crossline' : 'timeslice'; const target = mode === 'inline' ? inlineValues[sectionPos] : mode === 'crossline' ? crosslineValues[sectionPos] : sectionPos; if (target === undefined) return; setSectionLoading(true); setSectionError(''); setRemoteSection(null); getSeismicSection(datasetId, { mode, target, inlineByte: geometry.active_inline_byte, crosslineByte: geometry.active_crossline_byte }).then(result => { if (!cancelled) setRemoteSection(result); }).catch(error => { if (!cancelled) setSectionError(error instanceof Error ? error.message : String(error)); }).finally(() => { if (!cancelled) setSectionLoading(false); }); return () => { cancelled = true; }; }, [datasetId, geometry, sectionMode, sectionPos, inlineValues, crosslineValues]);
    const section = useMemo(() => {
        if (remoteSection) {
            const isSlice = remoteSection.mode === 'timeslice';
            const x = remoteSection.x_positions;
            const y = remoteSection.y_positions ?? [];
            const start = remoteSection.start_ms / 1000;
            const end = start + Math.max(0, remoteSection.sample_count - 1) * (remoteSection.sample_stride ?? 1) * remoteSection.sample_interval_us / 1e6;
            return { traces: remoteSection.traces, xRange: [x[0] ?? 0, x.at(-1) ?? 1] as [number, number], yRange: isSlice ? [y[0] ?? 0, y.at(-1) ?? 1] as [number, number] : [start, end] as [number, number], xTitle: isSlice ? 'CROSSLINE' : sectionMode === 'Inline' ? 'CROSSLINE' : 'INLINE', yTitle: isSlice ? 'INLINE' : 'TWT (s)', value: isSlice ? `${(start + sectionPos * remoteSection.sample_interval_us / 1e6).toFixed(3)} s TWT` : `${sectionMode} ${remoteSection.target}` };
        }
        if (!data)
            return { traces: [] as number[][], xRange: [0, 100] as [
                    number,
                    number
                ], yRange: [0, 2.4] as [
                    number,
                    number
                ], xTitle: 'TRACE', yTitle: 'TWT (s)', value: 'No dataset selected' };
        if (sectionMode === 'Time slice') {
            const xs = crosslineValues.length > 160 ? crosslineValues.filter((_, i) => i % Math.ceil(crosslineValues.length / 160) === 0) : crosslineValues;
            const ys = inlineValues.length > 160 ? inlineValues.filter((_, i) => i % Math.ceil(inlineValues.length / 160) === 0) : inlineValues;
            const lookup = new Map(data.traceHeaders.map((h, i) => [`${h.inline}:${h.crossline}`, data.traces[i]]));
            return { traces: xs.map(x => ys.map(y => lookup.get(`${y}:${x}`)?.[sectionPos] ?? 0)), xRange: [xs[0] ?? 0, xs.at(-1) ?? 1] as [
                    number,
                    number
                ], yRange: [ys[0] ?? 0, ys.at(-1) ?? 1] as [
                    number,
                    number
                ], xTitle: 'CROSSLINE', yTitle: 'INLINE', value: `${(sectionPos / Math.max(1, previewSamples - 1) * (data.samplesPerTrace * data.sampleIntervalUs / 1e6)).toFixed(3)} s TWT` };
        }
        const target = sectionMode === 'Inline' ? inlineValues[sectionPos] : crosslineValues[sectionPos];
        const indices = data.traceHeaders.map((h, i) => ({ h, i })).filter(({ h }) => sectionMode === 'Inline' ? h.inline === target : h.crossline === target).sort((a, b) => sectionMode === 'Inline' ? a.h.crossline - b.h.crossline : a.h.inline - b.h.inline);
        const lateral = indices.map(({ h }) => sectionMode === 'Inline' ? h.crossline : h.inline);
        return { traces: indices.map(({ i }) => data.traces[i]), xRange: [lateral[0] ?? 0, lateral.at(-1) ?? 1] as [
                number,
                number
            ], yRange: [0, data.samplesPerTrace * data.sampleIntervalUs / 1e6] as [
                number,
                number
            ], xTitle: sectionMode === 'Inline' ? 'CROSSLINE' : 'INLINE', yTitle: 'TWT (s)', value: `${sectionMode} ${target ?? '—'}` };
    }, [data, sectionMode, sectionPos, inlineValues, crosslineValues, previewSamples, remoteSection]);
    const real = section.traces.map((trace, i) => {
        const base = section.traces.length === 1 ? 50 : i / (section.traces.length - 1) * 100;
        const spacing = 100 / Math.max(1, section.traces.length - 1);
        return trace.map((amp, j) => { const value = typeof amp === 'number' && Number.isFinite(amp) ? Math.max(-1, Math.min(1, amp * gain)) : 0; return `${j ? 'L' : 'M'} ${(base + value * spacing * .48).toFixed(2)} ${(j / Math.max(1, trace.length - 1) * 100).toFixed(2)}`; }).join(' ');
    });
    const traces = real;
    const width = 100 / localZoomX, height = 100 / localZoomY, viewX = (100 - width) / 2, viewY = (100 - height) / 2;
    const xRange = section.xRange;
    const xMid = (xRange[0] + xRange[1]) / 2, xSpan = (xRange[1] - xRange[0]) / localZoomX;
    const xStart = xMid - xSpan / 2, xEnd = xMid + xSpan / 2;
    const yMid = (section.yRange[0] + section.yRange[1]) / 2, ySpan = (section.yRange[1] - section.yRange[0]) / localZoomY;
    const yStart = yMid - ySpan / 2, yEnd = yMid + ySpan / 2;
    const ticks = [0, 1, 2, 3, 4];
    const cursorLabel = cursor ? `${section.xTitle} ${Math.round(xStart + cursor.x * (xEnd - xStart))} · ${section.yTitle} ${sectionMode === 'Time slice' ? Math.round(yStart + cursor.y * (yEnd - yStart)) : (yStart + cursor.y * (yEnd - yStart)).toFixed(3)}` : 'Move cursor over section';
    useEffect(() => {
        if (!large || (!data && !remoteSection) || (datasetId && geometry && !remoteSection) || display === 'Wiggle' || !canvasRef.current)
            return;
        const canvas = canvasRef.current, w = 900, h = 520;
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        if (!ctx)
            return;
        const image = ctx.createImageData(w, h), traceN = section.traces.length, sampleN = section.traces[0]?.length ?? 0;
        if (!traceN || !sampleN)
            return;
        const visibleTraces = traceN / localZoomX, traceStart = (traceN - visibleTraces) / 2;
        const visibleSamples = sampleN / localZoomY, sampleStart = (sampleN - visibleSamples) / 2;
        const valueAt = (x: number, y: number) => {
            const tx = Math.max(0, Math.min(traceN - 1, traceStart + x / (w - 1) * Math.max(0, visibleTraces - 1)));
            const sy = Math.max(0, Math.min(sampleN - 1, sampleStart + y / (h - 1) * Math.max(0, visibleSamples - 1)));
            const x0 = Math.floor(tx), x1 = Math.min(traceN - 1, x0 + 1), y0 = Math.floor(sy), y1 = Math.min(sampleN - 1, y0 + 1);
            const fx = tx - x0, fy = sy - y0;
            const sample = (trace: number, sampleIndex: number) => {
                const value = section.traces[trace]?.[sampleIndex];
                return typeof value === 'number' && Number.isFinite(value) ? value : 0;
            };
            const a = sample(x0, y0) * (1 - fx) + sample(x1, y0) * fx;
            const b = sample(x0, y1) * (1 - fx) + sample(x1, y1) * fx;
            return Math.max(-1, Math.min(1, (a * (1 - fy) + b * fy) * gain));
        };
        let row = 0;
        let frame = 0;
        const drawChunk = () => {
          const stop = Math.min(h, row + 24);
          for (let y = row; y < stop; y++)
            for (let x = 0; x < w; x++) {
                const amp = valueAt(x, y), p = (y * w + x) * 4;
                if (display === 'Grayscale') {
                    const c = Math.round(138 - amp * 112);
                    image.data[p] = c;
                    image.data[p + 1] = c;
                    image.data[p + 2] = c;
                }
                else if (amp >= 0) {
                    image.data[p] = Math.round(245 - amp * 190);
                    image.data[p + 1] = Math.round(245 - amp * 215);
                    image.data[p + 2] = Math.round(245 - amp * 205);
                }
                else {
                    const n = -amp;
                    image.data[p] = Math.round(245 - n * 205);
                    image.data[p + 1] = Math.round(245 - n * 155);
                    image.data[p + 2] = Math.round(245 - n * 45);
                }
                image.data[p + 3] = 255;
            }
          row = stop;
          if (row < h) frame = requestAnimationFrame(drawChunk);
          else ctx.putImageData(image, 0, 0);
        };
        frame = requestAnimationFrame(drawChunk);
        return () => cancelAnimationFrame(frame);
    }, [data, datasetId, display, gain, geometry, large, localZoomX, localZoomY, section, remoteSection]);
    const updateCursor = (e: ReactMouseEvent<SVGSVGElement | HTMLCanvasElement>) => { const r = e.currentTarget.getBoundingClientRect(); setCursor({ x: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), y: Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)) }); };
    return <div className={`seismic-canvas ${large ? 'large calibrated' : ''}`}>
    {large && <>{geometry && <div className="geometry-mapping"><span>GEOMETRY</span><select aria-label="Inline and crossline trace-header mapping" value={`${geometry.active_inline_byte}:${geometry.active_crossline_byte}`} onChange={e => { const [ib, xb] = e.target.value.split(':').map(Number); if (datasetId) { setSectionLoading(true); getSeismicGeometry(datasetId, ib, xb).then(setGeometry).catch(error => setSectionError(error instanceof Error ? error.message : String(error))).finally(() => setSectionLoading(false)); } }}>
      {geometry.geometry_candidates.map(candidate => <option key={`${candidate.inline_byte}:${candidate.crossline_byte}`} value={`${candidate.inline_byte}:${candidate.crossline_byte}`}>IL {candidate.inline_byte} / XL {candidate.crossline_byte} · {candidate.label}</option>)}
      </select><b>{geometry.geometry_confidence}</b></div>}<div className="canvas-controls"><div className="slice-controls"><div className="slice-tabs">{(['Inline', 'Crossline', 'Time slice'] as const).map(mode => <button key={mode} className={`btn-segment ${sectionMode === mode ? 'active' : ''}`} onClick={() => setSectionMode(mode)}>{mode}</button>)}</div><input aria-label={`${sectionMode} position`} type="range" min="0" max={sectionMax} value={Math.min(sectionPos, sectionMax)} onChange={e => setSectionPos(Number(e.target.value))}/><strong>{section.value}</strong><small>{geometry ? `${geometry.trace_count.toLocaleString()} full-volume traces indexed · ${remoteSection?.present ?? 0}/${remoteSection?.total ?? 0} section traces present` : data ? `${data.scannedTraceCount.toLocaleString()} preview traces` : 'No dataset loaded'}</small>{sectionLoading && <small role="status">Loading section…</small>}{sectionError && <small className="error-text" role="alert">{sectionError}</small>}</div><div className="render-controls"><div><span>GAIN</span><button className="btn-icon btn-icon-compact" aria-label="Decrease display gain" onClick={() => setGain(v => Math.max(.5, +(v - .2).toFixed(1)))}>−</button><b>{gain.toFixed(1)}×</b><button className="btn-icon btn-icon-compact" aria-label="Increase display gain" onClick={() => setGain(v => Math.min(4, +(v + .2).toFixed(1)))}>+</button></div><div className="display-mode"><span>DISPLAY</span><select aria-label="Seismic display style" value={display} onChange={e => setDisplay(e.target.value as typeof display)}><option>Grayscale</option><option>Red-blue</option><option>Wiggle</option></select></div><div><span>HORIZONTAL</span><button className="btn-icon btn-icon-compact" aria-label="Zoom out horizontally" title="Zoom out horizontally" onClick={() => setLocalZoomX(v => Math.max(1, +(v - .5).toFixed(1)))}>−</button><b>{localZoomX.toFixed(1)}×</b><button className="btn-icon btn-icon-compact" aria-label="Zoom in horizontally" title="Zoom in horizontally" onClick={() => setLocalZoomX(v => Math.min(8, +(v + .5).toFixed(1)))}>+</button></div><div><span>VERTICAL</span><button className="btn-icon btn-icon-compact" aria-label="Zoom out vertically" title="Zoom out vertically" onClick={() => setLocalZoomY(v => Math.max(1, +(v - .5).toFixed(1)))}>−</button><b>{localZoomY.toFixed(1)}×</b><button className="btn-icon btn-icon-compact" aria-label="Zoom in vertically" title="Zoom in vertically" onClick={() => setLocalZoomY(v => Math.min(8, +(v + .5).toFixed(1)))}>+</button></div><button className="btn btn-secondary btn-compact reset-view" aria-label="Reset seismic view" onClick={() => { setLocalZoomX(1); setLocalZoomY(1); setGain(1.4); }}><RotateCcw />Reset</button></div></div></>}
    {large && <><div className="scale-title-x">{section.xTitle}</div><div className="scale-title-y">{section.yTitle}</div><div className="scale-x">{ticks.map(t => <span key={t} style={{ left: `${t * 25}%` }}>{Math.round(xStart + t / 4 * (xEnd - xStart))}</span>)}</div><div className="scale-y">{ticks.map(t => <span key={t} style={{ top: `${t * 25}%` }}>{sectionMode === 'Time slice' ? Math.round(yStart + t / 4 * (yEnd - yStart)) : (yStart + t / 4 * (yEnd - yStart)).toFixed(2)}</span>)}</div></>}
    {!data && !geometry && <div className="visual-empty">Select a SEG-Y dataset to display seismic traces.</div>}
    {large && (datasetId ? !!remoteSection : !!data) && display !== 'Wiggle' && <canvas ref={canvasRef} aria-label={`${display} seismic variable-density display`} onMouseMove={updateCursor} onMouseLeave={() => setCursor(null)}/>}
    {(datasetId ? !!remoteSection : !!data) && (!large || display === 'Wiggle') && <svg viewBox={`${viewX} ${viewY} ${width} ${height}`} preserveAspectRatio="none" aria-label="Parsed seismic trace preview" onMouseMove={large ? updateCursor : undefined} onMouseLeave={() => setCursor(null)}>
      <defs><linearGradient id="sea" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#091d22"/><stop offset="1" stopColor="#102e2a"/></linearGradient></defs>
      <rect width="100" height="100" fill="url(#sea)"/>
      {large && ticks.slice(1).map(t => <line key={`v${t}`} x1={t * 25} x2={t * 25} y1="0" y2="100" stroke="#aac8bf" strokeOpacity=".10" strokeWidth=".15"/>)}
      {large && ticks.slice(1).map(t => <line key={`h${t}`} x1="0" x2="100" y1={t * 25} y2={t * 25} stroke="#aac8bf" strokeOpacity=".10" strokeWidth=".15"/>)}
      {traces.map((d, i) => <path key={i} d={d} fill="none" stroke={i % 2 ? '#8ad8c4' : '#e8e4d4'} strokeOpacity={.72} strokeWidth=".18"/>)}
      {large && cursor && <><line x1={viewX + cursor.x * width} x2={viewX + cursor.x * width} y1={viewY} y2={viewY + height} stroke="#ffe1a8" strokeWidth=".18" vectorEffect="non-scaling-stroke"/><line x1={viewX} x2={viewX + width} y1={viewY + cursor.y * height} y2={viewY + cursor.y * height} stroke="#ffe1a8" strokeWidth=".18" vectorEffect="non-scaling-stroke"/></>}
    </svg>}
    {large && (datasetId ? !!remoteSection : !!data) && display !== 'Wiggle' && <svg className="density-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">{ticks.slice(1).map(t => <line key={`dv${t}`} x1={t * 25} x2={t * 25} y1="0" y2="100" stroke="#10221d" strokeOpacity=".22" strokeWidth=".18"/>)}{ticks.slice(1).map(t => <line key={`dh${t}`} x1="0" x2="100" y1={t * 25} y2={t * 25} stroke="#10221d" strokeOpacity=".22" strokeWidth=".18"/>)}{cursor && <><line x1={cursor.x * 100} x2={cursor.x * 100} y1="0" y2="100" stroke="#ffd279" strokeWidth=".22"/><line x1="0" x2="100" y1={cursor.y * 100} y2={cursor.y * 100} stroke="#ffd279" strokeWidth=".22"/></>}</svg>}
    <div className="seismic-label">{geometry ? `${geometry.trace_count.toLocaleString()} traces · ${geometry.sample_interval_us} µs · ${geometry.format_code}` : data ? `${data.traceCount.toLocaleString()} traces · ${data.sampleIntervalUs} µs` : 'No seismic data'}</div>
    {large && <div className="cursor-readout">{cursorLabel}</div>}
  </div>;
}
export function WellTrack({ data, offset = 0 }: {
    data?: WellData;
    offset?: number;
}) {
    const colors = ['#e4b36e', '#71cbb7', '#b983ee'];
    const requestedCols = [1 + offset * 3, 2 + offset * 3, 3 + offset * 3].filter(col => data && col < data.curves.length);
    const selectedCols = requestedCols.length ? requestedCols : [1, 2, 3].filter(col => data && col < data.curves.length);
    const realPaths = data ? selectedCols.map((col, pathIndex) => {
        const values = data.rows.map(r => r[col]).filter(Number.isFinite);
        const min = Math.min(...values);
        const max = Math.max(...values);
        const span = max - min || 1;
        return { color: colors[pathIndex], name: data.curves[col], d: data.rows.map((row, i) => `${i ? 'L' : 'M'} ${((row[col] - min) / span * 26 + 4 + pathIndex * 33).toFixed(2)} ${(i / Math.max(1, data.rows.length - 1) * 160).toFixed(2)}`).join(' ') };
    }) : [];
    return <div className="well-track theme-inset">
    <div className="track-head theme-meta">{realPaths.map(p => <span key={p.name}>{p.name}</span>)}</div>
    {data && realPaths.length ? <svg viewBox="0 0 100 160" preserveAspectRatio="none" aria-label="Parsed well log preview">
      {[20, 40, 60, 80].map(x => <line key={x} x1={x} x2={x} y1="0" y2="160" stroke="#2c4640" strokeWidth=".4"/>)}
      {realPaths.map(p => <path key={p.name} d={p.d} fill="none" stroke={p.color} strokeWidth="1.15"/>)}
    </svg> : <div className="visual-empty">Select a LAS dataset to display well curves.</div>}
  </div>;
}
export function TablePreview({ data }: {
    data?: TableData;
}) {
    if (!data)
        return <div className="visual-empty">Select a CSV or TSV dataset to preview its rows.</div>;
    const headers = data.headers.slice(0, 4);
    const rows = data.rows.slice(0, 3);
    return <div className="csv-preview theme-inset"><div>{headers.map(h => <span key={h}>{h}</span>)}</div>{rows.map((row, i) => <p key={i}>{row.slice(0, 4).map((v, j) => <span key={j}>{v}</span>)}</p>)}</div>;
}
export function WellTableApiViewer({ datasetId, kind, datasetName, compact = false }: {
    datasetId: string;
    kind: 'deviation' | 'checkshot';
    datasetName: string;
    compact?: boolean;
}) {
    const pageSize = 100;
    const [offset, setOffset] = useState(0);
    const [table, setTable] = useState<BackendDatasetTable | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    useEffect(() => { setOffset(0); setTable(null); setError(''); }, [datasetId]);
    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        setError('');
        getDatasetTable(datasetId, offset, pageSize)
            .then(result => { if (!cancelled)
            setTable(result); })
            .catch(reason => { if (!cancelled)
            setError(reason instanceof Error ? reason.message : String(reason)); })
            .finally(() => { if (!cancelled)
            setLoading(false); });
        return () => { cancelled = true; };
    }, [datasetId, offset]);
    const columns = table?.columns ?? [];
    const rows = table?.rows ?? [];
    const rowCount = table?.row_count ?? 0;
    const firstRow = rowCount ? offset + 1 : 0;
    const lastRow = Math.min(offset + (table?.returned_rows ?? 0), rowCount);
    const findColumn = (aliases: string[]) => columns.find(column => aliases.includes(column.name.toLowerCase().replace(/[^a-z0-9]/g, '')));
    const valueFor = (row: Record<string, unknown>, aliases: string[]) => {
        const column = findColumn(aliases);
        const value = column ? Number(row[column.name]) : NaN;
        return Number.isFinite(value) ? value : undefined;
    };
    const chartRows = rows.filter(row => Object.values(row).some(value => typeof value === 'number' && Number.isFinite(value)));
    const depthAliases = ['md', 'mdkb', 'measureddepth', 'depth'];
    const mdValues = chartRows.map(row => valueFor(row, depthAliases)).filter((value): value is number => value !== undefined);
    const depthMin = mdValues.length ? Math.min(...mdValues) : 0;
    const depthMax = mdValues.length ? Math.max(...mdValues) : 1;
    const chartWidth = 640;
    const chartHeight = 230;
    const plotPoints = (xAliases: string[], yAliases: string[]) => {
        const points = chartRows.map(row => ({ x: valueFor(row, xAliases), y: valueFor(row, yAliases) })).filter((point): point is { x: number; y: number } => point.x !== undefined && point.y !== undefined);
        if (points.length < 2) return '';
        const xMin = Math.min(...points.map(point => point.x));
        const xMax = Math.max(...points.map(point => point.x));
        const yMin = Math.min(...points.map(point => point.y));
        const yMax = Math.max(...points.map(point => point.y));
        return points.map((point, index) => `${index ? 'L' : 'M'} ${(36 + (point.x - xMin) / (xMax - xMin || 1) * (chartWidth - 56)).toFixed(1)} ${(12 + (1 - (point.y - yMin) / (yMax - yMin || 1)) * (chartHeight - 42)).toFixed(1)}`).join(' ');
    };
    const mdAliases = ['md', 'mdkb', 'measureddepth', 'depth'];
    const inclinationAliases = ['incl', 'inc', 'inclination'];
    const azimuthAliases = ['azim', 'azi', 'azimuth'];
    const twtAliases = ['twt', 'two way time', 'twowaytime', 'time', 'twtms'];
    const tvdAliases = ['tvd', 'tvdss', 'trueverticaldepth'];
    const deviationPath = kind === 'deviation' ? plotPoints(mdAliases, inclinationAliases) : '';
    const deviationAzimuthPath = kind === 'deviation' ? plotPoints(mdAliases, azimuthAliases) : '';
    const checkshotPath = kind === 'checkshot' ? plotPoints(mdAliases, twtAliases) : '';
    const checkshotTvdPath = kind === 'checkshot' ? plotPoints(tvdAliases, twtAliases) : '';
    if (compact) {
        const profilePath = kind === 'deviation' ? deviationPath : (checkshotPath || checkshotTvdPath);
        const plotLabel = kind === 'deviation' ? 'Inclination vs measured depth' : 'Two-way travel time vs depth';
        return <div className={`input-data-survey-preview ${kind}`} aria-label={kind === 'deviation' ? 'Deviation profile preview' : 'Checkshot time-depth preview'}>
            {error ? <div className="well-table-error"><AlertTriangle /><span>{error}</span></div> : <div className="well-survey-chart">
                <span>{plotLabel}</span>
                {profilePath ? <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} role="img" aria-label={kind === 'deviation' ? 'Inclination deviation profile plotted against measured depth' : 'Checkshot time-depth plot'}>
                    <path className="chart-gridline" d={`M36 12H${chartWidth-20} M36 ${chartHeight/2}H${chartWidth-20} M36 ${chartHeight-30}H${chartWidth-20}`} />
                    <path className="chart-axis" d={`M36 12V${chartHeight-30}H${chartWidth-20}`} />
                    <path className="well-survey-line" d={profilePath} />
                    <text x="36" y={chartHeight-8}>{kind === 'deviation' ? `MD ${depthMin.toFixed(1)}` : 'Depth'}</text>
                    <text x={chartWidth-20} y={chartHeight-8} textAnchor="end">{kind === 'deviation' ? depthMax.toFixed(1) : (checkshotPath ? 'MD' : 'TVD')}</text>
                </svg> : <p className="well-survey-empty">{loading ? 'Loading preview…' : kind === 'deviation' ? 'Inclination and measured-depth columns were not identified.' : 'Depth and two-way travel-time columns were not identified.'}</p>}
            </div>}
        </div>;
    }
    return <section className={`well-table-api theme-panel`} aria-label={`${kind} parsed table`}>
    <div className="well-table-api-head">
      <div><span className="eyebrow">{kind === 'deviation' ? 'WELL TRAJECTORY' : 'TIME–DEPTH CALIBRATION'} · PARSED API DATA</span><h2>{kind === 'deviation' ? 'Deviation survey' : 'Checkshot survey'}</h2><p>{datasetName} · Original TXT file is retained; rows below are parsed from the backend.</p></div>
      <div className="well-table-count"><strong>{rowCount.toLocaleString()}</strong><span>total rows</span></div>
    </div>
    <div className="well-table-metrics theme-meta">
      {(columns.length ? columns : (kind === 'deviation' ? [{ name: 'MDKB', source_name: 'MDKB', unit: 'm' }, { name: 'AZIM', source_name: 'AZIM', unit: null }, { name: 'INCL', source_name: 'INCL', unit: null }] : [{ name: 'MDKB', source_name: 'MDKB', unit: null }, { name: 'TWT', source_name: 'TWT', unit: null }, { name: 'TVDSS', source_name: 'TVDSS', unit: null }])).map(column => <div key={column.name}><span>{column.name}</span><strong>{column.unit ? `Unit: ${column.unit}` : 'Numeric values'}</strong><small>Source label: {column.source_name}</small></div>)}
    </div>
    {error ? <div className="well-table-error"><AlertTriangle /><span><strong>Could not retrieve parsed table</strong><small>{error}</small></span></div> : <>
      <div className="well-survey-visualization theme-panel">
        <div className="well-survey-visual-head"><strong>{kind === 'deviation' ? 'Deviation profile' : 'Checkshot time–depth plot'}</strong><span>{loading ? 'Updating plot…' : `Visualizing ${chartRows.length} rows on this page`}</span></div>
        {kind === 'deviation' ? <div className="well-survey-chart-grid">
          <div className="well-survey-chart"><span>Inclination vs measured depth</span>{deviationPath ? <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} role="img" aria-label="Inclination plotted against measured depth"><path className="chart-gridline" d={`M36 12H${chartWidth-20} M36 ${chartHeight/2}H${chartWidth-20} M36 ${chartHeight-30}H${chartWidth-20}`}/><path className="chart-axis" d={`M36 12V${chartHeight-30}H${chartWidth-20}`}/><path className="well-survey-line" d={deviationPath}/><text x="36" y={chartHeight-8}>MD {depthMin.toFixed(1)}</text><text x={chartWidth-20} y={chartHeight-8} textAnchor="end">{depthMax.toFixed(1)}</text></svg> : <p className="well-survey-empty">Inclination and measured-depth columns were not identified.</p>}</div>
          <div className="well-survey-chart"><span>Azimuth vs measured depth</span>{deviationAzimuthPath ? <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} role="img" aria-label="Azimuth plotted against measured depth"><path className="chart-gridline" d={`M36 12H${chartWidth-20} M36 ${chartHeight/2}H${chartWidth-20} M36 ${chartHeight-30}H${chartWidth-20}`}/><path className="chart-axis" d={`M36 12V${chartHeight-30}H${chartWidth-20}`}/><path className="well-survey-line secondary-line" d={deviationAzimuthPath}/><text x="36" y={chartHeight-8}>MD {depthMin.toFixed(1)}</text><text x={chartWidth-20} y={chartHeight-8} textAnchor="end">{depthMax.toFixed(1)}</text></svg> : <p className="well-survey-empty">Azimuth and measured-depth columns were not identified.</p>}</div>
        </div> : <div className="well-survey-chart"><span>Two-way travel time vs measured depth</span>{(checkshotPath || checkshotTvdPath) ? <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} role="img" aria-label="Checkshot two-way travel time plotted against depth"><path className="chart-gridline" d={`M36 12H${chartWidth-20} M36 ${chartHeight/2}H${chartWidth-20} M36 ${chartHeight-30}H${chartWidth-20}`}/><path className="chart-axis" d={`M36 12V${chartHeight-30}H${chartWidth-20}`}/><path className="well-survey-line" d={checkshotPath || checkshotTvdPath}/><text x="36" y={chartHeight-8}>Depth</text><text x={chartWidth-20} y={chartHeight-8} textAnchor="end">{checkshotPath ? 'MD' : 'TVD'}</text></svg> : <p className="well-survey-empty">Measured-depth and TWT columns were not identified. Check the table below for available fields.</p>}</div>}
        <small className="well-survey-note">Plots use the currently loaded table page (up to 100 rows). Use pagination to inspect other intervals.</small>
      </div>
      <div className="well-table-scroll"><table><thead><tr><th>#</th>{columns.map(column => <th key={column.name}>{column.name}{column.unit ? ` (${column.unit})` : ''}</th>)}</tr></thead><tbody>
        {rows.map((row, index) => <tr key={`${offset + index}-${JSON.stringify(row)}`}><td>{offset + index + 1}</td>{columns.map(column => <td key={column.name}>{typeof row[column.name] === 'number' ? row[column.name].toLocaleString(undefined, { maximumFractionDigits: 6 }) : String(row[column.name] ?? '—')}</td>)}</tr>)}
        {!loading && rows.length === 0 && <tr><td colSpan={Math.max(1, columns.length + 1)} className="well-table-empty">No parsed rows returned for this page.</td></tr>}
      </tbody></table></div>
      <div className="well-table-pagination theme-meta"><span>{loading ? 'Loading parsed rows…' : `Showing ${firstRow.toLocaleString()}–${lastRow.toLocaleString()} of ${rowCount.toLocaleString()} rows`}</span><div><button className="btn btn-secondary" onClick={() => setOffset(Math.max(0, offset - pageSize))} disabled={loading || offset === 0}>Previous 100</button><button className="btn btn-secondary" onClick={() => setOffset(offset + pageSize)} disabled={loading || offset + pageSize >= rowCount}>Next 100</button></div></div>
    </>}
  </section>;
}

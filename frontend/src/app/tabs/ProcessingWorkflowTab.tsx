import { Suspense, lazy, useEffect, useRef, useState } from 'react';
import {
    Activity, ArrowRight, BrainCircuit, Code2, Database, Layers3, MessageSquare,
    Target, Workflow, Box, FileCode2, FileOutput, Image as ImageIcon,
} from 'lucide-react';
import GeobodyTab from './GeobodyTab';
const ProspectVoxelCanvas = lazy(() => import('./ProspectVoxelCanvas'));
import { getProcessingConfig, updateProcessingConfig, listModelReleases, listProcessingJobs, type ProcessingConfigValue, type BackendProcessingConfig, type ModelReleaseListItem, type ProcessingJobListItem, getInferenceSlice, type InferenceSliceResponse, listVolumeProductArtifacts, getVolumeProductArtifact, volumeProductArtifactUrl, type VolumeProductArtifact, listStageResultArtifacts, getStageResultArtifact, stageResultArtifactUrl, getProspectVoxelPreview, type ProspectVoxelPreview, type StageResultArtifact } from '../../api/backend';

type StageId = 'foundations' | 'inference' | 'volume-products' | 'geobody' | 'well-geobody' | 'prospect';
type Stage = {
    id: StageId;
    letter: string;
    title: string;
    short: string;
    description: string;
    script: string;
    config: string;
    inputs: string[];
    outputs: string[];
    settings: string[];
    icon: typeof Database;
};

const stages: Stage[] = [
    {
        id: 'foundations', letter: 'A', title: 'Foundations',
        short: 'Register, QC, tie, label and train',
        description: 'Build the project data foundation and train candidate models. The worker executes GeoGG Stages 00–07 in sequence.',
        script: 'process_A_foundations.py', config: 'A_foundations.yaml',
        inputs: [
            'Backend-selected dataset manifest: registered seismic volumes (full, near, mid, far, ultrafar), well LAS logs, deviation surveys and checkshots',
            'Optional registered well-header workbook and explicit seismic role mappings',
            'Survey geometry, well associations and project metadata from the backend registry',
        ],
        outputs: [
            'Inventory and QC: data manifest, well/log/seismic inventories, SEG-Y geometry and stack-consistency checks',
            'Well ties: minimum-curvature trajectories, T–Z tables, velocity QC and synthetic-seismic tie reports',
            'Petrophysical labels and evidence; aligned well/seismic samples and native-grid training patches',
            'Feature tables, fold-local feature selection, nested whole-well model evaluation and model registry',
            'Immutable model release (only when at least one final model succeeds), including model files, schemas, transforms and provenance',
        ],
        settings: ['Stages 00–07', 'Whole-well model validation', 'Release is conditional on successful final models'],
        icon: Database,
    },
    {
        id: 'inference', letter: 'B', title: 'Inference',
        short: 'Apply a released model to seismic',
        description: 'Load an immutable model release and generate predictions for the selected survey-volume inputs. Training is not repeated here.',
        script: 'process_B_inference.py', config: 'B_inference.yaml',
        inputs: [
            'Immutable model release ID created by Foundations',
            'Original registered seismic stack datasets resolved through the release’s Foundation job',
            'Inference selection: task, input representation and model (for example cnn3d_multitask or extratrees)',
            'Optional inline/crossline region of interest; empty region means the full survey',
        ],
        outputs: [
            'Survey-volume prediction arrays and spatial geometry/provenance',
            'Inference summary and per-model/task reporting',
            'Evaluation evidence from the release’s sealed outer-fold predictions',
            'Resumable tiled inference checkpoints and progress records for large surveys',
        ],
        settings: ['Default mode: survey_volume', 'Default deep model: cnn3d_multitask', 'Tile-based, resumable processing'],
        icon: BrainCircuit,
    },
    {
        id: 'volume-products', letter: 'C', title: 'Volume Products',
        short: 'Audit and package prediction volumes',
        description: 'Turn inference outputs into validated, spatially described volume products for downstream interpretation and extraction.',
        script: 'process_C_volume_products.py', config: 'C_volume_products.yaml',
        inputs: [
            'Source inference processing-job ID',
            'Canonical prediction arrays and geometry from the selected inference job',
            'Source Foundation and model-release provenance used for independent audit',
        ],
        outputs: [
            'Stage 10 volume summary, scientific audit, dashboard and seismic-section index',
            'Volume-product manifest with source fingerprints, coverage and integrity checks',
            'Stage 11 canonical NPY handoff and native-volume export manifest',
            'Prediction freeze/reproducibility records and validation summary',
            'Optional derived SEG-Y exports with per-product manifests and reopen QC',
        ],
        settings: ['generate_sections: true', 'NPY remains canonical', 'SEG-Y is an interoperability export'],
        icon: Layers3,
    },
    {
        id: 'geobody', letter: 'D', title: 'Geobody',
        short: 'Extract and inspect connected bodies',
        description: 'Extract connected candidate geobodies from a selected Stage 11 prediction product, then inspect their geometry and exports.',
        script: 'process_D_geobody.py', config: 'D_geobody.yaml',
        inputs: [
            'Foundation, inference and source volume-products job IDs',
            'Stage 11 canonical NPY product (default: prediction_lithology)',
            'Volume geometry and validity mask from the native-volume manifest',
        ],
        outputs: [
            'Geobody preflight and summary manifests, component table and geobody catalogue',
            '3D overview, catalogue and per-body slice figures',
            'PLY meshes where configured size limits permit',
            'SEG-Y geobody mask/ID exports when reference geometry is available',
            'Run-length spatial index and catalogue reconciliation; optional per-voxel XYZ export',
        ],
        settings: ['Hysteresis thresholds: 0.35 / 0.65', '6-connectivity', 'Exploratory, uncalibrated candidate interpretation'],
        icon: Box,
    },
    {
        id: 'well-geobody', letter: 'E', title: 'Well Geobody Analysis',
        short: 'Intersect wells with extracted bodies',
        description: 'Sample registered well trajectories against Stage 12 geobody geometry and characterise intersected bodies with depth and prediction evidence.',
        script: 'process_E_well_geobody_analysis.py', config: 'E_well_geobody_analysis.yaml',
        inputs: [
            'Foundation Stage 01 trajectory/T–Z samples and selected well IDs',
            'Stage 12 geobody labels, catalogue and run-length spatial index',
            'Stage 11 frozen prediction_phit.npy and prediction_swt.npy when available',
            'Source inference and volume jobs for provenance cross-checks',
        ],
        outputs: [
            'Well path samples, path intervals and well–geobody intersection tables',
            'Selected well/geobody analysis and selected-body column tables',
            'Depth-model comparison and specialist PHIT/SWT summary',
            'Per-well intersection counts, depth characterisation and analysis manifest',
        ],
        settings: ['Configured well inventory', 'Trajectory-sample intersections', 'PHIT/SWT remain model-prediction screening'],
        icon: Activity,
    },
    {
        id: 'prospect', letter: 'F', title: 'Prospect',
        short: 'Define and screen well-anchored prospects',
        description: 'Synthesize the upstream volume, geobody and well evidence into prospect geometry, volumetric screening and well-log characterisation.',
        script: 'process_F_prospect.py', config: 'F_prospect.yaml',
        inputs: [
            'Foundation trajectories, T–Z/depth models and interpreted well logs',
            'Stage 11 volume geometry and Stage 12 geobody catalogue/spatial index',
            'Inference and upstream job IDs for source provenance',
            'Configured well inventory and anchor-selection policy',
        ],
        outputs: [
            'Prospect well-intersection audit and anchor intervals',
            'Prospect definitions and well-property tables',
            'Cutoff sensitivity, GRV-by-depth-model and deterministic volumetric tables',
            'Uncertainty screening, voxel counts by sample and optional point-coordinate exports',
            'Prospect manifest with source fingerprints, QC gates and limitations',
        ],
        settings: ['Automatic configured-well anchor selection', 'GRV requires confirmed XY coordinate units', 'HCPV is conditional scenario screening only'],
        icon: Target,
    },
];


type ConfigNodeProps = {
    name: string;
    value: ProcessingConfigValue;
    onChange: (value: ProcessingConfigValue) => void;
    depth?: number;
    stageId: StageId;
    modelReleases: ModelReleaseListItem[];
    processingJobs: ProcessingJobListItem[];
};

const upstreamStageById: Partial<Record<StageId, StageId>> = {
    'volume-products': 'inference', geobody: 'volume-products', 'well-geobody': 'geobody', prospect: 'well-geobody',
};
const scriptByStage: Record<StageId, string> = {
    foundations: 'process_A_foundations.py', inference: 'process_B_inference.py', 'volume-products': 'process_C_volume_products.py',
    geobody: 'process_D_geobody.py', 'well-geobody': 'process_E_well_geobody_analysis.py', prospect: 'process_F_prospect.py',
};
function isCompletedJob(job: ProcessingJobListItem): boolean {
    const status = String(job.status ?? '').trim().toLowerCase().replace(/[\s_-]+/g, '');
    return ['completed', 'complete', 'succeeded', 'success', 'successful', 'done'].includes(status);
}
function jobMatchesStage(job: ProcessingJobListItem, stageId: StageId): boolean {
    // `operation` is stored as the worker stem (for example, process_A_foundations),
    // while older job metadata may omit `python_script`. Check every worker identity
    // field rather than allowing the runner's shell_script to mask the worker name.
    const legacyNames: Partial<Record<StageId, string[]>> = { foundations: ['process_geogg_foundations'], 'volume-products': ['process_geogg_volume_products'], 'well-geobody': ['process_geogg_well_geobody_analysis'], prospect: ['process_geogg_prospect'] };
    const expectedNames = [scriptByStage[stageId].replace(/\.py$/i, '').toLowerCase(), ...(legacyNames[stageId] ?? [])];
    const identities = [job.python_script, job.operation, job.python_script_path]
        .filter((item): item is string => typeof item === 'string' && item.trim().length > 0);
    return identities.some(item => {
        const normalized = item.trim().replace(/\\/g, '/').split('/').pop()?.replace(/\.py$/i, '').toLowerCase() ?? '';
        return expectedNames.includes(normalized);
    });
}
function stageForJobField(name: string, stageId: StageId): StageId | null {
    const key = name.toLowerCase();
    const aliases: Record<string, StageId> = {
        foundation: 'foundations', foundations: 'foundations', inference: 'inference',
        volume: 'volume-products', volume_product: 'volume-products', volume_products: 'volume-products', geobody: 'geobody',
        well_geobody: 'well-geobody', prospect: 'prospect',
    };
    const match = key.match(/(?:source_)?(foundation|foundations|inference|volume|volume_products?|geobody|well_geobody|prospect)_job_id$/);
    if (match) return aliases[match[1]] ?? null;
    if (key === 'source_processing_job_id' || key === 'processing_job_id') return upstreamStageById[stageId] ?? null;
    return null;
}
function ConfigNode({ name, value, onChange, depth = 0, stageId, modelReleases, processingJobs }: ConfigNodeProps) {
    const [newKey, setNewKey] = useState('');
    const [nullType, setNullType] = useState<'string' | 'number' | 'boolean'>('string');
    const [nullDraft, setNullDraft] = useState('');
    const [choiceOpen, setChoiceOpen] = useState(false);
    const locked = name === 'output_root' || name === 'project_root';
    const updateArray = (index: number, next: ProcessingConfigValue) => {
        if (!Array.isArray(value)) return;
        const copy = [...value];
        copy[index] = next;
        onChange(copy);
    };
    if (Array.isArray(value)) {
        const template = value[0];
        const addValue: ProcessingConfigValue = template === undefined ? '' :
            typeof template === 'number' ? 0 : typeof template === 'boolean' ? false :
            Array.isArray(template) ? [] : template !== null && typeof template === 'object' ? {} : '';
        return <details className="processing-config-node processing-config-array" open={depth === 0}>
            <summary><span>{name}</span><small>{value.length} items</small></summary>
            <div className="processing-config-node-body">
                {value.map((item, index) => <div className="processing-config-array-item" key={index}>
                    <ConfigNode name={`Item ${index + 1}`} value={item} onChange={next => updateArray(index, next)} depth={depth + 1} stageId={stageId} modelReleases={modelReleases} processingJobs={processingJobs}/>
                    <button type="button" className="processing-config-remove" onClick={() => onChange(value.filter((_, itemIndex) => itemIndex !== index))} aria-label={`Remove ${name} item ${index + 1}`}>Remove</button>
                </div>)}
                <button type="button" className="processing-config-add" onClick={() => onChange([...value, addValue])}>+ Add item</button>
            </div>
        </details>;
    }
    if (value !== null && typeof value === 'object') {
        const entries = Object.entries(value);
        return <details className="processing-config-node processing-config-object" open={depth === 0}>
            <summary><span>{name}</span><small>{entries.length} fields</small></summary>
            <div className="processing-config-node-body">
                {entries.map(([key, child]) => <ConfigNode key={key} name={key} value={child} depth={depth + 1}
                    onChange={next => onChange({ ...value, [key]: next })} stageId={stageId} modelReleases={modelReleases} processingJobs={processingJobs}/>)}
                <div className="processing-config-add-field">
                    <input aria-label={`New field in ${name}`} value={newKey} onChange={event => setNewKey(event.target.value)} placeholder="New field name"/>
                    <button type="button" className="processing-config-add" disabled={!newKey.trim() || Object.prototype.hasOwnProperty.call(value, newKey.trim())}
                        onClick={() => { const key = newKey.trim(); if (key) { onChange({ ...value, [key]: null }); setNewKey(''); } }}>+ Add field</button>
                </div>
            </div>
        </details>;
    }
    return <div className="processing-config-field">
        <label>{name}{locked ? <small className="processing-config-locked"> · runtime-managed</small> : null}</label>
        {typeof value === 'boolean' ? <label className="processing-config-toggle"><input type="checkbox" checked={value} disabled={locked} onChange={event => onChange(event.target.checked)}/><span>{value ? 'Enabled' : 'Disabled'}</span></label> :
            typeof value === 'number' ? <input type="number" step="any" value={value} disabled={locked} onChange={event => onChange(event.target.value === '' ? null : Number(event.target.value))}/> :
            value === null && name.toLowerCase() !== 'model_release_id' && !stageForJobField(name, stageId) ? <div className="processing-config-null">
                <span>Not set (null)</span>
                <select aria-label={`Value type for ${name}`} value={nullType} disabled={locked} onChange={event => setNullType(event.target.value as 'string' | 'number' | 'boolean')}>
                    <option value="string">Text</option><option value="number">Number</option><option value="boolean">Boolean</option>
                </select>
                {nullType === 'boolean' ? <select aria-label={`Value for ${name}`} value={nullDraft || 'false'} disabled={locked} onChange={event => setNullDraft(event.target.value)}><option value="false">False</option><option value="true">True</option></select> :
                    <input aria-label={`New value for ${name}`} value={nullDraft} disabled={locked} type={nullType === 'number' ? 'number' : 'text'} onChange={event => setNullDraft(event.target.value)} placeholder="Enter value"/>}
                <button type="button" disabled={locked || (nullType !== 'boolean' && nullDraft === '')} onClick={() => {
                    if (nullType === 'number') {
                        const parsed = Number(nullDraft);
                        if (Number.isFinite(parsed)) onChange(parsed);
                    } else if (nullType === 'boolean') onChange(nullDraft === 'true');
                    else onChange(nullDraft);
                    setNullDraft('');
                }}>Set</button>
            </div> :
            (() => {
                const isReleaseField = name.toLowerCase() === 'model_release_id';
                const requiredStage = stageForJobField(name, stageId);
                if (isReleaseField) {
                    const available = modelReleases.filter(item => ['ready', 'complete', 'completed', 'succeeded', 'success'].includes(String(item.status ?? '').toLowerCase()));
                    const selected = available.find(item => item.model_release_id === value);
                    return <div className="processing-config-choice">
                        <button type="button" className="processing-config-choice-trigger" aria-label={`Choose ${name}`} aria-expanded={choiceOpen} disabled={locked} onClick={() => setChoiceOpen(open => !open)}>
                            <strong>{selected?.model_release_id ?? String(value)}</strong>
                            <small>{selected ? `${selected.model_family ?? selected.pipeline ?? 'Model'} · ${selected.model_count} model files · ${selected.status}` : 'Current value · unavailable'}</small>
                        </button>
                        <small className="processing-config-choice-count">{available.length} ready model release{available.length === 1 ? '' : 's'} available</small>
                        <button type="button" className="processing-config-choice-clear" disabled={locked} onClick={() => { onChange(null); setChoiceOpen(false); }} title="Set to null">×</button>
                        {choiceOpen && <ul className="processing-job-file-list processing-release-list processing-config-choice-list">{available.map(item => <li key={item.model_release_id}><button type="button" className={`processing-file-button${selected?.model_release_id === item.model_release_id ? ' active' : ''}`} onClick={() => { onChange(item.model_release_id); setChoiceOpen(false); }}><strong>{item.model_release_id}</strong><small>{item.model_family ?? item.pipeline ?? 'Model'} · {item.model_count} model files · {item.status}</small></button></li>)}</ul>}
                    </div>;
                }
                if (requiredStage) {
                    const candidates = processingJobs.filter(job => isCompletedJob(job) && jobMatchesStage(job, requiredStage));
                    const selected = candidates.find(job => String(job.id ?? job.job_id) === value);
                    return <div className="processing-config-choice">
                        <button type="button" className="processing-config-choice-trigger" aria-label={`Choose ${name}`} aria-expanded={choiceOpen} disabled={locked} onClick={() => setChoiceOpen(open => !open)}>
                            <strong>{selected ? String(selected.id ?? selected.job_id) : String(value)}</strong>
                            <small>{selected ? `${String([selected.python_script, selected.shell_script, selected.operation].find(item => typeof item === 'string' && item.trim()) ?? scriptByStage[requiredStage]).replace(/\\/g, '/').split('/').pop()} · ${selected.status}` : 'Current value · not in completed jobs'}</small>
                        </button>
                        <small className="processing-config-choice-count">{candidates.length} completed {requiredStage.replace('-', ' ')} job{candidates.length === 1 ? '' : 's'}</small>
                        <button type="button" className="processing-config-choice-clear" disabled={locked} onClick={() => { onChange(null); setChoiceOpen(false); }} title="Set to null">×</button>
                        {choiceOpen && <ul className="processing-job-file-list processing-release-list processing-config-choice-list">{candidates.map(job => { const id = String(job.id ?? job.job_id ?? ''); const scriptName = String([job.python_script, job.shell_script, job.operation].find(item => typeof item === 'string' && item.trim()) ?? scriptByStage[requiredStage]).replace(/\\/g, '/').split('/').pop(); return <li key={id}><button type="button" className={`processing-file-button${selected && String(selected.id ?? selected.job_id) === id ? ' active' : ''}`} onClick={() => { onChange(id); setChoiceOpen(false); }}><strong>{id}</strong><small>{scriptName} · {job.status}</small></button></li>; })}</ul>}
                    </div>;
                }
                return <div className="processing-config-text"><input type="text" value={String(value)} disabled={locked} onChange={event => onChange(event.target.value)}/><button type="button" disabled={locked} onClick={() => onChange(null)} title="Set to null">×</button></div>;
            })()}
    </div>;
}


function InferenceResultViewer({ jobs, backendConnected }: { jobs: ProcessingJobListItem[]; backendConnected: boolean }) {
    const candidates = jobs.filter(job => isCompletedJob(job) && (jobMatchesStage(job, 'inference') || String(job.python_script ?? '').toLowerCase().includes('inference')));
    const [jobId, setJobId] = useState('');
    const [attribute, setAttribute] = useState<'lithology' | 'vsh' | 'phit'>('lithology');
    const [plane, setPlane] = useState<'inline' | 'crossline' | 'time'>('inline');
    const [coordinate, setCoordinate] = useState(0);
    const [slice, setSlice] = useState<InferenceSliceResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const canvasRef = useRef<HTMLCanvasElement>(null);
    useEffect(() => {
        if (!candidates.length) { setJobId(''); return; }
        if (!candidates.some(job => String(job.id ?? job.job_id) === jobId)) setJobId(String(candidates[0].id ?? candidates[0].job_id));
    }, [jobs, jobId]);
    useEffect(() => {
        if (!backendConnected || !jobId) { setSlice(null); return; }
        let current = true;
        setLoading(true); setError('');
        getInferenceSlice(jobId, attribute, plane, coordinate).then(result => {
            if (!current) return;
            setSlice(result); setCoordinate(result.coordinate);
        }).catch(err => { if (current) { setSlice(null); setError(err instanceof Error ? err.message : String(err)); } })
          .finally(() => { if (current) setLoading(false); });
        return () => { current = false; };
    }, [backendConnected, jobId, attribute, plane, coordinate]);
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !slice?.values?.length) return;
        const rows = slice.values.length, cols = slice.values[0]?.length ?? 0;
        if (!cols) return;
        canvas.width = cols; canvas.height = rows;
        const ctx = canvas.getContext('2d'); if (!ctx) return;
        const image = ctx.createImageData(cols, rows);
        const validValues = slice.values.flatMap((row, y) => row.filter((_, x) => slice.valid_mask[y]?.[x]));
        const min = attribute === 'lithology' ? 0 : Math.min(...validValues);
        const max = attribute === 'lithology' ? 1 : Math.max(...validValues);
        const span = max > min ? max - min : 1;
        for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
            const i = (y * cols + x) * 4;
            if (!slice.valid_mask[y]?.[x]) { image.data[i]=35; image.data[i+1]=39; image.data[i+2]=47; image.data[i+3]=255; continue; }
            const v = Math.max(0, Math.min(1, (slice.values[y][x] - min) / span));
            image.data[i] = Math.round(35 + 210 * v);
            image.data[i+1] = Math.round(80 + 100 * (1 - Math.abs(v - .5) * 2));
            image.data[i+2] = Math.round(220 - 180 * v);
            image.data[i+3] = 255;
        }
        ctx.putImageData(image, 0, 0);
    }, [slice, attribute]);
    const coordinateOptions = slice ? (plane === 'inline' ? slice.axes.y : plane === 'crossline' ? slice.axes.x : slice.axes.y) : [];
    return <div className="inference-result-viewer">
        <div className="inference-viewer-toolbar">
            <label>Completed inference job<select value={jobId} onChange={e=>{setJobId(e.target.value);setCoordinate(0);}}>{candidates.map(job=><option key={String(job.id??job.job_id)} value={String(job.id??job.job_id)}>{String(job.id??job.job_id)}</option>)}</select></label>
            <label>Attribute<select value={attribute} onChange={e=>setAttribute(e.target.value as typeof attribute)}><option value="lithology">Lithology</option><option value="vsh">Vsh</option><option value="phit">Phit</option></select></label>
            <label>Section<select value={plane} onChange={e=>{setPlane(e.target.value as typeof plane);setCoordinate(0);}}><option value="inline">Inline</option><option value="crossline">Crossline</option><option value="time">Time slice</option></select></label>
        </div>
        {!candidates.length ? <div className="processing-stage-viewer-empty">No completed Stage B inference jobs were found.</div> :
            error ? <div className="processing-stage-viewer-empty"><strong>Prediction volume unavailable</strong><p>{error}</p><small>Check that the job contains stages/09_inference/volumes and the selected prediction NPY file.</small></div> :
            <div className="inference-viewer-content">
                <div className="inference-viewer-stats">
                    <span><small>Volume shape</small><strong>{slice?.volume_shape.join(' × ') ?? '—'}</strong></span>
                    <span><small>Predicted voxels</small><strong>{Number((slice?.summary as any)?.predicted_voxels ?? 0).toLocaleString() || '—'}</strong></span>
                    <span><small>Inference device</small><strong>{String((slice?.summary as any)?.device ?? '—')}</strong></span>
                    <span><small>Valid samples in slice</small><strong>{slice?.valid_count.toLocaleString() ?? '—'}</strong></span>
                    <span><small>Invalid / no-data</small><strong>{slice?.invalid_count.toLocaleString() ?? '—'}</strong></span>
                    <span><small>Downsample stride</small><strong>{slice?.downsample_stride ?? '—'}</strong></span>
                </div>
                <div className="inference-viewer-coordinate">
                    <label>{plane === 'inline' ? 'Inline' : plane === 'crossline' ? 'Crossline' : 'TWT (ms)'} <strong>{coordinate}</strong></label>
                    <input type="range" min={0} max={Math.max(0, (slice?.coordinate_count ?? 1)-1)}
                        value={slice ? Math.max(0, Math.min(slice.coordinate_count-1, Math.round((coordinate-slice.coordinate_min)/slice.coordinate_step))) : 0}
                        onChange={e=>setCoordinate(slice ? slice.coordinate_min + Number(e.target.value)*slice.coordinate_step : 0)}/>
                </div>
                <div className="inference-viewer-canvas-wrap">{loading && <span className="inference-viewer-loading">Loading slice…</span>}<canvas ref={canvasRef} aria-label={`${attribute} ${plane} prediction slice`}/></div>
                {slice?.geometry_flags?.length ? <div className="inference-viewer-warning"><strong>Geometry limitation</strong><p>{slice.geometry_flags.join('; ')}. Display is indexed in native IL/XL coordinates; XY positioning is not asserted.</p></div> : null}
            </div>}
    </div>;
}


function VolumeProductsResultViewer({ jobs, backendConnected }: { jobs: ProcessingJobListItem[]; backendConnected: boolean }) {
    const candidates = jobs.filter(job => isCompletedJob(job) && jobMatchesStage(job, 'volume-products'));
    const [jobId, setJobId] = useState('');
    const [artifacts, setArtifacts] = useState<VolumeProductArtifact[]>([]);
    const [selectedPath, setSelectedPath] = useState('');
    const [content, setContent] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [view, setView] = useState<'overview' | 'figures' | 'tables' | 'exports'>('overview');
    const [galleryIndex, setGalleryIndex] = useState<number | null>(null);
    const galleryTouchStartX = useRef<number | null>(null);
    useEffect(() => {
        if (!candidates.length) { setJobId(''); return; }
        if (!candidates.some(job => String(job.id ?? job.job_id) === jobId)) setJobId(String(candidates[0].id ?? candidates[0].job_id));
    }, [jobs, jobId]);
    useEffect(() => {
        if (!backendConnected || !jobId) { setArtifacts([]); return; }
        let active = true; setLoading(true); setError('');
        listVolumeProductArtifacts(jobId).then(result => { if (active) { setArtifacts(result.items); const summary = result.items.find(a => a.name === 'volume_summary.json') ?? result.items.find(a => a.name === 'geogg_volume_products_summary.json'); setSelectedPath(summary?.path ?? ''); setContent(''); } })
            .catch(err => { if (active) setError(err instanceof Error ? err.message : String(err)); })
            .finally(() => { if (active) setLoading(false); });
        return () => { active = false; };
    }, [backendConnected, jobId]);
    useEffect(() => {
        if (!jobId || !selectedPath || ['.png', '.svg', '.html', '.sgy', '.segy', '.npy'].includes(artifacts.find(a => a.path === selectedPath)?.suffix ?? '')) { setContent(''); return; }
        let active = true; setLoading(true);
        getVolumeProductArtifact(jobId, selectedPath).then(result => { if (active) setContent(result.content); })
            .catch(err => { if (active) setError(err instanceof Error ? err.message : String(err)); })
            .finally(() => { if (active) setLoading(false); });
        return () => { active = false; };
    }, [jobId, selectedPath, artifacts]);
    const figures = artifacts.filter(a => ['.png', '.svg'].includes(a.suffix));
    useEffect(() => {
        if (galleryIndex === null) return;
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') setGalleryIndex(null);
            if (event.key === 'ArrowRight') setGalleryIndex(index => index === null ? null : (index + 1) % figures.length);
            if (event.key === 'ArrowLeft') setGalleryIndex(index => index === null ? null : (index - 1 + figures.length) % figures.length);
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [galleryIndex, figures.length]);
    const tables = artifacts.filter(a => a.suffix === '.csv');
    const exports = artifacts.filter(a => ['.segy', '.sgy', '.npy', '.json', '.html'].includes(a.suffix) && a.category === 'export');
    const summaries = artifacts.filter(a => a.suffix === '.json' && a.category === 'summary');
    const activeArtifact = artifacts.find(a => a.path === selectedPath);
    const summaryArtifact = summaries.find(a => a.name === 'volume_summary.json') ?? summaries.find(a => a.name === 'geogg_volume_products_summary.json');
    const selectedJson = selectedPath && activeArtifact?.suffix === '.json' ? content : '';
    let parsed: any = null; try { parsed = selectedJson ? JSON.parse(selectedJson) : null; } catch { parsed = null; }
    const rows = activeArtifact?.suffix === '.csv' && content ? content.trim().split(/\r?\n/).slice(0, 80).map(line => line.split(',')) : [];
    return <div className="volume-products-viewer">
        <div className="inference-viewer-toolbar">
            <label>Completed Volume Products job<select value={jobId} onChange={e => { setJobId(e.target.value); setSelectedPath(''); }}><option value="">Select job</option>{candidates.map(job => <option key={String(job.id ?? job.job_id)} value={String(job.id ?? job.job_id)}>{String(job.id ?? job.job_id)}</option>)}</select></label>
            <span className="processing-config-choice-count">{artifacts.length} discovered artefacts</span>
        </div>
        {!candidates.length ? <div className="processing-stage-viewer-empty">No completed Stage C jobs found. Both process_C_volume_products.py and legacy process_geogg_volume_products.py are recognised.</div> : error ? <div className="processing-stage-viewer-empty"><strong>Could not load Volume Products artefacts</strong><p>{error}</p></div> : <>
            <nav className="volume-products-tabs" aria-label="Volume Products result categories">{(['overview','figures','tables','exports'] as const).map(key => <button type="button" className={(key === 'figures' ? galleryIndex !== null : view === key) ? 'active' : ''} onClick={() => { if (key === 'figures') { if (figures.length) setGalleryIndex(0); return; } setView(key); setSelectedPath(''); }}>{key === 'overview' ? 'Overview' : key === 'figures' ? `View figures (${figures.length})` : key === 'tables' ? `Tables (${tables.length})` : `Exports & QC (${exports.length})`}</button>)}</nav>
            {loading && <div className="inference-viewer-loading">Loading artefacts…</div>}
            {view === 'overview' && <div className="volume-products-overview">
                <div className="volume-products-artifact-grid">{[summaryArtifact, ...summaries.filter(a => a !== summaryArtifact).slice(0, 5)].filter(Boolean).map(a => <button type="button" key={a!.path} className="processing-file-button" onClick={() => setSelectedPath(a!.path)}><strong>{a!.name}</strong><small>{(a!.size_bytes / 1024).toFixed(1)} KB · summary metadata</small></button>)}</div>
                {parsed && <div className="inference-viewer-stats"><span><small>Source inference job</small><strong>{parsed.source_inference_job_id ?? '—'}</strong></span><span><small>Volume shape</small><strong>{parsed.shape?.join(' × ') ?? parsed.geometry?.shape?.join(' × ') ?? '—'}</strong></span><span><small>Valid voxels</small><strong>{Number(parsed.valid_voxels ?? 0).toLocaleString()}</strong></span><span><small>Valid coverage</small><strong>{parsed.valid_voxel_fraction != null ? `${(parsed.valid_voxel_fraction * 100).toFixed(2)}%` : '—'}</strong></span></div>}
                {parsed?.attributes && <div className="volume-products-attribute-grid">{Object.entries(parsed.attributes).map(([key, value]: [string, any]) => <article><strong>{key.toUpperCase()}</strong><span>Mean {Number(value.mean).toFixed(4)}</span><span>Range {Number(value.min).toFixed(4)} – {Number(value.max).toFixed(4)}</span><span>Std {Number(value.std).toFixed(4)}</span></article>)}</div>}
                {parsed?.geometry?.spatial_reference?.xy_mapping_supported === false && <div className="inference-viewer-warning"><strong>Spatial reference limitation</strong><p>XY mapping is not supported by the reported geometry QC. Interpret using native IL/XL coordinates; do not treat the affine XY approximation as authoritative.</p></div>}
                {!parsed && <p className="processing-stage-viewer-empty">Select the volume summary JSON to inspect product statistics and provenance.</p>}
                {summaryArtifact && <button type="button" className="btn btn-secondary" onClick={() => setSelectedPath(summaryArtifact.path)}>Open volume summary</button>}
            </div>}
            {view === 'tables' && <div className="volume-products-table-list">{tables.map(a => <button type="button" className={`processing-file-button${selectedPath === a.path ? ' active' : ''}`} key={a.path} onClick={() => setSelectedPath(a.path)}><strong>{a.name}</strong><small>{a.path} · {(a.size_bytes / 1024).toFixed(1)} KB</small></button>)}</div>}
            {view === 'exports' && <div className="volume-products-table-list">{[...exports, ...artifacts.filter(a => a.category === 'export' && a.suffix === '.csv'), ...artifacts.filter(a => a.name === 'dashboard.html')].map(a => <div className="processing-file-button" key={a.path}><strong>{a.name}</strong><small>{a.path} · {(a.size_bytes / 1024 / 1024).toFixed(2)} MB</small><a href={volumeProductArtifactUrl(jobId, a.path)} target="_blank" rel="noreferrer">Open / download</a></div>)}</div>}
            {selectedPath && activeArtifact && <section className="volume-products-selected-artifact"><div><strong>{activeArtifact.name}</strong><button type="button" onClick={() => setSelectedPath('')}>Close</button></div>{activeArtifact.suffix === '.png' || activeArtifact.suffix === '.svg' ? <img src={volumeProductArtifactUrl(jobId, activeArtifact.path)} alt={activeArtifact.name}/> : activeArtifact.suffix === '.csv' ? <div className="volume-products-csv"><table><tbody>{rows.map((row, i) => <tr>{row.map(cell => i === 0 ? <th>{cell}</th> : <td>{cell}</td>)}</tr>)}</tbody></table><small>Preview limited to first 80 rows.</small></div> : <pre>{content.slice(0, 30000)}</pre>}</section>}
        </>}
        {galleryIndex !== null && figures[galleryIndex] && <div className="geobody-lightbox" role="presentation" onClick={() => setGalleryIndex(null)}>
            <section className="geobody-lightbox-dialog" role="dialog" aria-modal="true" aria-label="Volume Products figure gallery" onClick={event => event.stopPropagation()}
                onTouchStart={event => { galleryTouchStartX.current = event.changedTouches[0]?.clientX ?? null; }}
                onTouchEnd={event => {
                    const start = galleryTouchStartX.current;
                    const end = event.changedTouches[0]?.clientX;
                    galleryTouchStartX.current = null;
                    if (start === null || end === undefined || Math.abs(end - start) < 45 || figures.length < 2) return;
                    setGalleryIndex(index => index === null ? null : (index + (end < start ? 1 : -1) + figures.length) % figures.length);
                }}>
                <header className="geobody-lightbox-head"><div><span className="eyebrow">STAGE C · VOLUME PRODUCTS FIGURES</span><strong>{figures[galleryIndex].name}</strong></div><button type="button" className="geobody-lightbox-close" onClick={() => setGalleryIndex(null)} aria-label="Close figure viewer">×</button></header>
                <div className="geobody-lightbox-image-wrap">
                    {figures.length > 1 && <button type="button" className="geobody-lightbox-nav previous" onClick={() => setGalleryIndex((galleryIndex - 1 + figures.length) % figures.length)} aria-label="Previous figure">‹</button>}
                    <img src={volumeProductArtifactUrl(jobId, figures[galleryIndex].path)} alt={figures[galleryIndex].name}/>
                    {figures.length > 1 && <button type="button" className="geobody-lightbox-nav next" onClick={() => setGalleryIndex((galleryIndex + 1) % figures.length)} aria-label="Next figure">›</button>}
                </div>
                <footer className="geobody-lightbox-foot"><button type="button" onClick={() => setGalleryIndex((galleryIndex - 1 + figures.length) % figures.length)} disabled={figures.length < 2}>Previous</button><span>{galleryIndex + 1} / {figures.length}</span><button type="button" onClick={() => setGalleryIndex((galleryIndex + 1) % figures.length)} disabled={figures.length < 2}>Next</button></footer>
            </section>
        </div>}
    </div>;
}

function StageScientificResultViewer({ stage, jobs, backendConnected }: { stage: 'well-geobody' | 'prospect'; jobs: ProcessingJobListItem[]; backendConnected: boolean }) {
    const candidates = jobs.filter(job => isCompletedJob(job) && jobMatchesStage(job, stage));
    const [jobId, setJobId] = useState('');
    const [artifacts, setArtifacts] = useState<StageResultArtifact[]>([]);
    const [selectedPath, setSelectedPath] = useState('');
    const [content, setContent] = useState('');
    const [summary, setSummary] = useState<any>(null);
    const [manifest, setManifest] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [tab, setTab] = useState<'overview' | 'figures' | 'tables' | 'exports' | 'voxels'>('overview');
    const [voxelProspect, setVoxelProspect] = useState('');
    const [voxelPreview, setVoxelPreview] = useState<ProspectVoxelPreview | null>(null);
    const [voxelLoading, setVoxelLoading] = useState(false);
    const [voxelError, setVoxelError] = useState('');
    const [galleryIndex, setGalleryIndex] = useState<number | null>(null);
    const touchX = useRef<number | null>(null);
    const legacy = stage === 'well-geobody' ? 'process_geogg_well_geobody_analysis.py' : 'process_geogg_prospect.py';
    useEffect(() => {
        if (!candidates.length) { setJobId(''); return; }
        if (!candidates.some(job => String(job.id ?? job.job_id) === jobId)) setJobId(String(candidates[0].id ?? candidates[0].job_id));
    }, [jobs, jobId]);
    useEffect(() => {
        if (!backendConnected || !jobId) { setArtifacts([]); setSummary(null); setManifest(null); return; }
        let active = true; setLoading(true); setError(''); setSelectedPath('');
        listStageResultArtifacts(jobId, stage).then(async result => {
            if (!active) return;
            setArtifacts(result.items);
            const summaryName = stage === 'well-geobody' ? 'geogg_well_geobody_analysis_summary.json' : 'geogg_prospect_summary.json';
            const manifestName = stage === 'well-geobody' ? 'well_geobody_analysis_manifest.json' : 'prospect_manifest.json';
            const summaryFile = result.items.find(a => a.name === summaryName);
            const manifestFile = result.items.find(a => a.name === manifestName);
            const [summaryResult, manifestResult] = await Promise.all([
                summaryFile ? getStageResultArtifact(jobId, stage, summaryFile.path).catch(() => null) : Promise.resolve(null),
                manifestFile ? getStageResultArtifact(jobId, stage, manifestFile.path).catch(() => null) : Promise.resolve(null),
            ]);
            if (!active) return;
            try { setSummary(summaryResult ? JSON.parse(summaryResult.content) : null); } catch { setSummary(null); }
            try { setManifest(manifestResult ? JSON.parse(manifestResult.content) : null); } catch { setManifest(null); }
        }).catch(err => { if (active) setError(err instanceof Error ? err.message : String(err)); })
          .finally(() => { if (active) setLoading(false); });
        return () => { active = false; };
    }, [backendConnected, jobId, stage]);
    useEffect(() => {
        if (!jobId || !selectedPath) { setContent(''); return; }
        const artifact = artifacts.find(a => a.path === selectedPath);
        if (!artifact || ['.png','.jpg','.jpeg','.svg','.pdf','.npy','.npz','.ply'].includes(artifact.suffix)) { setContent(''); return; }
        let active = true; setLoading(true);
        getStageResultArtifact(jobId, stage, selectedPath).then(result => { if (active) setContent(result.content); })
            .catch(err => { if (active) setError(err instanceof Error ? err.message : String(err)); })
            .finally(() => { if (active) setLoading(false); });
        return () => { active = false; };
    }, [jobId, stage, selectedPath, artifacts]);
    const figures = artifacts.filter(a => a.category === 'figure');
    const tables = artifacts.filter(a => a.category === 'table');
    const exports = artifacts.filter(a => a.category === 'export');
    const voxelProspects = artifacts.map(a => a.path.match(/(?:^|\/)prospect_voxels\/(P\d{3})_ijk\.npy$/)?.[1]).filter((v): v is string => !!v).sort();
    const activeArtifact = artifacts.find(a => a.path === selectedPath);
    useEffect(() => { if (stage !== 'prospect') return; if (!voxelProspects.length) { setVoxelProspect(''); setVoxelPreview(null); return; } if (!voxelProspects.includes(voxelProspect)) setVoxelProspect(voxelProspects[0]); }, [stage, voxelProspects.join('|'), voxelProspect]);
    useEffect(() => { if (stage !== 'prospect' || !jobId || !voxelProspect) { setVoxelPreview(null); return; } let active=true; setVoxelLoading(true); setVoxelError(''); getProspectVoxelPreview(jobId,voxelProspect).then(v=>{if(active)setVoxelPreview(v);}).catch(e=>{if(active){setVoxelPreview(null);setVoxelError(e instanceof Error?e.message:String(e));}}).finally(()=>{if(active)setVoxelLoading(false);}); return ()=>{active=false;}; }, [stage,jobId,voxelProspect]);
    useEffect(() => {
        if (galleryIndex === null) return;
        const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') setGalleryIndex(null); if (event.key === 'ArrowRight') setGalleryIndex(i => i === null ? null : (i+1)%figures.length); if (event.key === 'ArrowLeft') setGalleryIndex(i => i === null ? null : (i-1+figures.length)%figures.length); };
        window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
    }, [galleryIndex, figures.length]);
    const count = summary?.counts ?? {};
    const perWell = Array.isArray(summary?.per_well) ? summary.per_well : [];
    const prospects = summary?.point_exports && typeof summary.point_exports === 'object' ? Object.entries(summary.point_exports) as [string, any][] : [];
    const referencedProducts = Array.isArray(manifest?.products) ? manifest.products : Array.isArray(summary?.products) ? summary.products : [];
    const missingLarge = referencedProducts.filter((item: any) => item?.path && !artifacts.some(a => a.path === item.path));
    const csvRows = activeArtifact?.suffix === '.csv' && content ? content.split(/\r?\n/).filter(Boolean).slice(0,81).map(line => line.split(',')) : [];
    const fmt = (v: any) => typeof v === 'number' ? v.toLocaleString(undefined,{maximumFractionDigits:2}) : v == null || v === '' ? '—' : String(v);
    return <div className="volume-products-viewer stage-scientific-viewer">
        <div className="inference-viewer-toolbar"><label>Completed {stage === 'well-geobody' ? 'Well Geobody Analysis' : 'Prospect'} job<select value={jobId} onChange={e=>setJobId(e.target.value)}>{candidates.map(job=><option key={String(job.id??job.job_id)} value={String(job.id??job.job_id)}>{String(job.id??job.job_id)}</option>)}</select></label><span className="processing-config-choice-count">{artifacts.length} available artefacts</span></div>
        {!candidates.length ? <div className="processing-stage-viewer-empty">No completed {stage === 'well-geobody' ? 'Stage E' : 'Stage F'} jobs found. Legacy worker {legacy} is also recognised.</div> : error ? <div className="processing-stage-viewer-empty"><strong>Could not load stage results</strong><p>{error}</p></div> : <>
            <nav className="volume-products-tabs" aria-label="Stage result categories">{(['overview','figures','tables','exports',...(stage==='prospect'?['voxels' as const]:[])] as const).map(key=><button type="button" className={(key==='figures' ? galleryIndex!==null : tab===key)?'active':''} onClick={()=>{if(key==='figures'){if(figures.length)setGalleryIndex(0);return;}setTab(key);setSelectedPath('');}}>{key==='overview'?'Overview':key==='figures'?`View figures (${figures.length})`:key==='tables'?`Tables (${tables.length})`:key==='voxels'?`3D Voxel (${voxelProspects.length})`:`Exports & QC (${exports.length})`}</button>)}</nav>
            {loading && <div className="inference-viewer-loading">Loading stage results…</div>}
            {!loading && !error && !summary && <div className="inference-viewer-warning"><strong>Prospect summary was not loaded</strong><p>The selected job is marked completed, but the backend did not return its Stage F summary JSON. Check that backend v166 is deployed and that the job output directory is accessible to the backend.</p></div>}
            {tab==='overview' && <div className="volume-products-overview">
                <div className="inference-viewer-stats">{stage==='well-geobody' ? <>
                    <span><small>Wells analysed</small><strong>{fmt(count.wells_analysed)}</strong></span><span><small>Wells with intersections</small><strong>{fmt(count.wells_with_intersections)}</strong></span><span><small>Wells without intersections</small><strong>{fmt(count.wells_without_intersections)}</strong></span><span><small>Intersected geobodies</small><strong>{fmt(count.distinct_intersected_geobodies)}</strong></span><span><small>Trajectory samples</small><strong>{fmt(count.trajectory_samples)}</strong></span><span><small>Path intervals</small><strong>{fmt(count.path_intervals)}</strong></span>
                </> : <><span><small>Prospects</small><strong>{fmt(count.prospects)}</strong></span><span><small>Anchor intervals</small><strong>{fmt(count.anchor_intervals)}</strong></span><span><small>Wells with log support</small><strong>{fmt(count.wells_with_log_support)}</strong></span><span><small>Anchor wells</small><strong>{fmt(summary?.anchor_well_ids?.length)}</strong></span><span><small>XY units confirmed</small><strong>{summary?.qc?.xy_coordinate_units_confirmed ? 'Yes' : 'No'}</strong></span><span><small>HCPV status</small><strong>{fmt(summary?.qc?.hcpv_status)}</strong></span></>}</div>
                {summary && <div className="inference-viewer-warning"><strong>{String(summary.interpretation_status ?? summary.status ?? 'Result summary')}</strong><p>{stage==='well-geobody' ? 'Trajectory-sample intersections are not continuous wellbore interpolation. Selected largest bodies are ranked by catalogue voxel count, not geological quality.' : 'Prospects are exploratory geometric candidates. Volumetric and HCPV interpretations remain subject to the qualification gates and stated limitations.'}</p></div>}
                {stage==='well-geobody' && <><h4>Per-well intersection summary</h4><div className="volume-products-csv"><table><thead><tr><th>Well</th><th>Status</th><th>Path samples</th><th>Intersecting samples</th><th>Geobodies</th><th>Selected body</th></tr></thead><tbody>{perWell.map((w:any)=><tr key={w.well}><td>{w.well}</td><td>{w.status}</td><td>{fmt(w.counts?.trajectory_samples)}</td><td>{fmt(w.counts?.samples_intersecting_geobodies)}</td><td>{fmt(w.counts?.distinct_intersected_geobodies)}</td><td>{fmt(w.selected_geobody_id)}</td></tr>)}</tbody></table></div>
                    {summary?.specialist_phit_swt_analysis && <div className="inference-viewer-warning"><strong>Specialist PHIT / SWT: {summary.specialist_phit_swt_analysis.status}</strong><p>{summary.specialist_phit_swt_analysis.interpretation ?? 'Property statistics are only shown when the frozen source arrays and coverage gate are available.'}</p></div>}
                    <div className="volume-products-artifact-grid">{artifacts.filter(a=>a.category==='summary').map(a=><button type="button" className="processing-file-button" key={a.path} onClick={()=>setSelectedPath(a.path)}><strong>{a.name}</strong><small>{(a.size_bytes/1024).toFixed(1)} KB · summary / manifest</small></button>)}</div></>}
                {stage==='prospect' && <><h4>Prospect catalogue</h4><div className="volume-products-csv"><table><thead><tr><th>Prospect</th><th>Export status</th><th>Point rows</th></tr></thead><tbody>{prospects.map(([id,info])=><tr key={id}><td>{id}</td><td>{info?.status ?? '—'}</td><td>{fmt(info?.rows)}</td></tr>)}</tbody></table></div>
                    <div className="volume-products-artifact-grid">{artifacts.filter(a=>a.category==='summary').map(a=><button type="button" className="processing-file-button" key={a.path} onClick={()=>setSelectedPath(a.path)}><strong>{a.name}</strong><small>{(a.size_bytes/1024).toFixed(1)} KB · summary / manifest</small></button>)}</div>
                    <div className="inference-viewer-warning"><strong>Spatial export availability</strong><p>{exports.length} large spatial/export files are present. Missing manifest-listed files are not treated as job failure; this example intentionally omits large voxel and XYZ point exports.</p></div>
                </>}
                {summary?.limitations?.length ? <details className="processing-config-collapsible"><summary><span><strong>Scientific limitations</strong><small>{summary.limitations.length} documented limitations</small></span></summary><ul>{summary.limitations.map((item:string)=><li>{item}</li>)}</ul></details> : null}
                {missingLarge.length>0 && <details className="processing-config-collapsible"><summary><span><strong>Manifest products not present in this archive</strong><small>{missingLarge.length} referenced products absent</small></span></summary><ul>{missingLarge.map((item:any)=><li><code>{item.path}</code> · {item.format ?? 'file'} · {fmt(item.bytes)} bytes</li>)}</ul></details>}
            </div>}
            {tab==='voxels' && stage==='prospect' && <section className="prospect-voxel-section"><div className="prospect-voxel-select"><label>Prospect voxel set<select value={voxelProspect} onChange={e=>setVoxelProspect(e.target.value)}>{voxelProspects.map(id=><option key={id} value={id}>{id}</option>)}</select></label><span>{voxelProspects.length} voxel arrays discovered</span></div>{voxelLoading&&<div className="inference-viewer-loading">Loading voxel coordinates…</div>}{voxelError&&<div className="inference-viewer-warning">{voxelError}</div>}{!voxelProspects.length&&<div className="inference-viewer-warning">No voxel arrays are present in this job. The viewer does not synthesize missing geometry.</div>}{voxelPreview&&<Suspense fallback={<div className="inference-viewer-loading">Loading 3D renderer…</div>}><ProspectVoxelCanvas preview={voxelPreview}/></Suspense>}</section>}
            {tab==='tables' && <div className="volume-products-table-list">{tables.map(a=><button type="button" className={`processing-file-button${selectedPath===a.path?' active':''}`} key={a.path} onClick={()=>setSelectedPath(a.path)}><strong>{a.name}</strong><small>{a.path} · {(a.size_bytes/1024).toFixed(1)} KB</small></button>)}</div>}
            {tab==='exports' && <div className="volume-products-table-list">{[...exports,...artifacts.filter(a=>a.category==='summary' && a.suffix==='.json')].map(a=><div className="processing-file-button" key={a.path}><strong>{a.name}</strong><small>{a.path} · {(a.size_bytes/1024/1024).toFixed(2)} MB</small><a href={stageResultArtifactUrl(jobId,stage,a.path)} target="_blank" rel="noreferrer">Open / download</a></div>)}</div>}
            {selectedPath && activeArtifact && <section className="volume-products-selected-artifact"><div><strong>{activeArtifact.name}</strong><button type="button" onClick={()=>setSelectedPath('')}>Close</button></div>{['.png','.jpg','.jpeg','.svg'].includes(activeArtifact.suffix) ? <img src={stageResultArtifactUrl(jobId,stage,activeArtifact.path)} alt={activeArtifact.name}/> : activeArtifact.suffix==='.csv' ? <div className="volume-products-csv"><table><tbody>{csvRows.map((row,i)=><tr>{row.map(cell=>i===0?<th>{cell}</th>:<td>{cell}</td>)}</tr>)}</tbody></table><small>Preview limited to 80 rows. Download the original for complete data.</small></div> : <pre>{content.slice(0,30000)}</pre>}</section>}
        </>}
        {galleryIndex!==null && figures[galleryIndex] && <div className="geobody-lightbox" role="presentation" onClick={()=>setGalleryIndex(null)}><section className="geobody-lightbox-dialog" role="dialog" aria-modal="true" aria-label="Stage figures" onClick={e=>e.stopPropagation()} onTouchStart={e=>{touchX.current=e.changedTouches[0]?.clientX??null;}} onTouchEnd={e=>{const start=touchX.current,end=e.changedTouches[0]?.clientX;touchX.current=null;if(start===null||end===undefined||Math.abs(end-start)<45||figures.length<2)return;setGalleryIndex(i=>i===null?null:(i+(end<start?1:-1)+figures.length)%figures.length);}}><header className="geobody-lightbox-head"><div><span className="eyebrow">{stage==='well-geobody'?'STAGE E · WELL GEOBODY ANALYSIS':'STAGE F · PROSPECT'} FIGURES</span><strong>{figures[galleryIndex].name}</strong></div><button type="button" className="geobody-lightbox-close" onClick={()=>setGalleryIndex(null)}>×</button></header><div className="geobody-lightbox-image-wrap">{figures.length>1&&<button type="button" className="geobody-lightbox-nav previous" onClick={()=>setGalleryIndex((galleryIndex-1+figures.length)%figures.length)}>‹</button>}<img src={stageResultArtifactUrl(jobId,stage,figures[galleryIndex].path)} alt={figures[galleryIndex].name}/>{figures.length>1&&<button type="button" className="geobody-lightbox-nav next" onClick={()=>setGalleryIndex((galleryIndex+1)%figures.length)}>›</button>}</div><footer className="geobody-lightbox-foot"><button type="button" disabled={figures.length<2} onClick={()=>setGalleryIndex((galleryIndex-1+figures.length)%figures.length)}>Previous</button><span>{galleryIndex+1} / {figures.length}</span><button type="button" disabled={figures.length<2} onClick={()=>setGalleryIndex((galleryIndex+1)%figures.length)}>Next</button></footer></section></div>}
    </div>;
}

function FoundationsResultViewer({ jobs, backendConnected }: { jobs: ProcessingJobListItem[]; backendConnected: boolean }) {
    const candidates = jobs.filter(job => isCompletedJob(job) && jobMatchesStage(job, 'foundations'));
    const [jobId, setJobId] = useState('');
    const [items, setItems] = useState<StageResultArtifact[]>([]);
    const [summary, setSummary] = useState<any>(null);
    const [activeStage, setActiveStage] = useState('00_inventory');
    const [selectedPath, setSelectedPath] = useState('');
    const [content, setContent] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [galleryIndex, setGalleryIndex] = useState<number | null>(null);
    const stages = [
        ['00_inventory','Data inventory & QC','Dataset registration, seismic geometry, well/log availability'],
        ['01_well_ties','Well ties & trajectories','Trajectory stations, T–Z models, velocity QC and synthetic ties'],
        ['02_labels','Petrophysical labels','Interpreted logs, class definitions and label evidence'],
        ['03_extraction','Seismic extraction','Aligned labels, seismic patches and extraction coverage'],
        ['04_features','Feature engineering','Stack attributes, AVO and elastic feature products'],
        ['05_training_tables','Training tables','Predictor matrices, targets, folds and well-level splits'],
        ['06_feature_selection','Feature selection','Screening, stability, collinearity and selected predictors'],
        ['07_models','Model development','Model registry, fit audit, CV scores and training evidence'],
    ] as const;
    useEffect(() => { if (!candidates.length) { setJobId(''); return; } if (!candidates.some(j=>String(j.id??j.job_id)===jobId)) setJobId(String(candidates[0].id??candidates[0].job_id)); }, [jobs, jobId]);
    useEffect(() => { if (!backendConnected || !jobId) { setItems([]); setSummary(null); return; } let live=true; setLoading(true); setError(''); listStageResultArtifacts(jobId,'foundations').then(async r=>{ if(!live)return; setItems(r.items); const f=r.items.find(x=>x.name==='geogg_foundations_summary.json'); if(f){const v=await getStageResultArtifact(jobId,'foundations',f.path); if(live)try{setSummary(JSON.parse(v.content));}catch{setSummary(null);}} else setSummary(null); }).catch(e=>{if(live)setError(e instanceof Error?e.message:String(e));}).finally(()=>{if(live)setLoading(false);}); return()=>{live=false;}; },[backendConnected,jobId]);
    useEffect(()=>{if(!jobId||!selectedPath){setContent('');return;}let live=true;getStageResultArtifact(jobId,'foundations',selectedPath).then(v=>{if(live)setContent(v.content);}).catch(e=>{if(live)setContent(String(e));});return()=>{live=false;};},[jobId,selectedPath]);
    const stageItems=(id:string)=>items.filter(a=>a.path.replace(/\\/g,'/').includes(`stages/${id}/`));
    const figures=stageItems(activeStage).filter(a=>a.category==='figure');
    const tables=stageItems(activeStage).filter(a=>a.category==='table');
    const otherOutputs=stageItems(activeStage).filter(a=>!['figure','table'].includes(a.category));
    const fixedFigures=['stages/00_inventory/figures/data_availability_matrix.png','stages/01_well_ties/figures/well_trajectories_map.png'];
    const fixedAvailable=fixedFigures.map(path=>items.find(a=>a.path.endsWith(path))).filter(Boolean) as StageResultArtifact[];
    const metric=(key:string)=>summary?.stages?.find((s:any)=>s.stage_id===activeStage)?.metrics?.[key];
    return <div className="foundation-result-viewer">
      <div className="foundation-viewer-toolbar"><div><span className="eyebrow">FOUNDATION RUN</span><strong>{candidates.length} completed run{candidates.length===1?'':'s'} available</strong></div><select value={jobId} onChange={e=>setJobId(e.target.value)}>{candidates.map(j=><option key={String(j.id??j.job_id)} value={String(j.id??j.job_id)}>{String(j.id??j.job_id).slice(0,8)} · {String(j.python_script??j.script??'Foundations')}</option>)}</select></div>
      {loading&&<div className="inference-viewer-loading">Loading Foundation stage outputs…</div>}{error&&<div className="inference-viewer-warning">{error}</div>}
      {!candidates.length&&<div className="processing-stage-viewer-empty"><strong>No completed Foundation job found.</strong><p>Completed process_A_foundations.py runs will appear here.</p></div>}
      {!!jobId&&<>
        <section className="foundation-fixed-grid"><article className="foundation-fixed-card"><header><div><span className="eyebrow">00 · INVENTORY</span><strong>Data availability matrix</strong></div><small>Fixed overview</small></header>{fixedAvailable.find(a=>a.path.includes('data_availability_matrix'))?<img src={stageResultArtifactUrl(jobId,'foundations',fixedAvailable.find(a=>a.path.includes('data_availability_matrix'))!.path)} alt="Data availability matrix"/>:<p>Matrix figure not present in this run.</p>}</article><article className="foundation-fixed-card"><header><div><span className="eyebrow">01 · WELL TIES</span><strong>Well trajectory map</strong></div><small>Fixed overview</small></header>{fixedAvailable.find(a=>a.path.includes('well_trajectories_map'))?<img src={stageResultArtifactUrl(jobId,'foundations',fixedAvailable.find(a=>a.path.includes('well_trajectories_map'))!.path)} alt="Well trajectory map"/>:<p>Trajectory map not present in this run.</p>}</article></section>
        <section className="foundation-stage-navigator"><header><span className="eyebrow">STAGE PIPELINE · 00–07</span><strong>Foundation processing sequence</strong><small>Choose a substage to inspect its figures, tables and outputs.</small></header><div className="foundation-stage-grid">{stages.map(([id,title,desc],i)=>{const files=stageItems(id);const stageSummary=summary?.stages?.find((s:any)=>s.stage_id===id);const status=stageSummary?.status??(files.length?'OUTPUTS FOUND':'NO OUTPUTS');return <button type="button" key={id} className={`foundation-stage-card ${activeStage===id?'active':''}`} onClick={()=>{setActiveStage(id);setSelectedPath('');}}><span className="foundation-stage-number">{String(i+1).padStart(2,'0')}</span><span className="foundation-stage-copy"><strong>{title}</strong><small>{desc}</small><em>{files.length} artefacts · {status}</em></span><span className="foundation-stage-chevron">›</span></button>;})}</div></section>
        <section className="foundation-stage-detail"><header><div><span className="eyebrow">SELECTED SUBSTAGE</span><strong>{stages.find(s=>s[0]===activeStage)?.[1]}</strong></div><span>{stageItems(activeStage).length} outputs</span></header>{summary?.stages?.find((s:any)=>s.stage_id===activeStage)?.summary&&<p className="foundation-stage-summary">{summary.stages.find((s:any)=>s.stage_id===activeStage).summary}</p>}<div className="foundation-artifact-columns"><div><strong>Figures ({figures.length})</strong>{figures.map(a=><button key={a.path} onClick={()=>setSelectedPath(a.path)}>{a.name}</button>)}</div><div><strong>Tables ({tables.length})</strong>{tables.map(a=><button key={a.path} onClick={()=>setSelectedPath(a.path)}>{a.name}</button>)}</div><div><strong>Other outputs ({otherOutputs.length})</strong>{otherOutputs.map(a=><button key={a.path} onClick={()=>setSelectedPath(a.path)}>{a.name} · {a.category}</button>)}</div></div>
          {selectedPath&&<article className="foundation-artifact-preview"><header><strong>{selectedPath.split('/').pop()}</strong><button onClick={()=>setSelectedPath('')}>Close</button></header>{selectedPath.match(/\.(png|jpg|jpeg|svg)$/i)?<img src={stageResultArtifactUrl(jobId,'foundations',selectedPath)} alt="Foundation stage figure"/>:selectedPath.endsWith('.csv')?<pre>{content.split('\n').slice(0,81).join('\n')}</pre>:<pre>{content.slice(0,30000)}</pre>}</article>}
        </section>
      </>}
    </div>;
}

export default function ProcessingWorkflowTab({ backendConnected, onOpenScripting, onAskGeo }: { backendConnected: boolean; onOpenScripting: (scriptName: string) => void; onAskGeo?: (question: string) => void }) {
    const [activeStageId, setActiveStageId] = useState<StageId>('foundations');
    const activeIndex = stages.findIndex(stage => stage.id === activeStageId);
    const activeStage = stages[activeIndex];
    const ActiveIcon = activeStage.icon;
    const [loadedConfig, setLoadedConfig] = useState<BackendProcessingConfig | null>(null);
    const [draftConfig, setDraftConfig] = useState<BackendProcessingConfig['config'] | null>(null);
    const [configLoading, setConfigLoading] = useState(false);
    const [configSaving, setConfigSaving] = useState(false);
    const [configError, setConfigError] = useState('');
    const [configNotice, setConfigNotice] = useState('');
    const [modelReleases, setModelReleases] = useState<ModelReleaseListItem[]>([]);
    const [processingJobs, setProcessingJobs] = useState<ProcessingJobListItem[]>([]);
    useEffect(() => {
        if (!backendConnected) { setModelReleases([]); setProcessingJobs([]); return; }
        let current = true;
        Promise.all([listModelReleases(500, 0), listProcessingJobs(500, 0)]).then(([releases, jobs]) => {
            if (!current) return;
            setModelReleases(releases.items ?? []);
            setProcessingJobs(jobs.items ?? []);
        }).catch(error => {
            console.warn('Could not load processing configuration choices:', error);
        });
        return () => { current = false; };
    }, [backendConnected]);
    const configDirty = !!loadedConfig && !!draftConfig && JSON.stringify(loadedConfig.config) !== JSON.stringify(draftConfig);

    const loadStageConfig = async (configId: string) => {
        setConfigLoading(true);
        setConfigError('');
        setConfigNotice('');
        try {
            const loaded = await getProcessingConfig(configId);
            setLoadedConfig(loaded);
            setDraftConfig(loaded.config);
        } catch (error) {
            setLoadedConfig(null);
            setDraftConfig(null);
            setConfigError(error instanceof Error ? error.message : 'Could not load stage configuration.');
        } finally {
            setConfigLoading(false);
        }
    };

    useEffect(() => {
        if (activeStage.id === 'geobody') return;
        let current = true;
        setConfigLoading(true);
        setConfigError('');
        setConfigNotice('');
        getProcessingConfig(activeStage.config.replace(/\.yaml$/, '')).then(loaded => {
            if (!current) return;
            setLoadedConfig(loaded);
            setDraftConfig(loaded.config);
        }).catch(error => {
            if (!current) return;
            setLoadedConfig(null);
            setDraftConfig(null);
            setConfigError(error instanceof Error ? error.message : 'Could not load stage configuration.');
        }).finally(() => { if (current) setConfigLoading(false); });
        return () => { current = false; };
    }, [activeStage.id, activeStage.config]);

    const saveStageConfig = async () => {
        if (!loadedConfig || !draftConfig) return;
        setConfigSaving(true);
        setConfigError('');
        setConfigNotice('');
        try {
            const saved = await updateProcessingConfig(loadedConfig.id, draftConfig, loadedConfig.revision);
            setLoadedConfig(saved);
            setDraftConfig(saved.config);
            setConfigNotice('Saved to the backend configuration directory.');
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Could not save stage configuration.';
            setConfigError(message);
        } finally {
            setConfigSaving(false);
        }
    };

    return <section className="page system-page processing-flow-page">
        <div className="page-title">
            <div>
                <span className="eyebrow">END-TO-END GEOSCIENCE WORKFLOW</span>
                <h1>Processing</h1>
                <p>Six linked stages. Each stage consumes registered upstream products and publishes auditable outputs.</p>
            </div>
            <div className="processing-flow-status">
                <span className={`processing-flow-api${backendConnected ? ' is-connected' : ''}`}><i />{backendConnected ? 'Backend connected' : 'Backend offline'}</span>
                <span>6 stages · A → F</span>
            </div>
        </div>

        <section className="processing-flow-overview theme-panel" aria-label="Processing workflow sequence">
            <div className="processing-flow-overview-head">
                <div><span className="eyebrow">WORKFLOW</span><h2>Processing sequence</h2></div>
                <span className="processing-flow-sequence">A <ArrowRight /> B <ArrowRight /> C <ArrowRight /> D <ArrowRight /> E <ArrowRight /> F</span>
            </div>
            <div className="processing-flow-steps" role="tablist" aria-label="Processing stages">
                {stages.map((stage, index) => {
                    const Icon = stage.icon;
                    const selected = stage.id === activeStageId;
                    return <button key={stage.id} type="button" role="tab" id={`processing-stage-tab-${stage.id}`} aria-selected={selected} aria-controls="processing-stage-panel" className={`processing-flow-step${selected ? ' active' : ''}`} onClick={() => setActiveStageId(stage.id)}>
                        <span className="processing-flow-step-marker"><b>{stage.letter}</b></span>
                        <span className="processing-flow-step-icon"><Icon /></span>
                        <strong>{stage.title}</strong>
                        <small>{stage.short}</small>
                    </button>;
                })}
            </div>
            <div className="processing-flow-progress" aria-hidden="true"><span style={{ width: `${(activeIndex / (stages.length - 1)) * 100}%` }} /></div>
            <div className="processing-flow-foot"><span>STAGE <strong>{activeStage.letter} / F</strong></span><span>{activeIndex + 1} of 6 selected</span></div>
        </section>

        <section id="processing-stage-panel" className="processing-stage-panel" role="tabpanel" aria-labelledby={`processing-stage-tab-${activeStage.id}`}>
            <header className="processing-stage-heading">
                <div className="processing-stage-icon"><ActiveIcon /></div>
                <div className="processing-stage-title">
                    <span className="eyebrow">STAGE {activeStage.letter} · {String(activeIndex + 1).padStart(2, '0')} / 06</span>
                    <h2>{activeStage.title}</h2>
                    <p>{activeStage.description}</p>
                </div>
                <span className="processing-stage-state"><Workflow /> {backendConnected ? 'Ready to configure' : 'Backend offline'}</span>
            </header>

            <details className="processing-stage-section processing-stage-run" open>
                <summary>
                    <span><strong>Run this stage</strong><small>Review inputs, configure parameters and open the job submission form</small></span>
                    <span className="processing-stage-section-hint">{activeStage.script}</span>
                </summary>
                <div className="processing-stage-section-content">
            <div className="processing-stage-io">
                <article className="processing-stage-card theme-panel">
                    <div className="processing-stage-card-head"><span className="processing-stage-card-icon input"><Database /></span><div><span className="eyebrow">INPUTS</span><h3>What this stage reads</h3></div></div>
                    <ul>{activeStage.inputs.map(item => <li key={item}>{item}</li>)}</ul>
                </article>
                <article className="processing-stage-card theme-panel">
                    <div className="processing-stage-card-head"><span className="processing-stage-card-icon output"><FileOutput /></span><div><span className="eyebrow">OUTPUTS</span><h3>What this stage writes</h3></div></div>
                    <ul>{activeStage.outputs.map(item => <li key={item}>{item}</li>)}</ul>
                </article>
            </div>
            {activeStage.id === 'geobody' ? <>
                <details className="processing-config-collapsible">
                    <summary><span><strong>Edit geobody parameters</strong><small>Expand to review or change the stage YAML settings</small></span><span className={`processing-config-state${configDirty ? ' dirty' : ''}`}>{configDirty ? 'Unsaved changes' : 'Parameters'}</span></summary>
                    <section className="processing-config-editor theme-panel">
                    <div className="processing-config-header">
                        <div><span className="eyebrow">STAGE D CONFIGURATION</span><h3>Edit geobody parameters</h3><p>Changes are saved to D_geobody.yaml on the backend.</p></div>
                        <div className="processing-config-actions">
                            <span className={`processing-config-state${configDirty ? ' dirty' : ''}`}>{configLoading ? 'Loading…' : configDirty ? 'Unsaved changes' : loadedConfig ? 'Saved configuration' : 'Unavailable'}</span>
                            <button type="button" className="btn btn-secondary" disabled={configLoading || configSaving} onClick={() => loadStageConfig('D_geobody')}>Reload</button>
                            <button type="button" className="btn btn-primary" disabled={!configDirty || configSaving || configLoading} onClick={saveStageConfig}>{configSaving ? 'Saving…' : 'Save YAML'}</button>
                        </div>
                    </div>
                    {configError && <div className="processing-config-message error">{configError}{configError.includes('changed on the backend') ? <button type="button" onClick={() => loadStageConfig('D_geobody')}>Reload latest</button> : null}</div>}
                    {configNotice && <div className="processing-config-message success">{configNotice}</div>}
                    {configLoading ? <div className="processing-config-loading">Loading configuration…</div> :
                        draftConfig ? <div className="processing-config-tree">{Object.entries(draftConfig).map(([key, value]) =>
                            <ConfigNode key={key} name={key} value={value} depth={0} stageId={activeStage.id} modelReleases={modelReleases} processingJobs={processingJobs} onChange={next => setDraftConfig(current => current ? { ...current, [key]: next } : current)}/>
                        )}</div> : <div className="processing-config-loading">Configuration could not be loaded.</div>}
                    <div className="processing-config-footnote">Saving validates the YAML structure and updates the stage configuration on the backend. A running job keeps its existing configuration snapshot.</div>
                    </section>
                </details>
            </> : <>
                <details className="processing-config-collapsible">
                    <summary><span><strong>Edit parameters</strong><small>Expand to review or change the stage YAML settings</small></span><span className={`processing-config-state${configDirty ? ' dirty' : ''}`}>{configDirty ? 'Unsaved changes' : 'Parameters'}</span></summary>
                    <section className="processing-config-editor theme-panel">
                    <div className="processing-config-header">
                        <div><span className="eyebrow">STAGE CONFIGURATION</span><h3>Edit parameters</h3><p>Changes are saved to the backend YAML used by this stage's worker.</p></div>
                        <div className="processing-config-actions">
                            <span className={`processing-config-state${configDirty ? ' dirty' : ''}`}>{configLoading ? 'Loading…' : configDirty ? 'Unsaved changes' : loadedConfig ? 'Saved configuration' : 'Unavailable'}</span>
                            <button type="button" className="btn btn-secondary" disabled={configLoading || configSaving} onClick={() => loadStageConfig(activeStage.config.replace(/\.yaml$/, ''))}>Reload</button>
                            <button type="button" className="btn btn-primary" disabled={!configDirty || configSaving || configLoading} onClick={saveStageConfig}>{configSaving ? 'Saving…' : 'Save YAML'}</button>
                        </div>
                    </div>
                    {configError && <div className="processing-config-message error">{configError}{configError.includes('changed on the backend') ? <button type="button" onClick={() => loadStageConfig(activeStage.config.replace(/\.yaml$/, ''))}>Reload latest</button> : null}</div>}
                    {configNotice && <div className="processing-config-message success">{configNotice}</div>}
                    {configLoading ? <div className="processing-config-loading">Loading configuration…</div> :
                        draftConfig ? <div className="processing-config-tree">{Object.entries(draftConfig).map(([key, value]) =>
                            <ConfigNode key={key} name={key} value={value} depth={0} stageId={activeStage.id} modelReleases={modelReleases} processingJobs={processingJobs} onChange={next => setDraftConfig(current => current ? { ...current, [key]: next } : current)}/>
                        )}</div> : <div className="processing-config-loading">Configuration could not be loaded.</div>}
                    <div className="processing-config-footnote">Saving validates the YAML structure and updates the stage configuration on the backend. A running job keeps its existing configuration snapshot.</div>
                    </section>
                </details>


            </>}
                <div className="processing-stage-script">
                    <span><FileCode2 /> <b>{activeStage.script}</b></span>
                    <span><Code2 /> {activeStage.config}</span>
                </div>

                <div className="processing-stage-settings">
                    <span className="eyebrow">CONFIGURATION NOTES</span>
                    <div>{activeStage.settings.map(item => <span key={item}>{item}</span>)}</div>
                </div>
            <div className="processing-stage-action">
                <div><strong>Run this stage</strong><p>Open Scripting with <code>{activeStage.script}</code> selected. Review inputs and submit the job when ready.</p></div>
                <button type="button" className="btn btn-primary" onClick={() => onOpenScripting(activeStage.script)}>Run this stage <Code2 /></button>
            </div>
            {onAskGeo && <div className="processing-stage-action">
                <div><strong>Understand the results</strong><p>Ask Geo reads the latest Process {activeStage.letter} run and explains its verdict, key tables and next steps in plain language.</p></div>
                <button type="button" className="btn btn-secondary" onClick={() => onAskGeo(`Explain the results of Process ${activeStage.letter}`)}>Explain results <MessageSquare /></button>
            </div>}
                </div>
            </details>
            <details className="processing-stage-section processing-stage-result">
                <summary>
                    <span><strong>Result Viewer</strong><small>{activeStage.id === 'geobody' ? 'Inspect extracted geobodies, slices and exports' : activeStage.id === 'volume-products' ? 'Inspect summary, figures, tables, validation and native exports' : 'Inspect outputs produced by this stage'}</small></span>
                    <span className="processing-stage-section-hint">{activeStage.id === 'geobody' ? 'Stage 12 viewer' : 'Viewer availability varies by output type'}</span>
                </summary>
                <div className="processing-stage-section-content">
                    {activeStage.id === 'geobody' ? <GeobodyTab backendConnected={backendConnected} /> : activeStage.id === 'inference' ? <InferenceResultViewer jobs={processingJobs} backendConnected={backendConnected} /> : activeStage.id === 'volume-products' ? <VolumeProductsResultViewer jobs={processingJobs} backendConnected={backendConnected} /> : activeStage.id === 'well-geobody' ? <StageScientificResultViewer stage="well-geobody" jobs={processingJobs} backendConnected={backendConnected} /> : activeStage.id === 'prospect' ? <StageScientificResultViewer stage="prospect" jobs={processingJobs} backendConnected={backendConnected} /> : activeStage.id === 'foundations' ? <FoundationsResultViewer jobs={processingJobs} backendConnected={backendConnected} /> :
                        <div className="processing-stage-viewer-empty"><strong>No dedicated viewer is embedded for this stage yet.</strong><p>Use the stage outputs and the relevant project viewer to inspect generated products. The processing job and its artifacts remain available through Scripting.</p></div>}
                </div>
            </details>
        </section>
        <div className="processing-flow-next">
            <span>{activeIndex === stages.length - 1 ? 'END OF WORKFLOW' : `NEXT · ${stages[activeIndex + 1].letter} ${stages[activeIndex + 1].title.toUpperCase()}`}</span>
            <button type="button" className="btn btn-secondary" disabled={activeIndex === stages.length - 1} onClick={() => setActiveStageId(stages[activeIndex + 1].id)}>Next stage <ArrowRight /></button>
        </div>
    </section>;
}

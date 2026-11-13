import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Play, RefreshCw, Trash2, UploadCloud } from 'lucide-react';
import MarkdownMessage from './MarkdownMessage';
import {
  deleteModelRelease, deleteProcessingJob, deleteProcessingScript, getProcessingJob, getProcessingJobFile,
  getModelRelease, listModelReleases, validateModelRelease,
  getProcessingScript, listDatasetsFiltered, listProcessingJobFiles, listProcessingJobs,
  listProcessingScripts, submitProcessingJob, updateProcessingScript, uploadProcessingScript,
  type BackendDataset,
  type ProcessingJobResponse, type ModelReleaseListItem, type ModelReleaseDetail, type ModelReleaseValidation,
} from '../../api/backend';


type ProcessingResult = { id: string; title: string; ok: boolean; message: string; time: string };
type SubmittedJob = { id: string; title: string; submittedAt: string; response: ProcessingJobResponse };
type ProcessingScript = { name: string; content: string; size_bytes?: number; updated_at?: string };


function parseCsv(content: string): string[][] {
    const rows: string[][] = [];
    let row: string[] = [];
    let field = '';
    let quoted = false;
    const input = content.replace(/^\uFEFF/, '');
    for (let i = 0; i < input.length; i += 1) {
        const char = input[i];
        if (quoted) {
            if (char === '"' && input[i + 1] === '"') { field += '"'; i += 1; }
            else if (char === '"') quoted = false;
            else field += char;
        } else if (char === '"' && field.length === 0) quoted = true;
        else if (char === ',') { row.push(field); field = ''; }
        else if (char === '\n' || char === '\r') {
            if (char === '\r' && input[i + 1] === '\n') i += 1;
            row.push(field); field = '';
            if (row.some(value => value.length > 0)) rows.push(row);
            row = [];
        } else field += char;
    }
    row.push(field);
    if (row.some(value => value.length > 0)) rows.push(row);
    return rows;
}

function ProcessingError({ title, message }: { title: string; message: string }) {
    return <div className="system-error"><AlertTriangle/><div><strong>{title}</strong><p>{message}</p></div></div>;
}

function ProcessingPage({ projectId, backendConnected, initialJobId, initialScriptName }: {
    projectId: string;
    backendConnected: boolean;
    initialJobId?: string;
    initialScriptName?: string;
}) {
    const [datasets, setDatasets] = useState<BackendDataset[]>([]);
    const [datasetIds, setDatasetIds] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);
    const [loadingScripts, setLoadingScripts] = useState(false);
    const [savingScript, setSavingScript] = useState(false);
    const [running, setRunning] = useState(false);
    const [results, setResults] = useState<ProcessingResult[]>([]);
    const [submittedJobs, setSubmittedJobs] = useState<SubmittedJob[]>([]);
    const [selectedJobId, setSelectedJobId] = useState('');
    const jobBrowserRef = useRef<HTMLDivElement>(null);
    const [selectedJobDetails, setSelectedJobDetails] = useState<ProcessingJobResponse | null>(null);
    const [jobDetailsLoading, setJobDetailsLoading] = useState(false);
    const [jobDetailsError, setJobDetailsError] = useState('');
    const [jobFiles, setJobFiles] = useState<{ name: string; size_bytes: number; updated_at: number }[]>([]);
    const [selectedJobFile, setSelectedJobFile] = useState('');
    const [selectedJobFileContent, setSelectedJobFileContent] = useState('');
    const [jobFilesLoading, setJobFilesLoading] = useState(false);
    const [jobFileError, setJobFileError] = useState('');
    const [loadError, setLoadError] = useState('');
    const [scriptError, setScriptError] = useState('');
    const [scripts, setScripts] = useState<ProcessingScript[]>([]);
    const [scriptName, setScriptName] = useState('');
    const [scriptContent, setScriptContent] = useState('');
    const [scriptMode, setScriptMode] = useState<'view' | 'edit'>('view');
    const [modelReleases, setModelReleases] = useState<ModelReleaseListItem[]>([]);
    const [modelReleasesLoading, setModelReleasesLoading] = useState(false);
    const [modelReleaseError, setModelReleaseError] = useState('');
    const [selectedReleaseId, setSelectedReleaseId] = useState('');
    const [selectedRelease, setSelectedRelease] = useState<ModelReleaseDetail | null>(null);
    const [releaseValidation, setReleaseValidation] = useState<ModelReleaseValidation | null>(null);
    const [releaseDetailLoading, setReleaseDetailLoading] = useState(false);
    const loadJobInspector = async (jobId: string, preferredFile?: string) => {
        if (!jobId) return;
        setJobDetailsLoading(true); setJobFilesLoading(true); setJobDetailsError(''); setJobFileError('');
        try {
            const [detail, files] = await Promise.all([getProcessingJob(jobId), listProcessingJobFiles(jobId)]);
            setSelectedJobDetails(detail);
            setJobFiles(files.items);
            const nextFile = preferredFile && files.items.some(file => file.name === preferredFile) ? preferredFile : (files.items.some(file => file.name === selectedJobFile) ? selectedJobFile : files.items.find(file => /\.(csv|md|json|log|txt|out|err)$/i.test(file.name))?.name ?? '');
            setSelectedJobFile(nextFile);
            if (nextFile) {
                const content = await getProcessingJobFile(jobId, nextFile);
                setSelectedJobFileContent(content.content);
            } else setSelectedJobFileContent('');
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Could not inspect processing job.';
            setJobDetailsError(message); setJobFileError(message); setJobFiles([]); setSelectedJobFileContent('');
        } finally { setJobDetailsLoading(false); setJobFilesLoading(false); }
    };
    const loadExistingJobs = async () => {
        if (!backendConnected) { setSubmittedJobs([]); return; }
        try {
            const response = await listProcessingJobs(100, 0);
            const jobs = response.items.map(item => {
                const id = String(item.id ?? item.job_id ?? '');
                const title = `${String(item.python_script ?? item.operation ?? 'Processing job')} · ${id}`;
                return { id, title, submittedAt: String(item.created_at ?? ''), response: item };
            }).filter(item => item.id);
            setSubmittedJobs(current => {
                const byId = new Map<string, SubmittedJob>();
                jobs.forEach(job => byId.set(job.id, job));
                current.forEach(job => { if (!byId.has(job.id)) byId.set(job.id, job); });
                return Array.from(byId.values());
            });
        } catch (error) { setJobDetailsError(error instanceof Error ? error.message : 'Could not list processing jobs.'); }
    };
    const loadModelReleases = async (preferredId?: string) => {
        if (!backendConnected) { setModelReleases([]); return; }
        setModelReleasesLoading(true); setModelReleaseError('');
        try {
            const response = await listModelReleases(100, 0);
            setModelReleases(response.items);
            const nextId = preferredId && response.items.some(item => item.model_release_id === preferredId)
                ? preferredId : (response.items.some(item => item.model_release_id === selectedReleaseId) ? selectedReleaseId : response.items[0]?.model_release_id ?? '');
            setSelectedReleaseId(nextId);
            if (nextId) await inspectModelRelease(nextId);
            else { setSelectedRelease(null); setReleaseValidation(null); }
        } catch (error) { setModelReleaseError(error instanceof Error ? error.message : 'Could not load model releases.'); }
        finally { setModelReleasesLoading(false); }
    };
    const inspectModelRelease = async (releaseId: string) => {
        if (!releaseId) return;
        setReleaseDetailLoading(true); setModelReleaseError(''); setReleaseValidation(null);
        try { setSelectedRelease(await getModelRelease(releaseId)); }
        catch (error) { setSelectedRelease(null); setModelReleaseError(error instanceof Error ? error.message : 'Could not inspect model release.'); }
        finally { setReleaseDetailLoading(false); }
    };
    const checkModelRelease = async () => {
        if (!selectedReleaseId) return;
        setReleaseDetailLoading(true); setModelReleaseError('');
        try { setReleaseValidation(await validateModelRelease(selectedReleaseId)); }
        catch (error) { setModelReleaseError(error instanceof Error ? error.message : 'Could not validate model release.'); }
        finally { setReleaseDetailLoading(false); }
    };
    const removeModelRelease = async () => {
        if (!selectedReleaseId) return;
        const releaseId = selectedReleaseId;
        const release = modelReleases.find(item => item.model_release_id === releaseId);
        const sourceJobId = release?.processing_job_id ?? String(selectedRelease?.manifest.foundation_job_id ?? selectedRelease?.manifest.processing_job_id ?? '');
        const pairedJob = sourceJobId ? ` and its source processing job (${sourceJobId})` : ' and its linked source processing job, if present';
        if (!window.confirm(`Delete model release ${releaseId}${pairedJob}? Both records and their stored folders will be removed. This cannot be undone.`)) return;
        setReleaseDetailLoading(true); setModelReleaseError('');
        try {
            await deleteModelRelease(releaseId);
            setSelectedReleaseId(''); setSelectedRelease(null); setReleaseValidation(null);
            if (sourceJobId) {
                setSubmittedJobs(current => current.filter(job => job.id !== sourceJobId));
                if (selectedJobId === sourceJobId) {
                    setSelectedJobId(''); setSelectedJobDetails(null); setJobFiles([]); setSelectedJobFile(''); setSelectedJobFileContent('');
                }
            }
            await Promise.all([loadModelReleases(), loadExistingJobs()]);
            setResults(old => [{ id: `${Date.now()}-release-delete`, title: `Deleted ${releaseId}`, ok: true, message: 'Model release and linked processing job (when present) deleted.', time: new Date().toLocaleTimeString() }, ...old]);
        } catch (error) { setModelReleaseError(error instanceof Error ? error.message : 'Could not delete model release.'); }
        finally { setReleaseDetailLoading(false); }
    };
    const loadDatasets = async () => {
        if (!projectId || !backendConnected) { setDatasets([]); setDatasetIds([]); return; }
        setLoading(true); setLoadError('');
        try {
            const result = await listDatasetsFiltered({ project_id: projectId, limit: 200, offset: 0 });
            setDatasets(result.items);
            setDatasetIds(current => current.filter(id => result.items.some(item => item.id === id)));
        } catch (error) { setLoadError(error instanceof Error ? error.message : 'Could not load project datasets.'); }
        finally { setLoading(false); }
    };
    const loadScripts = async (preferredName?: string) => {
        if (!backendConnected) { setScripts([]); setScriptName(''); setScriptContent(''); return; }
        setLoadingScripts(true); setScriptError('');
        try {
            const result = await listProcessingScripts();
            const items = result.items.map(item => ({ name: item.id, content: '', size_bytes: item.size_bytes, updated_at: item.updated_at }));
            setScripts(items);
            const nextName = preferredName && items.some(item => item.name === preferredName) ? preferredName : (items.some(item => item.name === scriptName) ? scriptName : items[0]?.name ?? '');
            setScriptName(nextName);
            if (nextName) {
                const detail = await getProcessingScript(nextName);
                setScriptContent(detail.content ?? '');
                setScripts(current => current.map(item => item.name === nextName ? { ...item, content: detail.content ?? '', size_bytes: detail.size_bytes, updated_at: detail.updated_at } : item));
            } else setScriptContent('');
        } catch (error) { setScriptError(error instanceof Error ? error.message : 'Could not load processing scripts.'); }
        finally { setLoadingScripts(false); }
    };
    useEffect(() => { void loadDatasets(); }, [projectId, backendConnected]);
    useEffect(() => { void loadScripts(); }, [backendConnected]);
    useEffect(() => {
        if (!initialScriptName || !scripts.some(script => script.name === initialScriptName) || scriptName === initialScriptName) return;
        void chooseScript(initialScriptName);
    }, [initialScriptName, scripts]);
    useEffect(() => { void loadExistingJobs(); }, [backendConnected]);
    useEffect(() => { void loadModelReleases(); }, [backendConnected]);
    useEffect(() => {
        if (!initialJobId || selectedJobId === initialJobId) return;
        setSelectedJobId(initialJobId);
        setSelectedJobDetails(null);
        setJobFiles([]);
        setSelectedJobFile('');
        setSelectedJobFileContent('');
        void loadJobInspector(initialJobId);
    }, [initialJobId, selectedJobId]);
    useEffect(() => {
        if (!initialJobId) return;
        jobBrowserRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, [initialJobId]);
    const chooseScript = async (name: string) => {
        setScriptName(name); setScriptMode('view'); setScriptError('');
        if (!name) { setScriptContent(''); return; }
        try {
            const detail = await getProcessingScript(name);
            setScriptContent(detail.content ?? '');
            setScripts(old => old.map(script => script.name === name ? { ...script, content: detail.content ?? '', size_bytes: detail.size_bytes, updated_at: detail.updated_at } : script));
        } catch (error) { setScriptError(error instanceof Error ? error.message : 'Could not read this script.'); }
    };
    const uploadScript = async (file?: File) => {
        if (!file) return;
        setScriptError('');
        if (!file.name.toLowerCase().endsWith('.py')) { setScriptError('Please choose a .py file.'); return; }
        try {
            const uploaded = await uploadProcessingScript(file);
            await loadScripts(uploaded.id);
            setResults(old => [{ id: `${Date.now()}-upload`, title: `Uploaded ${uploaded.name}`, ok: true, message: 'Script saved to the backend Data Directory.', time: new Date().toLocaleTimeString() }, ...old]);
        } catch (error) { setScriptError(error instanceof Error ? error.message : 'Could not upload the script.'); }
    };
    const saveScript = async () => {
        if (!scriptName) return;
        setSavingScript(true); setScriptError('');
        try {
            const saved = await updateProcessingScript(scriptName, scriptContent);
            setScripts(old => old.map(script => script.name === scriptName ? { ...script, content: saved.content ?? scriptContent, size_bytes: saved.size_bytes, updated_at: saved.updated_at } : script));
            setScriptMode('view');
            setResults(old => [{ id: `${Date.now()}-save`, title: `Saved ${scriptName}`, ok: true, message: 'Script changes saved to the backend Data Directory.', time: new Date().toLocaleTimeString() }, ...old]);
        } catch (error) { setScriptError(error instanceof Error ? error.message : 'Could not save the script.'); }
        finally { setSavingScript(false); }
    };
    const removeScript = async () => {
        if (!scriptName || !window.confirm(`Delete ${scriptName} from the backend Data Directory?`)) return;
        const removed = scriptName;
        try { await deleteProcessingScript(removed); await loadScripts(); }
        catch (error) { setScriptError(error instanceof Error ? error.message : 'Could not delete the script.'); }
    };
    const loadInspectorAfterSubmit = async (jobId: string) => { await loadExistingJobs(); await loadJobInspector(jobId); };
    const removeProcessingJob = async () => {
        if (!selectedJobId) return;
        const selected = submittedJobs.find(job => job.id === selectedJobId);
        if (!window.confirm(`Delete processing job ${selectedJobId} and permanently remove its folder and database record? This cannot be undone.`)) return;
        setJobDetailsError('');
        try {
            await deleteProcessingJob(selectedJobId);
            setSubmittedJobs(current => current.filter(job => job.id !== selectedJobId));
            setSelectedJobId('');
            setSelectedJobDetails(null); setJobFiles([]); setSelectedJobFile(''); setSelectedJobFileContent('');
            await loadModelReleases();
            setResults(old => [{ id: `${Date.now()}-delete`, title: `Deleted ${selected?.title ?? selectedJobId}`, ok: true, message: 'Processing job, folder, and any linked model release deleted.', time: new Date().toLocaleTimeString() }, ...old]);
        } catch (error) {
            setJobDetailsError(error instanceof Error ? error.message : 'Could not delete processing job.');
        }
    };
    const submitJob = async () => {
        if (!scriptName) return;
        setRunning(true);
        const title = `Processing job · ${scriptName}`;
        try {
            const response = await submitProcessingJob({ python_script: scriptName, dataset_ids: datasetIds });
            const jobId = String(response.id ?? response.job_id ?? '');
            setResults(old => [{ id: `${Date.now()}-job`, title, ok: true, message: JSON.stringify(response), time: new Date().toLocaleTimeString() }, ...old]);
            if (jobId) {
                const job = { id: jobId, title: `${scriptName} · ${jobId}`, submittedAt: new Date().toLocaleTimeString(), response };
                setSubmittedJobs(old => [job, ...old.filter(item => item.id !== jobId)]);
                setSelectedJobId(jobId);
                setSelectedJobDetails(response);
                void loadInspectorAfterSubmit(jobId);
            }
        } catch (error) {
            const message = error instanceof Error ? error.message : 'The processing request failed.';
            setResults(old => [{ id: `${Date.now()}-job`, title, ok: false, message, time: new Date().toLocaleTimeString() }, ...old]);
        } finally { setRunning(false); }
    };
    const metadataContent = JSON.stringify(selectedJobDetails ?? submittedJobs.find(job => job.id === selectedJobId)?.response ?? {}, null, 2);
    const metadataLineCount = metadataContent.split('\n').length;
    const selectedFileIsCsv = /\.csv$/i.test(selectedJobFile);
    const selectedFileIsMarkdown = /\.md$/i.test(selectedJobFile);
    const csvRows = selectedFileIsCsv ? parseCsv(selectedJobFileContent) : [];
    const csvHeaders = csvRows[0] ?? [];
    const csvDataRows = csvRows.slice(1);
    const visibleCsvRows = csvDataRows.slice(0, 500);

    return <section className="page system-page processing-page">
      <div className="page-title"><div><span className="eyebrow">SCRIPTING · EXECUTION ENVIRONMENT</span><h1>Scripting</h1><p>Manage backend Python scripts, select input datasets, submit jobs, and inspect execution outputs.</p></div><div className="title-actions theme-action-group"><button className="btn btn-secondary" onClick={() => void loadDatasets()} disabled={loading || !backendConnected}><RefreshCw className={loading ? 'spin' : ''}/>{loading ? 'Loading…' : 'Reload datasets'}</button><button className="btn btn-secondary" onClick={() => void loadScripts()} disabled={loadingScripts || !backendConnected}><RefreshCw className={loadingScripts ? 'spin' : ''}/>{loadingScripts ? 'Loading scripts…' : 'Reload scripts'}</button></div></div>
      {!backendConnected && <ProcessingError title="Backend unavailable" message="Connect to the API before submitting processing requests."/>}
      {loadError && <ProcessingError title="Could not load datasets" message={loadError}/>}
      {scriptError && <ProcessingError title="Script operation failed" message={scriptError}/>}
      <div className="processing-workspace">
        <section className="system-section processing-column theme-panel"><div className="system-section-heading"><div><span className="eyebrow">COLUMN 1</span><h2>Input Resource</h2><p>Optionally select datasets to include in <code>dataset_ids</code>.</p></div><span>{datasetIds.length} selected</span></div>
          <label className="processing-field"><span>Project datasets</span><select multiple size={Math.min(10, Math.max(4, datasets.length))} value={datasetIds} onChange={event => setDatasetIds(Array.from(event.currentTarget.selectedOptions, (option: HTMLOptionElement) => option.value))} disabled={loading || !datasets.length}>{datasets.map(item => <option key={item.id} value={item.id}>{item.name} · {item.format.toUpperCase()} · {item.status}</option>)}</select><small>Use Ctrl (Windows/Linux) or Command (macOS) to select multiple datasets.</small></label>
          <div className="processing-selected-resource">{datasetIds.length ? datasets.filter(item => datasetIds.includes(item.id)).map(item => <div key={item.id} className="processing-resource-chip"><strong>{item.name}</strong><span>{item.kind} · {item.format.toUpperCase()}</span><code>{item.id}</code><button type="button" className="btn btn-secondary" onClick={() => setDatasetIds(ids => ids.filter(id => id !== item.id))}>Remove</button></div>) : <div className="processing-empty-selection">{loading ? 'Loading project datasets…' : datasets.length ? 'No datasets selected yet.' : 'No datasets available. You can still submit a job without dataset inputs.'}</div>}</div>
        </section>
        <section className="system-section processing-column theme-panel"><div className="system-section-heading"><div><span className="eyebrow">COLUMN 2</span><h2>Operation</h2><p>Choose a backend-managed Python script to execute.</p></div></div>
          <label className="processing-field"><span>Python script</span><select value={scriptName} onChange={event => void chooseScript(event.target.value)} disabled={loadingScripts || !scripts.length}>{scripts.map(script => <option key={script.name} value={script.name}>{script.name}</option>)}</select></label>
          <div className="processing-script-toolbar"><div className="processing-toggle"><button className={`btn-segment ${scriptMode === 'view' ? 'active' : ''}`} onClick={() => setScriptMode('view')} type="button">Viewer</button><button className={`btn-segment ${scriptMode === 'edit' ? 'active' : ''}`} onClick={() => setScriptMode('edit')} type="button" disabled={!scriptName}>Editor</button></div><label className="btn btn-secondary processing-upload-script"><UploadCloud /> Upload .py<input type="file" accept=".py,text/x-python,text/plain" onChange={event => { void uploadScript(event.currentTarget.files?.[0]); event.currentTarget.value = ''; }} /></label></div>
          {scriptMode === 'edit' ? <textarea className="processing-script-editor" value={scriptContent} onChange={event => setScriptContent(event.target.value)} spellCheck={false} aria-label="Python script editor" /> : <pre className="processing-script-viewer">{scriptContent || (loadingScripts ? 'Loading script…' : scripts.length ? '# Select a script to view its contents.' : '# No scripts found in the backend Data Directory.')}</pre>}
          <div className="processing-submit-row"><span>{scriptName || 'No script selected'}</span><div className="title-actions theme-action-group"><button type="button" className="btn btn-secondary" onClick={() => void saveScript()} disabled={!scriptName || scriptMode !== 'edit' || savingScript}>{savingScript ? 'Saving…' : 'Save changes'}</button><button type="button" className="btn btn-danger" onClick={() => void removeScript()} disabled={!scriptName}>Delete</button></div></div>
          <div className="processing-submit-row"><span>{datasetIds.length ? `${datasetIds.length} dataset${datasetIds.length === 1 ? '' : 's'}` : 'No dataset inputs'} · {scriptName || 'No script'}</span><button className="btn btn-primary" onClick={() => void submitJob()} disabled={!backendConnected || !scriptName || running || scriptMode === 'edit'}>{running ? 'Submitting…' : 'Submit job'}<Play /></button></div>
          <p className="processing-caveat"><AlertTriangle /> Scripts are listed, uploaded, edited, and deleted on the backend in the configured Data Directory. Save edits before submitting a job.</p>
        </section>
      </div>
      <div className="system-section processing-model-releases theme-panel">
        <div className="system-section-heading"><div><span className="eyebrow">TRAINED MODEL REGISTRY</span><h2>Model releases</h2><p>Reusable model packages created after successful GeoGG Stage 07 training. A release and its source processing job share one deletion lifecycle.</p></div><button className="btn btn-secondary" onClick={() => void loadModelReleases(selectedReleaseId)} disabled={modelReleasesLoading || !backendConnected}><RefreshCw className={modelReleasesLoading ? 'spin' : ''}/>{modelReleasesLoading ? 'Loading…' : 'Reload releases'}</button></div>
        {modelReleaseError && <ProcessingError title="Model release operation failed" message={modelReleaseError}/>}
        {!modelReleases.length ? <div className="page-empty">{modelReleasesLoading ? 'Loading registered model releases…' : 'No model releases found. Complete a GeoGG training run through Stage 07 to register a reusable release.'}</div> : <div className="processing-release-layout">
          <section className="processing-inspector-panel"><div className="processing-inspector-panel-heading"><h3>Available releases</h3><span>{modelReleases.length}</span></div><ul className="processing-job-file-list processing-release-list">{modelReleases.map(release => <li key={release.model_release_id}><button type="button" className={`processing-file-button ${selectedReleaseId === release.model_release_id ? 'active' : ''}`} onClick={() => { setSelectedReleaseId(release.model_release_id); void inspectModelRelease(release.model_release_id); }}><strong>{release.model_release_id}</strong><small>{release.model_family ?? 'Model'} · {release.model_count} model files · {release.status}</small></button></li>)}</ul></section>
          <section className="processing-inspector-panel"><div className="processing-inspector-panel-heading"><h3>Release details</h3><div className="title-actions theme-action-group"><button className="btn btn-secondary" disabled={!selectedReleaseId || releaseDetailLoading} onClick={() => void inspectModelRelease(selectedReleaseId)}><RefreshCw/>Inspect</button><button className="btn btn-secondary" disabled={!selectedReleaseId || releaseDetailLoading} onClick={() => void checkModelRelease()}>{releaseDetailLoading ? 'Checking…' : 'Verify integrity'}</button><button className="btn btn-danger" disabled={!selectedReleaseId || releaseDetailLoading} onClick={() => void removeModelRelease()}><Trash2/>Delete</button></div></div>
            {releaseValidation && <div className={`processing-release-validation ${releaseValidation.valid ? 'valid' : 'invalid'}`}><strong>{releaseValidation.valid ? 'Integrity verified' : 'Integrity check failed'}</strong><span>{releaseValidation.checked_files} files checked</span>{releaseValidation.errors.map((error, index) => <small key={index}>{error}</small>)}</div>}
            <pre className="processing-job-details">{releaseDetailLoading ? 'Loading release manifest…' : selectedRelease ? JSON.stringify(selectedRelease.manifest, null, 2) : 'Select a release to inspect its manifest.'}</pre>
          </section>
        </div>}
      </div>
      <div ref={jobBrowserRef} className="system-section processing-job-browser theme-panel">
        <div className="system-section-heading"><div><span className="eyebrow">JOB INSPECTOR</span><h2>Processing job folder</h2><p>Choose any saved processing job to inspect its folder contents, including CSV tables and text-based output files.</p></div><div className="title-actions theme-action-group"><button className="btn btn-secondary" onClick={() => void loadExistingJobs()} disabled={jobDetailsLoading || jobFilesLoading}><RefreshCw/>Reload jobs</button><button className="btn btn-secondary" disabled={!selectedJobId || jobDetailsLoading || jobFilesLoading} onClick={() => void loadJobInspector(selectedJobId)}><RefreshCw className={jobDetailsLoading || jobFilesLoading ? 'spin' : ''}/>{jobDetailsLoading || jobFilesLoading ? 'Refreshing…' : 'Refresh job'}</button><button className="btn btn-danger" disabled={!selectedJobId || jobDetailsLoading || jobFilesLoading} onClick={() => void removeProcessingJob()}><Trash2/>Delete job</button></div></div>
        <label className="processing-field"><span>Processing job</span><select value={selectedJobId} onChange={event => { const id = event.currentTarget.value; setSelectedJobId(id); setSelectedJobDetails(null); setJobFiles([]); setSelectedJobFile(''); setSelectedJobFileContent(''); if (id) void loadJobInspector(id); }} disabled={!submittedJobs.length || jobDetailsLoading}><option value="">Select a processing job…</option>{submittedJobs.map(job => <option key={job.id} value={job.id}>{job.title}</option>)}</select><small>Jobs are loaded from the backend, including jobs from earlier frontend sessions.</small></label>
        {jobDetailsError && <ProcessingError title="Could not inspect processing job" message={jobDetailsError}/>}
        {!selectedJobId ? <div className="page-empty">No job selected. Submit a job or reload the job list.</div> : <div className="processing-job-inspector-grid">
          <section className="processing-inspector-panel processing-metadata-panel"><div className="processing-inspector-panel-heading"><h3>Job metadata</h3><span>{metadataLineCount} lines</span></div><pre className="processing-job-details processing-metadata">{jobDetailsLoading ? 'Loading job details…' : metadataContent}</pre></section>
          <section className="processing-inspector-panel"><div className="processing-inspector-panel-heading"><h3>Actual folder files</h3><span>{jobFiles.length} {jobFiles.length === 1 ? 'file' : 'files'}</span></div>{jobFilesLoading ? <p>Loading folder files…</p> : jobFiles.length ? <ul className="processing-job-file-list">{jobFiles.map(file => <li key={file.name}><button type="button" className={`processing-file-button ${selectedJobFile === file.name ? 'active' : ''}`} onClick={async () => { setSelectedJobFile(file.name); setJobFileError(''); setSelectedJobFileContent(''); try { const result = await getProcessingJobFile(selectedJobId, file.name); setSelectedJobFileContent(result.content); } catch (error) { setJobFileError(error instanceof Error ? error.message : 'Could not read file.'); } }}>{file.name}</button><small>{file.size_bytes.toLocaleString()} bytes</small></li>)}</ul> : <p className="processing-job-note">No files found in this job folder.</p>}</section>
          <section className="processing-inspector-panel"><div className="processing-inspector-panel-heading"><h3>Selected file</h3><span>{selectedJobFile || 'No file selected'}</span></div>{jobFileError && <ProcessingError title="Could not read file" message={jobFileError}/>}{selectedFileIsCsv && !jobFilesLoading && !jobFileError ? (csvHeaders.length ? <div className="processing-csv-panel"><div className="processing-csv-meta">{csvHeaders.length} columns · {csvDataRows.length.toLocaleString()} data rows{csvDataRows.length > visibleCsvRows.length ? ` · showing first ${visibleCsvRows.length}` : ''}</div><div className="processing-csv-scroll"><table className="processing-csv-table"><thead><tr><th>#</th>{csvHeaders.map((header, index) => <th key={`${header}-${index}`}>{header || `Column ${index + 1}`}</th>)}</tr></thead><tbody>{visibleCsvRows.map((row, rowIndex) => <tr key={rowIndex}><td>{rowIndex + 1}</td>{csvHeaders.map((_, columnIndex) => <td key={columnIndex}>{row[columnIndex] ?? ''}</td>)}</tr>)}</tbody></table></div></div> : <div className="processing-job-details">{selectedJobFileContent ? 'No CSV rows found.' : 'This file is empty.'}</div>) : selectedFileIsMarkdown && !jobFilesLoading && !jobFileError ? (selectedJobFileContent ? <div className="processing-markdown-preview"><MarkdownMessage content={selectedJobFileContent}/></div> : <div className="processing-job-details">This file is empty.</div>) : <pre className="processing-job-details">{jobFilesLoading ? 'Loading…' : selectedJobFile ? selectedJobFileContent || (jobFileError ? '' : 'This file is empty.') : 'Select a supported text, JSON, Markdown, log, or CSV file.'}</pre>}</section>
        </div>}
      </div>
      <div className="system-section processing-request-log theme-panel"><div className="system-section-heading"><div><span className="eyebrow">REQUEST LOG</span><h2>Recent submissions</h2></div><button className="btn btn-secondary" onClick={() => setResults([])} disabled={!results.length}>Clear log</button></div>
        {!results.length ? <div className="page-empty">No processing requests submitted in this session.</div> : <div className="system-list">{results.map(result => <article key={result.id}><div><strong>{result.title}</strong><small>{result.time}</small><p>{result.message}</p></div><span className={`availability ${result.ok ? 'available' : 'placeholder'}`}>{result.ok ? 'Accepted' : 'Failed'}</span></article>)}</div>}
      </div>
    </section>;
}

export default ProcessingPage;

import { useEffect, useState } from 'react';
import { AlertTriangle, Check, RefreshCw } from 'lucide-react';
import { getRuntimeSettings, updateRuntimeSettings, type BackendRuntimeSettings } from '../api/backend';

export default function SettingsPage({ backendConnected, setToast }: {
    backendConnected: boolean;
    setToast: (message: string) => void;
}) {
    const [settings, setSettings] = useState<BackendRuntimeSettings | null>(null);
    const [draft, setDraft] = useState<BackendRuntimeSettings | null>(null);
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState('');
    const [dirty, setDirty] = useState(false);
    const formats = ['sgy', 'segy', 'las', 'csv', 'txt', 'xlsx', 'xls', 'npy', 'npz', 'h5', 'hdf5', 'zarr', 'vtk'];
    const load = async () => {
        setLoading(true);
        setError('');
        try {
            const value = await getRuntimeSettings();
            setSettings(value);
            setDraft(structuredClone(value));
            setDirty(false);
        }
        catch (err) {
            setError(err instanceof Error ? err.message : 'Could not load settings.');
        }
        finally {
            setLoading(false);
        }
    };
    useEffect(() => { void load(); }, []);
    const updateIngestion = (patch: Partial<BackendRuntimeSettings['ingestion']>) => {
        setDraft(current => current ? { ...current, ingestion: { ...current.ingestion, ...patch } } : current);
        setDirty(true);
    };
    const updateProcessing = (patch: Partial<BackendRuntimeSettings['processing']>) => {
        setDraft(current => current ? { ...current, processing: { ...current.processing, ...patch } } : current);
        setDirty(true);
    };
    const updateStorage = (patch: Partial<BackendRuntimeSettings['storage']>) => {
        setDraft(current => current ? { ...current, storage: { ...current.storage, ...patch } } : current);
        setDirty(true);
    };
    const updateGeoChat = (patch: Partial<BackendRuntimeSettings['geochat']>) => {
        setDraft(current => current ? { ...current, geochat: { ...current.geochat, ...patch } } : current);
        setDirty(true);
    };
    const save = async () => {
        if (!draft || !settings)
            return;
        setSaving(true);
        setError('');
        try {
            const updated = await updateRuntimeSettings({
                ingestion: draft.ingestion,
                processing: draft.processing,
                storage: draft.storage,
                geochat: draft.geochat,
                expected_version: settings.version,
            });
            setSettings(updated);
            setDraft(structuredClone(updated));
            setDirty(false);
            setToast(`Settings saved · version ${updated.version}`);
        }
        catch (err) {
            const message = err instanceof Error ? err.message : 'Could not save settings.';
            setError(message.includes('409') ? 'Settings were changed by another session. Reload the latest values before saving again.' : message);
        }
        finally {
            setSaving(false);
        }
    };
    const reset = () => { if (settings) {
        setDraft(structuredClone(settings));
        setDirty(false);
        setError('');
    } };
    return <section className="page system-page runtime-settings-page">
    <div className="page-title"><div><span className="eyebrow">SYSTEM · CONFIGURATION</span><h1>Settings</h1><p>Manage persisted ingestion, processing, storage, and Ask Geo settings. Database connection settings are configured separately at backend startup.</p></div><div className="title-actions theme-action-group"><button className="btn btn-secondary" onClick={() => void load()} disabled={loading || saving}><RefreshCw className={loading ? 'spin' : ''}/>{loading ? 'Loading…' : 'Reload settings'}</button><button className="btn btn-primary" onClick={() => void save()} disabled={!draft || !settings || !dirty || saving || !backendConnected}><Check />{saving ? 'Saving…' : 'Save settings'}</button></div></div>
    <div className="system-meta theme-meta"><span>Endpoint: <code>GET /api/v1/settings</code></span><span>Update: <code>PATCH /api/v1/settings</code></span><span>Configuration version: {settings?.version ?? '—'}</span><span>Storage: SQLite settings</span><span>GeoChat: <code>/api/v1/geochat</code></span></div>
    {error && <div className="system-error"><AlertTriangle /><div><strong>Settings issue</strong><p>{error}</p></div></div>}
    {loading && !draft ? <div className="page-empty">Loading settings from backend…</div> : !draft ? <div className="page-empty">Settings are unavailable. Confirm the backend is running and reload.</div> : <>
      <div className="system-section theme-panel"><div className="system-section-heading"><div><span className="eyebrow">INGESTION</span><h2>Upload validation and limits</h2></div><span>Persisted backend configuration</span></div>
        <div className="runtime-setting-grid">
          <label className="runtime-setting-field"><span>Maximum upload size (MB)</span><input type="number" min={1} max={65536} value={draft.ingestion.max_upload_size_mb} onChange={e => updateIngestion({ max_upload_size_mb: Number(e.target.value) })}/><small>Allowed range: 1–65,536 MB.</small></label>
          <label className="runtime-setting-toggle"><span><strong>Validate file extension</strong><small>Reject uploads whose extension is not enabled below.</small></span><input type="checkbox" checked={draft.ingestion.validate_file_format} onChange={e => updateIngestion({ validate_file_format: e.target.checked })}/></label>
        </div>
        <div className="runtime-format-heading theme-meta"><strong>Allowed upload formats</strong><small>Select one or more formats. The backend requires at least one.</small></div>
        <div className="runtime-format-grid">{formats.map(format => <label key={format} className={`theme-inset ${draft.ingestion.allowed_formats.includes(format) ? 'selected' : ''}`}><input type="checkbox" checked={draft.ingestion.allowed_formats.includes(format)} onChange={e => { const next = e.target.checked ? [...draft.ingestion.allowed_formats, format] : draft.ingestion.allowed_formats.filter(value => value !== format); updateIngestion({ allowed_formats: next }); }}/><span>{format}</span></label>)}</div>
        {!draft.ingestion.allowed_formats.length && <p className="runtime-validation">At least one format must be enabled before saving.</p>}
      </div>
      <div className="system-section theme-panel"><div className="system-section-heading"><div><span className="eyebrow">PROCESSING</span><h2>Execution and timeout</h2></div><span>Persisted backend configuration</span></div>
        <div className="runtime-setting-grid">
          <label className="runtime-setting-field runtime-storage-field"><span>Primary execution runner</span><input type="text" value={draft.processing.default_runner} onChange={e => updateProcessing({ default_runner: e.target.value })} spellCheck={false} required/><small>Attempted first. Default: <code>pyrunner.local.sh</code>.</small></label>
          <label className="runtime-setting-field runtime-storage-field"><span>Fallback execution runner</span><input type="text" value={draft.processing.fallback_runner} onChange={e => updateProcessing({ fallback_runner: e.target.value })} spellCheck={false} required/><small>Attempted if the primary runner cannot start or submit. Default: <code>pyrunner.sbatch.sh</code>.</small></label>
          <label className="runtime-setting-field runtime-timeout-field"><span>Timeout (seconds)</span><input type="number" min={1} max={86400} value={draft.processing.default_timeout_seconds} onChange={e => updateProcessing({ default_timeout_seconds: Number(e.target.value) })}/><small>Allowed range: 1–86,400 seconds.</small></label>
        </div>
        <div className="runtime-settings-note theme-inset"><strong>Runner execution order</strong><p>Runner files must exist in the backend scripts directory. The backend tries the primary runner first, then the fallback if startup or submission fails. If both fail, the job request returns an error with both failure reasons.</p></div>
      </div>
      <div className="system-section theme-panel"><div className="system-section-heading"><div><span className="eyebrow">STORAGE</span><h2>Data directories</h2></div><span>Absolute paths on the backend host</span></div>
        <div className="runtime-setting-grid">
          <label className="runtime-setting-field runtime-storage-field"><span>Primary data root</span><input type="text" value={draft.storage.data_root} onChange={e => updateStorage({ data_root: e.target.value })} spellCheck={false}/><small>Preferred root for new uploads, processing scripts, and GeoChat transcripts. If it cannot be created or written to, the backend tries the fallback root.</small></label>
          <label className="runtime-setting-field runtime-storage-field"><span>Fallback data root</span><input type="text" value={draft.storage.fallback_data_root} onChange={e => updateStorage({ fallback_data_root: e.target.value })} spellCheck={false}/><small>Secondary root used when the primary is unavailable or unwritable. Previously stored datasets are also searched across configured roots.</small></label>
        </div>
        <div className="runtime-settings-note theme-inset"><strong>Database location is separate</strong><p>The SQLite database must be located before the settings table can be read, so it is configured at startup rather than saved here. Default: <code>backend/data/geohackathon.db</code>. To use another database, set <code>DATABASE_URL</code> in <code>backend/.env</code> and restart the backend. Do not point it at a new path unless you intend to use or migrate that database.</p></div>
      </div>
      <div className="system-section theme-panel"><div className="system-section-heading"><div><span className="eyebrow">ASK GEO · LLM</span><h2>Chat model and generation</h2></div><span>Persisted backend configuration</span></div>
        <div className="runtime-setting-grid">
          <label className="runtime-setting-field runtime-storage-field"><span>OpenAI-compatible base URL</span><input type="url" value={draft.geochat.base_url} onChange={e => updateGeoChat({ base_url: e.target.value })} spellCheck={false} required/><small>Backend-side endpoint; the API key remains a server-side placeholder.</small></label>
          <label className="runtime-setting-field runtime-storage-field"><span>Model</span><input type="text" value={draft.geochat.model} onChange={e => updateGeoChat({ model: e.target.value })} maxLength={200} required/><small>Default: google/gemma-4-26B-A4B-it.</small></label>
          <label className="runtime-setting-field"><span>Temperature ({draft.geochat.temperature})</span><input type="number" min={0} max={2} step={0.1} value={draft.geochat.temperature} onChange={e => updateGeoChat({ temperature: Number(e.target.value) })}/><small>0 is more deterministic; higher values increase variation. Range: 0–2.</small></label>
          <label className="runtime-setting-field"><span>Maximum output tokens</span><input type="number" min={1} max={32768} value={draft.geochat.max_tokens} onChange={e => updateGeoChat({ max_tokens: Number(e.target.value) })}/><small>Range: 1–32,768 tokens.</small></label>
          <label className="runtime-setting-field"><span>Top-p ({draft.geochat.top_p})</span><input type="number" min={0.01} max={1} step={0.05} value={draft.geochat.top_p} onChange={e => updateGeoChat({ top_p: Number(e.target.value) })}/><small>Nucleus sampling; normally adjust temperature or top-p, not both.</small></label>
          <label className="runtime-setting-field"><span>Request timeout (seconds)</span><input type="number" min={5} max={600} value={draft.geochat.timeout_seconds} onChange={e => updateGeoChat({ timeout_seconds: Number(e.target.value) })}/><small>Range: 5–600 seconds.</small></label>
          <label className="runtime-setting-field"><span>Frequency penalty ({draft.geochat.frequency_penalty})</span><input type="number" min={-2} max={2} step={0.1} value={draft.geochat.frequency_penalty} onChange={e => updateGeoChat({ frequency_penalty: Number(e.target.value) })}/><small>Range: −2 to 2; support depends on the served model.</small></label>
          <label className="runtime-setting-field"><span>Presence penalty ({draft.geochat.presence_penalty})</span><input type="number" min={-2} max={2} step={0.1} value={draft.geochat.presence_penalty} onChange={e => updateGeoChat({ presence_penalty: Number(e.target.value) })}/><small>Range: −2 to 2; support depends on the served model.</small></label>
        </div>
        <label className="runtime-setting-field runtime-storage-field"><span>System role / prompt</span><textarea rows={5} value={draft.geochat.system_prompt} onChange={e => updateGeoChat({ system_prompt: e.target.value })} maxLength={12000} required/><small>{draft.geochat.system_prompt.length}/12,000 characters. This is sent as the system message for each request.</small></label>
      </div>
      <div className="runtime-settings-footer"><span>{dirty ? 'Unsaved changes' : 'All changes saved'} · Version {settings?.version ?? "—"}</span><div><button className="btn btn-secondary" onClick={reset} disabled={!dirty || saving}>Discard changes</button><button className="btn btn-primary" onClick={() => void save()} disabled={!dirty || saving || !backendConnected || !draft.ingestion.allowed_formats.length}><Check />{saving ? 'Saving…' : 'Save settings'}</button></div></div>
    </>}
  </section>;
}

import { useEffect, useMemo, useState, type ReactElement } from 'react';
import { Activity, ChevronUp, Eye, File, FileSpreadsheet, Folder, Pencil, RefreshCw, Trash2, Waves } from 'lucide-react';
import { browseLinkLocalDirectory, type LinkLocalBrowserEntry, type LinkLocalBrowserResult } from '../../api/backend';
import { canOpenDatasetViewer, getWellTableKind, isTextTableDataset, type DataKind } from '../data';
import DialogFrame from '../DialogFrame';
import type { TabContext } from './TabContext';
import { MiniSeismic, TablePreview, WellTableApiViewer, WellTrack } from './ViewerComponents';

const kindIcon = (kind: DataKind) => kind === 'Seismic' ? <Waves /> : kind === 'Well log' ? <Activity /> : <FileSpreadsheet />;

/** Tab UI extracted from App; shared application state/actions arrive via context. */
type Props = { context: TabContext };

export default function InputDataTab({ context }: Props): ReactElement {
  const [typeFilter, setTypeFilter] = useState('');
  const [formatFilter, setFormatFilter] = useState('');
  const [surveyFilter, setSurveyFilter] = useState('');
  const [linkDialogOpen, setLinkDialogOpen] = useState(false);
  const [linkPath, setLinkPath] = useState('');
  const [linkKind, setLinkKind] = useState<'seismic' | 'well_log' | 'deviation' | 'checkshot'>('seismic');
  const [linkName, setLinkName] = useState('');
  const [linkBusy, setLinkBusy] = useState(false);
  const [linkError, setLinkError] = useState('');
  const [browser, setBrowser] = useState<LinkLocalBrowserResult | null>(null);
  const [browserBusy, setBrowserBusy] = useState(false);
  const [browserError, setBrowserError] = useState('');
  const { AlertTriangle, Check, Database, Play, Plus, UploadCloud, uploadDialogOpen, setUploadDialogOpen, cancelUploadDialog, active, activeSeismic, activeTable, activeWell, addFiles, backendStatus, dragging, linkLocalDataset, fileRef, folderRef, handleAssignDatasetSurvey, handleAssignDatasetWell, handleDeleteDataset, handleEditDataset, items, loadDatasetPreview, loadingInventory, projectId, selected, setDragging, setSelected, setView, surveyBusy, surveyNameForDataset, surveys, totals, wells } = context;
  const typeOptions = useMemo(() => [...new Set(items.map(item => item.kind))].sort(), [items]);
  const formatOptions = useMemo(() => [...new Set(items.map(item => item.format).filter(Boolean))].sort((a, b) => a.localeCompare(b)), [items]);
  const surveyOptions = useMemo(() => [...new Set(items.map(item => surveyNameForDataset(item)).filter(Boolean))].sort((a, b) => a.localeCompare(b)), [items, surveyNameForDataset]);
  const filteredItems = useMemo(() => items.filter(item =>
    (!typeFilter || item.kind === typeFilter) &&
    (!formatFilter || item.format === formatFilter) &&
    (!surveyFilter || surveyNameForDataset(item) === surveyFilter)
  ), [items, typeFilter, formatFilter, surveyFilter, surveyNameForDataset]);
  const activeWellTableKind = active ? getWellTableKind(active) : null;
  const loadBrowserDirectory = async (path?: string) => {
    setBrowserBusy(true);
    setBrowserError('');
    try { setBrowser(await browseLinkLocalDirectory(path)); }
    catch (error) { setBrowserError(error instanceof Error ? error.message : 'Unable to browse server directories.'); }
    finally { setBrowserBusy(false); }
  };
  useEffect(() => {
    if (linkDialogOpen) void loadBrowserDirectory();
    else { setBrowser(null); setBrowserError(''); }
  }, [linkDialogOpen]);
  return (
    <section className="page input-page">
        <div className="page-title"><div><span className="eyebrow">DATA FOUNDATION</span><h1>Bring your subsurface data.</h1><p>Link existing files on the backend filesystem by default. Upload only when the source is not already available to the server.</p></div><div className="title-actions theme-action-group"><button className="btn btn-primary" onClick={() => { setLinkError(''); setLinkDialogOpen(true); }} disabled={backendStatus !== 'Connected' || !projectId}><Database />Link local</button><button className="btn btn-secondary" onClick={() => fileRef.current?.click()}><UploadCloud />Upload files</button><button className="btn btn-secondary" onClick={() => folderRef.current?.click()}><Plus />Upload folder</button><input className="hidden-input" ref={folderRef} type="file" multiple {...({ webkitdirectory: '' } as object)} onChange={event => { if (event.currentTarget.files) void addFiles(event.currentTarget.files); event.currentTarget.value = ''; }}/></div></div>

        <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }} onClick={() => fileRef.current?.click()}>
          <input ref={fileRef} type="file" multiple accept=".sgy,.segy,.las,.txt,.csv,.tsv,.xlsx,.xlsm" onChange={e => { if (e.target.files) void addFiles(e.target.files); e.currentTarget.value = ''; setUploadDialogOpen(false); }}/>
          <div className="upload-icon"><UploadCloud /></div><div><strong>Upload files from this computer</strong><span>Use Link local for files already on the backend filesystem</span></div>
          <div className="formats"><span><Waves /> SEG-Y</span><span><Activity /> LAS</span><span><FileSpreadsheet /> CSV / TSV / TXT</span></div>
          <small>Uploads copy files into project storage. For large files already on the server, use Link local instead.</small>
        </div>

        <div className="inventory-head"><div><h2>Data inventory</h2><span>{loadingInventory ? 'Loading backend inventory…' : `${items.length} file${items.length === 1 ? '' : 's'} · ${backendStatus === 'Connected' ? 'backend inventory' : 'current session'}`}</span></div><div className="summary-pills"><span><i className="teal"/>{totals.Seismic} Seismic</span><span><i className="amber"/>{totals.Wells} Well log</span><span><i className="violet"/>{totals.Tables} Table</span></div></div>

        <div className="inventory-layout"><div className="inventory-table theme-panel"><div className="table-row table-labels"><span>DATASET</span><label className="inventory-filter"><span>TYPE</span><select aria-label="Filter by type" value={typeFilter} onChange={event => setTypeFilter(event.target.value)}><option value="">All</option>{typeOptions.map(value => <option key={value} value={value}>{value}</option>)}</select></label><label className="inventory-filter"><span>FORMAT</span><select aria-label="Filter by format" value={formatFilter} onChange={event => setFormatFilter(event.target.value)}><option value="">All</option>{formatOptions.map(value => <option key={value} value={value}>{value}</option>)}</select></label><label className="inventory-filter"><span>SURVEY/WELL</span><select aria-label="Filter by survey or well" value={surveyFilter} onChange={event => setSurveyFilter(event.target.value)}><option value="">All</option>{surveyOptions.map(value => <option key={value} value={value}>{value}</option>)}</select></label><span>SIZE</span><span>ACTIONS</span></div>{items.length === 0 && <div className="inventory-empty">No datasets selected yet. Add files above to begin.</div>}{items.length > 0 && filteredItems.length === 0 && <div className="inventory-empty">No datasets match the selected filters. <button type="button" className="inventory-clear-filters" onClick={() => { setTypeFilter(''); setFormatFilter(''); setSurveyFilter(''); }}>Clear filters</button></div>}{filteredItems.map(item => <div key={item.id} className={`table-row inventory-data-row ${selected === item.id ? 'selected' : ''}`}><button type="button" className="inventory-row-select" onDoubleClick={() => canOpenDatasetViewer(item) && setView(item.kind === 'Seismic' ? 'Seismic Viewer' : 'Well Viewer')} onClick={() => { setSelected(item.id); void loadDatasetPreview(item); }}><span className="dataset"><i className={item.kind.toLowerCase().replace(' ', '-')}>{kindIcon(item.kind)}</i><span><strong>{item.name}</strong><small>{item.details}</small></span></span><span>{item.kind}</span><span><em>{item.format}</em></span><span className="inventory-survey">{surveyNameForDataset(item)}</span><span>{item.size}</span></button><span className="inventory-row-actions theme-action-group"><button type="button" className="btn btn-secondary inventory-action view" onClick={() => { setSelected(item.id); void loadDatasetPreview(item); setView(item.kind === 'Seismic' ? 'Seismic Viewer' : 'Well Viewer'); }} disabled={!canOpenDatasetViewer(item)} title="View dataset" aria-label="View dataset"><Eye /></button><button type="button" className="btn btn-secondary inventory-action edit" onClick={() => { setSelected(item.id); void handleEditDataset(item); }} disabled={backendStatus !== 'Connected' || surveyBusy} title="Edit dataset name and survey" aria-label="Edit dataset"><Pencil /></button><button type="button" className="btn btn-danger inventory-action delete" onClick={() => void handleDeleteDataset(item)} disabled={backendStatus !== 'Connected' || surveyBusy} title={item.storageMode === 'linked' ? "Remove dataset link (source file is retained)" : "Delete dataset and stored file"} aria-label="Delete dataset"><Trash2 /></button></span></div>)}</div>
          <aside className="preview-card theme-panel"><div className="preview-title"><div><span>FILE PREVIEW</span><strong>{active?.name ?? 'No dataset selected'}</strong></div></div>{!active ? <div className="visual-empty">Select or add a dataset to inspect its preview.</div> : active.status === 'Scanning' ? <div className="scanning-card"><i /><strong>Reading file locally…</strong><span>Headers → geometry → preview</span></div> : activeWellTableKind && active && active.id && !active.id.startsWith('local-') ? <WellTableApiViewer key={active.id} datasetId={active.id} kind={activeWellTableKind} datasetName={active.name} compact/> : active.issue ? <div className="parse-error"><AlertTriangle /><strong>Preview unavailable</strong><span>{active.issue}</span></div> : active.kind === 'Seismic' ? <MiniSeismic data={activeSeismic}/> : active.kind === 'Well log' ? <WellTrack data={activeWell}/> : <TablePreview data={activeTable}/>}<div className="preview-meta theme-inset"><div><span>FORMAT</span><strong>{active?.format ?? '—'}</strong></div><div><span>HEALTH</span><strong className={active?.status === 'Ready' ? 'healthy' : 'warn'}>{!active ? 'No dataset' : active.status === 'Scanning' ? 'Reading…' : active.status === 'Ready' ? 'Parsed · Ready' : 'Needs review'}</strong></div></div>{active && <><div className="qc-note theme-inset">{active.status === 'Ready' ? <Check /> : <AlertTriangle />}<span><strong>{active.status === 'Ready' ? 'Local parsing complete' : active.status === 'Scanning' ? 'Parsing in progress' : 'File needs review'}</strong><small>{active.status === 'Ready' ? (active.details.includes('saved to backend') || active.details.includes('loaded from backend') || active.details.includes('Linked local') ? (active.storageMode === 'linked' ? 'Linked to the original server file; no copy was made.' : 'Stored in the Python backend; preview parsed from the original file.') : 'Preview and metadata come from the selected file.') : active.issue ?? 'Reading headers and samples.'}</small></span></div>{active.kind === 'Seismic' && active.id && <div className="dataset-survey-assignment theme-inset"><label htmlFor="dataset-survey-assignment">SURVEY ASSOCIATION</label><select id="dataset-survey-assignment" value={active.surveyId ?? ''} onChange={event => { const surveyId = event.target.value; void handleAssignDatasetSurvey(active, surveyId); }} disabled={backendStatus !== 'Connected' || surveyBusy || !projectId}><option value="">Unassigned — no survey</option>{surveys.map(survey => <option key={survey.id} value={survey.id}>{survey.name}</option>)}</select><small>Changes are saved to the backend immediately.</small></div>}{(active.kind === 'Well log' || isTextTableDataset(active)) && active.id && <div className="dataset-survey-assignment theme-inset"><label htmlFor="dataset-well-assignment">WELL ASSOCIATION</label><select id="dataset-well-assignment" value={active.wellId ?? ''} onChange={event => { void handleAssignDatasetWell(active, event.target.value); }} disabled={backendStatus !== 'Connected' || surveyBusy || !projectId}><option value="">Unassigned — no well</option>{wells.map(well => <option key={well.id} value={well.id}>{well.name}</option>)}</select><small>Link this well-log file to a well record. Changes are saved immediately.</small></div>}<button className="btn btn-secondary open-viewer" disabled={!canOpenDatasetViewer(active)} onClick={() => setView(active.kind === 'Seismic' ? 'Seismic Viewer' : 'Well Viewer')}>{active.kind === 'Table' && !isTextTableDataset(active) ? 'Table preview shown' : active.kind === 'Seismic' ? 'Open Seismic Viewer' : 'Open Well Viewer'} <Play /></button></>}</aside>
        </div>
        {linkDialogOpen && <DialogFrame
          eyebrow="INPUT DATA"
          title="Link an existing server file"
          description="Register a file already available to the backend. The file is not copied or moved. The path must be inside a configured input_data directory."
          labelledBy="link-local-dialog-title"
          onClose={() => { if (!linkBusy) setLinkDialogOpen(false); }}
          closeDisabled={linkBusy}
          panelClassName="upload-dialog"
        >
          <div className="form-field link-local-picker">
            <label htmlFor="link-local-path">SELECT SERVER FILE</label>
            <div className="link-path-input"><input id="link-local-path" value={linkPath} onChange={event => setLinkPath(event.target.value)} placeholder="Choose a file below or enter its absolute path" autoFocus/><button type="button" className="btn btn-secondary" onClick={() => void loadBrowserDirectory(browser?.current_path ?? undefined)} disabled={browserBusy} title="Refresh directory"><RefreshCw size={16}/></button></div>
            <small>The browser shows directories permitted by the backend's DATASET_ALLOWED_ROOTS setting.</small>
          </div>
          <div className="server-file-browser">
            <div className="server-file-toolbar">
              <span title={browser?.current_path ?? 'Configured roots'}>{browser?.current_path ?? 'Server input locations'}</span>
              <div className="server-file-nav-actions"><button type="button" className="btn btn-secondary" onClick={() => void loadBrowserDirectory()} disabled={browserBusy || !browser?.current_path} title="Choose another input location">Locations</button><button type="button" className="btn btn-secondary" onClick={() => void loadBrowserDirectory(browser?.parent_path ?? undefined)} disabled={browserBusy || !browser?.parent_path} title="Go to parent directory"><ChevronUp size={16}/> Up</button></div>
            </div>
            {browserBusy ? <div className="server-file-empty">Loading server directories…</div> : browserError ? <div className="server-file-error">{browserError}</div> : !browser?.current_path ? <div className="server-file-list">{(browser?.roots ?? []).map(root => <button type="button" className="server-file-entry" key={root.path} onClick={() => void loadBrowserDirectory(root.path)}><Folder size={17}/><span>{root.name}</span></button>)}{browser?.roots.length === 0 && <div className="server-file-empty">No configured input directories are available.</div>}</div> : <div className="server-file-list">{(browser?.entries ?? []).map((entry: LinkLocalBrowserEntry) => entry.kind === 'directory' ? <button type="button" className="server-file-entry" key={entry.path} onClick={() => void loadBrowserDirectory(entry.path)}><Folder size={17}/><span>{entry.name}</span></button> : <button type="button" className={`server-file-entry server-file-select ${linkPath === entry.path ? 'selected' : ''}`} key={entry.path} onClick={() => { setLinkPath(entry.path); if (entry.extension === '.sgy' || entry.extension === '.segy') setLinkKind('seismic'); else if (entry.extension === '.las') setLinkKind('well_log'); }}><File size={17}/><span>{entry.name}</span><small>{entry.extension}</small></button>)}{browser?.entries.length === 0 && <div className="server-file-empty">No supported files or subdirectories in this folder.</div>}</div>}
          </div>
          <div className="form-field"><label htmlFor="link-local-kind">DATASET TYPE</label><select id="link-local-kind" value={linkKind} onChange={event => setLinkKind(event.target.value as typeof linkKind)}><option value="seismic">Seismic / inversion (SEG-Y)</option><option value="well_log">Well log (LAS)</option><option value="deviation">Well deviation (TXT)</option><option value="checkshot">Checkshot (TXT)</option></select></div>
          <div className="form-field"><label htmlFor="link-local-name">DISPLAY NAME (OPTIONAL)</label><input id="link-local-name" value={linkName} onChange={event => setLinkName(event.target.value)} placeholder="Use source filename"/></div>
          {linkError && <div className="parse-error"><span>{linkError}</span></div>}
          <div className="project-dialog-actions theme-action-group"><button type="button" className="btn btn-secondary" disabled={linkBusy} onClick={() => setLinkDialogOpen(false)}>Cancel</button><button type="button" className="btn btn-primary" disabled={linkBusy || !linkPath.trim()} onClick={async () => { setLinkBusy(true); setLinkError(''); try { await linkLocalDataset(linkPath, linkKind, linkName); setLinkDialogOpen(false); setLinkName(''); } catch (error) { setLinkError(error instanceof Error ? error.message : 'Could not link dataset.'); } finally { setLinkBusy(false); } }}>{linkBusy ? 'Validating…' : 'Link dataset'}</button></div>
        </DialogFrame>}
        {uploadDialogOpen && <DialogFrame
          eyebrow="INPUT DATA"
          title="Upload project data"
          description="Select one or more geological files to add to the current project."
          labelledBy="upload-dialog-title"
          onClose={cancelUploadDialog}
          panelClassName="upload-dialog"
        >
          <div className="upload-dialog-formats"><strong>Supported formats</strong><span>SEG-Y · LAS · TXT · CSV · TSV · XLSX</span><small>SEG-Y requires a survey association. TXT files require a dataset type selection.</small></div>
          <div className="project-dialog-actions theme-action-group"><button type="button" className="btn btn-secondary" onClick={cancelUploadDialog}>Cancel</button><button type="button" className="btn btn-primary" onClick={() => fileRef.current?.click()}><UploadCloud/> Choose files</button></div>
        </DialogFrame>}
      </section>
  );
}

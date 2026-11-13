import { useEffect, useMemo, useState } from 'react';
import { explainEvidenceItem, listEvidenceCatalog, listWell3DWells, getWell3DTable, well3DAssetUrl, type Well3DEntry, type Well3DTablePreview, type EvidenceCatalogItem, type EvidenceExplanation } from '../../api/backend';
import {
  Activity, Check, CircleHelp, Database, FileSearch, Filter, MessageSquare, Search, X,
} from 'lucide-react';
import MarkdownMessage from './MarkdownMessage';

type EvidenceSection = 'Well 3D' | 'Process Ledger';

type EvidenceVisualVariant = 'seismic' | 'model' | 'well' | 'default';

function getEvidenceVisualVariant(script?: string | null, operation?: string | null): EvidenceVisualVariant {
  const source = `${script ?? ''} ${operation ?? ''}`.toLowerCase();
  if (/seismic|sgy|segy|stack|migration/.test(source)) return 'seismic';
  if (/model|inference|predict|classification/.test(source)) return 'model';
  if (/well|log|las|petrophys/.test(source)) return 'well';
  return 'default';
}

const sections = [{ id: 'Well 3D' as EvidenceSection, label: 'Well 3D', icon: Activity }, { id: 'Process Ledger' as EvidenceSection, label: 'Process Ledger', icon: Database }];

/** The question that makes Ask Geo read one run with its process skill, e.g. "Process D job <id>". */
function askGeoQuestion(explanation: EvidenceExplanation): string {
  const letter = explanation.python_script?.match(/^process_([A-Za-z])_/)?.[1]?.toUpperCase();
  return letter
    ? `Explain the results of Process ${letter} job ${explanation.job_id}`
    : `Explain the results of ${explanation.python_script ?? 'processing'} job ${explanation.job_id}`;
}

export default function EvidenceTab({ onAskGeo }: { onAskGeo?: (question: string) => void }) {
  const [section, setSection] = useState<EvidenceSection>('Well 3D');
  const [search, setSearch] = useState('');
  const [kind, setKind] = useState('All types');
  const [evidence, setEvidence] = useState<EvidenceCatalogItem[]>([]);
  const [loadingEvidence, setLoadingEvidence] = useState(true);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [explainingJob, setExplainingJob] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<EvidenceExplanation | null>(null);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [well3d, setWell3d] = useState<Well3DEntry[]>([]);
  const [selectedWell, setSelectedWell] = useState('');
  const [selectedImage, setSelectedImage] = useState('');
  const [selectedTable, setSelectedTable] = useState('');
  const [tablePreview, setTablePreview] = useState<Well3DTablePreview | null>(null);
  const [well3dError, setWell3dError] = useState('');
  const [well3dLoading, setWell3dLoading] = useState(true);

  useEffect(() => {
    let active = true;
    listEvidenceCatalog()
      .then(result => { if (active) { setEvidence(result.items); setEvidenceError(null); } })
      .catch(error => { if (active) setEvidenceError(error instanceof Error ? error.message : 'Unable to scan processing results.'); })
      .finally(() => { if (active) setLoadingEvidence(false); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    listWell3DWells().then(result => { if (active) { setWell3d(result.items); setSelectedWell(current => current && result.items.some(item => item.well === current) ? current : result.items[0]?.well ?? ''); setWell3dError(''); } })
      .catch(error => { if (active) setWell3dError(error instanceof Error ? error.message : 'Unable to load Well 3D folders.'); })
      .finally(() => { if (active) setWell3dLoading(false); });
    return () => { active = false; };
  }, []);

  const activeWell = well3d.find(item => item.well === selectedWell);
  useEffect(() => {
    setSelectedImage(activeWell?.images[0] ?? '');
    setSelectedTable(activeWell?.tables[0] ?? '');
    setTablePreview(null);
  }, [selectedWell, activeWell?.images.join('|'), activeWell?.tables.join('|')]);
  useEffect(() => {
    if (!selectedWell || !selectedTable) { setTablePreview(null); return; }
    let active = true;
    getWell3DTable(selectedWell, selectedTable).then(result => { if (active) setTablePreview(result); }).catch(error => { if (active) setWell3dError(error instanceof Error ? error.message : 'Unable to preview table.'); });
    return () => { active = false; };
  }, [selectedWell, selectedTable]);

  const filteredEvidence = useMemo(() => evidence.filter(item => {
    const text = `${item.job_id} ${item.operation ?? ''} ${item.python_script ?? ''} ${item.target_well ?? ''} ${item.status} ${item.artifacts.map(file => file.name).join(' ')}`.toLowerCase();
    const matchesSearch = text.includes(search.toLowerCase());
    const matchesKind = kind === 'All types' || (kind === 'Well results' ? Boolean(item.target_well) : kind === 'Model outputs' ? /model|prediction|inference/i.test(text) : /seismic|sgy|segy/i.test(text));
    return matchesSearch && matchesKind;
  }), [evidence, search, kind]);

  const evidenceGroups = useMemo(() => {
    const groups = new Map<string, EvidenceCatalogItem[]>();
    for (const item of filteredEvidence) {
      const key = item.python_script || item.operation || 'Other processing runs';
      groups.set(key, [...(groups.get(key) ?? []), item]);
    }
    return [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [filteredEvidence]);

  const explainItem = async (jobId: string) => {
    setExplainingJob(jobId);
    setExplanation(null);
    setExplainError(null);
    try { setExplanation(await explainEvidenceItem(jobId)); }
    catch (error) { setExplainError(error instanceof Error ? error.message : 'Unable to summarize this processing output folder.'); }
    finally { setExplainingJob(null); }
  };

  return (
    <section className="page evidence-page">
      <div className="page-title evidence-page-heading">
        <div>
          <span className="eyebrow">PROCESSING-DERIVED RESULTS</span>
          <h1>Evidence</h1>
          <p>Inspect registered well geometry and browse processing outputs.</p>
        </div>
        <div className="evidence-heading-meta"><span><Database size={15} /> Evidence workspace</span></div>
      </div>

      <nav className="evidence-nav" aria-label="Evidence workspace sections">
        {sections.map(item => {
          const Icon = item.icon;
          return <button key={item.id} className={`evidence-nav-item ${section === item.id ? 'active' : ''}`} onClick={() => setSection(item.id)}>
            <Icon size={16} /><span>{item.label}</span>
          </button>;
        })}
      </nav>

      {section === 'Well 3D' && <div className="evidence-workspace-panel well3d-workspace">
        <div className="evidence-panel-heading"><div><span className="evidence-section-kicker">SITE DATA / WELL_3D</span><h2>Well 3D</h2><p>Embedded 3D interpretation, supporting figures and along-well data.</p></div><Activity size={23}/></div>
        {well3dLoading ? <div className="evidence-empty-state"><Activity size={22}/><h3>Discovering well folders…</h3></div> : well3dError && !well3d.length ? <div className="evidence-empty-state"><CircleHelp size={22}/><h3>Unable to load Well 3D</h3><p>{well3dError}</p></div> : !well3d.length ? <div className="evidence-empty-state"><Database size={22}/><h3>No Well 3D folders found</h3><p>Place well folders such as <code>eo-1</code> under the configured <code>site_data/well_3d/</code> directory.</p></div> : <>
          <div className="well3d-selector"><label>SELECT WELL <select value={selectedWell} onChange={event => setSelectedWell(event.target.value)}>{well3d.map(item => <option key={item.well} value={item.well}>{item.well}</option>)}</select></label><span>{activeWell?.file_count ?? 0} files</span></div>
          {activeWell?.html[0] ? <section className="well3d-html-section"><div className="well3d-section-heading"><div><h3>Interactive 3D interpretation</h3><span>{activeWell.html[0]}</span></div></div><iframe title={`${selectedWell} 3D interpretation`} src={well3DAssetUrl(selectedWell, activeWell.html[0])} loading="lazy" sandbox="allow-scripts" referrerPolicy="no-referrer" /></section> : <div className="evidence-empty-state"><h3>No HTML 3D viewer found</h3><p>This well folder has no .html or .htm visualisation.</p></div>}
          <section className="well3d-images-section"><div className="well3d-section-heading"><div><h3>Image viewer</h3><span>{activeWell?.images.length ?? 0} images</span></div></div>{activeWell?.images.length ? <><div className="well3d-image-main"><img src={well3DAssetUrl(selectedWell, selectedImage || activeWell.images[0])} alt={selectedImage}/><span>{selectedImage || activeWell.images[0]}</span></div><div className="well3d-image-strip">{activeWell.images.map(file => <button key={file} className={selectedImage === file ? 'active' : ''} onClick={() => setSelectedImage(file)}><img src={well3DAssetUrl(selectedWell, file)} alt=""/><span>{file}</span></button>)}</div></> : <p className="well3d-muted">No supported images found.</p>}</section>
          <section className="well3d-table-section"><div className="well3d-section-heading"><div><h3>Spreadsheet viewer</h3><span>{activeWell?.tables.length ?? 0} tables</span></div>{!!activeWell?.tables.length && <select value={selectedTable} onChange={event => setSelectedTable(event.target.value)}>{activeWell.tables.map(file => <option key={file} value={file}>{file}</option>)}</select>}</div>{tablePreview ? <><div className="well3d-table-meta">{tablePreview.total_rows.toLocaleString()} rows · {tablePreview.columns.length} columns</div><div className="well3d-table-scroll"><table><thead><tr>{tablePreview.columns.map(column => <th key={column}>{column}</th>)}</tr></thead><tbody>{tablePreview.rows.map((row, index) => <tr key={index}>{row.map((cell, i) => <td key={i}>{cell}</td>)}</tr>)}</tbody></table></div></> : <p className="well3d-muted">{activeWell?.tables.length ? 'Loading table preview…' : 'No CSV or TSV tables found.'}</p>}</section>
          {activeWell?.others.length ? <details className="well3d-other-files"><summary>Other supporting files ({activeWell.others.length})</summary>{activeWell.others.map(file => <a key={file} href={well3DAssetUrl(selectedWell, file)} target="_blank" rel="noreferrer">{file}</a>)}</details> : null}
        </>}
      </div>}
      {section === 'Process Ledger' && <>
        <div className="evidence-ledger-intro">
          <div><div className="evidence-section-kicker">01 / CATALOGUE</div><h2>Process Ledger</h2><p>A searchable catalogue of results generated by project processing workflows.</p></div>
          <div className="evidence-ledger-count"><strong>{evidence.length}</strong><span>discovered processing jobs</span></div>
        </div>
        <div className="evidence-ledger-toolbar">
          <label className="evidence-search"><Search size={16}/><input value={search} onChange={event => setSearch(event.target.value)} placeholder="Search evidence, well, workflow or model…" /></label>
          <label className="evidence-filter"><Filter size={15}/><select value={kind} onChange={event => setKind(event.target.value)}><option>All types</option><option>Well results</option><option>Seismic results</option><option>Model outputs</option></select></label>
          <button className="btn btn-secondary" disabled><Filter size={15}/> More filters</button>
        </div>
        {loadingEvidence ? <div className="evidence-empty-state"><Activity size={24}/><h3>Scanning processing results…</h3><p>Looking for existing job folders and result files.</p></div> : evidenceError ? <div className="evidence-empty-state"><CircleHelp size={24}/><h3>Could not scan processing results</h3><p>{evidenceError}</p><button className="btn btn-secondary" onClick={() => { setLoadingEvidence(true); setEvidenceError(null); listEvidenceCatalog().then(result => setEvidence(result.items)).catch(error => setEvidenceError(error instanceof Error ? error.message : 'Scan failed.')).finally(() => setLoadingEvidence(false)); }}>Retry scan</button></div> : filteredEvidence.length === 0 ? <div className="evidence-empty-state"><div className="evidence-empty-icon"><Database size={25}/></div><h3>{evidence.length ? 'No matching processing results' : 'No processing results found'}</h3><p>{evidence.length ? 'Adjust the search or type filter to see other results.' : 'The configured site_data/processing_jobs folders were scanned, but no job folders containing result metadata or output files were found.'}</p><div className="evidence-empty-note"><CircleHelp size={15}/><span>The Process Ledger discovers existing folders directly, including older jobs that may not have a database registry entry. It does not move or modify processing files.</span></div></div> : <div className="evidence-script-groups">{evidenceGroups.map(([script, runs]) => <section className="evidence-script-group" key={script}><div className="evidence-script-group-heading"><div><span className="evidence-section-kicker">PROCESSING SCRIPT</span><h3>{script}</h3></div><span>{runs.length} run{runs.length === 1 ? '' : 's'}</span></div><div className="evidence-catalog-list">{runs.map(item => <article className="evidence-catalog-item" key={item.id}><div className={`evidence-card-visual evidence-card-visual--${getEvidenceVisualVariant(item.python_script, item.operation)}`} aria-hidden="true">
                <img className="evidence-card-header-svg" src={`/evidence-headers/${getEvidenceVisualVariant(item.python_script, item.operation)}.svg`} alt="" />
              </div><div className="evidence-catalog-main"><div className="evidence-catalog-title"><strong>{item.target_well ? `Well ${item.target_well}` : item.operation || 'Processing run'}</strong><span className={`evidence-status evidence-status-${item.status.toLowerCase()}`}>{item.status}</span></div><div className="evidence-catalog-meta"><span>Job {item.job_id}</span><span>{item.artifact_count} output files</span>{item.created_at && <span>{item.created_at}</span>}</div>{item.summary && <p>{item.summary}</p>}<div className="evidence-artifact-list">{item.artifacts.filter(file => file.role !== 'metadata').map(file => <span key={file.name}><FileSearch size={13}/>{file.name}</span>)}</div></div><div className="evidence-item-actions"><button className="btn btn-secondary evidence-explain-button" onClick={() => explainItem(item.job_id)} disabled={explainingJob === item.job_id}><CircleHelp size={14}/>{explainingJob === item.job_id ? 'Analysing folder…' : 'What is this?'}</button></div></article>)}</div></section>)}</div>}
      </>}

      {(explanation || explainError) && <div className="evidence-modal-backdrop" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) { setExplanation(null); setExplainError(null); } }}>
        <section className="evidence-explain-dialog" role="dialog" aria-modal="true" aria-labelledby="evidence-explain-title">
          <button className="evidence-dialog-close" onClick={() => { setExplanation(null); setExplainError(null); }} aria-label="Close"><X size={17}/></button>
          <span className="evidence-section-kicker">{explanation?.markdown ? 'EXPERT RESULT ANALYSIS' : 'OUTPUT FOLDER ANALYSIS'}</span>
          <h2 id="evidence-explain-title">What is this?</h2>
          {explainError ? (
            <p className="evidence-compare-error">{explainError}</p>
          ) : explanation?.markdown ? (
            <>
              {/* A registered process skill read this run: show its fixed report, as in Ask Geo. */}
              <div className="evidence-explain-report"><MarkdownMessage content={explanation.markdown}/></div>
              {!explanation.llm_explanation && <p className="evidence-inline-note">The plain-language reading is unavailable because the language model did not respond. The verdict, tables and findings above are read directly from the run's files.</p>}
              {onAskGeo && <div className="evidence-explain-actions">
                <button className="btn btn-secondary" onClick={() => { onAskGeo(askGeoQuestion(explanation)); setExplanation(null); }}><MessageSquare size={14}/> Ask Geo about this run</button>
              </div>}
              <details className="evidence-explain-technical">
                <summary>Evidence files read ({explanation.artifacts.length})</summary>
                <ul>{explanation.artifacts.map(file => <li key={file.name}><code>{file.name}</code> · {(file.size_bytes / 1024).toFixed(1)} KB</li>)}</ul>
              </details>
            </>
          ) : explanation ? (
            <>
              {explanation.llm_explanation ? (
                <section className="evidence-explain-ai">
                  <span className="evidence-explain-ai-label">AI-GENERATED EXPLANATION</span>
                  <p>{explanation.llm_explanation}</p>
                </section>
              ) : (
                <section className="evidence-explain-ai evidence-explain-ai-fallback">
                  <span className="evidence-explain-ai-label">DESCRIPTION</span>
                  <p>{explanation.summary}</p>
                  <small>AI explanation is currently unavailable. The factual inventory below is still available.</small>
                </section>
              )}
              <details className="evidence-explain-technical">
                <summary>Technical inspection summary</summary>
                <p className="evidence-explain-summary">{explanation.summary}</p>
              </details>
              <div className="evidence-dialog-meta">
                <span>Script: <strong>{explanation.python_script || explanation.operation || 'Unknown'}</strong></span>
                <span>Job: <strong>{explanation.job_id}</strong></span>
                {explanation.target_well && <span>Well: <strong>{explanation.target_well}</strong></span>}
                <span>Status: <strong>{explanation.status}</strong></span>
              </div>
              <div className="evidence-explain-findings">
                <h3>Folder findings</h3>
                {explanation.artifacts.map(file => (
                  <article key={file.name}>
                    <strong>{file.name}</strong>
                    <span>
                      {file.format} · {(file.size_bytes / 1024).toFixed(1)} KB
                      {file.rows_scanned !== undefined ? ` · ${file.rows_scanned} rows` : ''}
                      {file.columns ? ` · ${file.columns.length} columns` : ''}
                    </span>
                    {file.columns && <p>Fields: {file.columns.join(', ')}</p>}
                    {file.numeric_fields && Object.entries(file.numeric_fields).length > 0 && (
                      <div className="evidence-explain-stats">
                        {Object.entries(file.numeric_fields).map(([name, stats]) => (
                          <span key={name}>
                            <b>{name}</b> mean {stats.mean.toPrecision(4)} · range {stats.min.toPrecision(4)}–{stats.max.toPrecision(4)}
                          </span>
                        ))}
                      </div>
                    )}
                    {file.top_level_keys && <p>JSON keys: {file.top_level_keys.join(', ')}</p>}
                    {file.preview && <pre>{file.preview}</pre>}
                    {file.skipped && <p>{file.skipped}</p>}
                  </article>
                ))}
              </div>
              {explanation.domain_analysis && (
                <section className="evidence-specialized-analysis">
                  <h3>Specialized analysis</h3>
                  <p>Additional structured findings supplied by the registered analysis skill.</p>
                  <pre>{JSON.stringify(explanation.domain_analysis, null, 2)}</pre>
                  {explanation.domain_interpretation_limits?.map(limit => <p key={limit}>{limit}</p>)}
                </section>
              )}
              {explanation.skipped.length > 0 && (
                <p className="evidence-inline-note">{explanation.skipped.length} file(s) could not be inspected.</p>
              )}
              <div className="evidence-explain-limitations">
                <strong>Interpretation limits</strong>
                <ul>{explanation.limitations.map(item => <li key={item}>{item}</li>)}</ul>
              </div>
            </>
          ) : null}
        </section>
      </div>}


    </section>
  );
}

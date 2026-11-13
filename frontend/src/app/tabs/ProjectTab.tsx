import { useRef, useState, type ChangeEvent, type ReactElement } from 'react';
import { importWellList } from '../../api/backend';
import type { TabContext } from './TabContext';

/** Tab UI extracted from App; shared application state/actions arrive via context. */
type Props = { context: TabContext };

export default function ProjectTab({ context }: Props): ReactElement {
  const wellListInput = useRef<HTMLInputElement>(null);
  const [wellListBusy, setWellListBusy] = useState(false);
  const [wellListMessage, setWellListMessage] = useState('');
  const handleWellListFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file || !context.projectId) return;
    setWellListBusy(true);
    setWellListMessage('');
    try {
      const result = await importWellList(context.projectId, file);
      await context.selectProject(context.projectId);
      const { created, updated, skipped } = result.summary;
      const firstError = result.summary.errors[0];
      setWellListMessage(`Imported ${created} new and updated ${updated} existing wells${skipped ? `; skipped ${skipped} rows${firstError ? ` (row ${firstError.row}: ${firstError.error})` : ''}` : ''}.`);
    } catch (error) {
      setWellListMessage(error instanceof Error ? error.message : 'Well-list import failed.');
    } finally {
      setWellListBusy(false);
    }
  };
  const { Check, Pencil, Plus, RefreshCw, Trash2, active, backendStatus, handleDeleteProject, handleDeleteSurvey, handleDeleteWell, handleRefreshProjects, inspectSurveyDatasets, items, loadingInventory, openCreateProjectDialog, openCreateSurveyDialog, openCreateWellDialog, openEditProjectDialog, openEditSurveyDialog, openEditWellDialog, projectBusy, projectId, projects, refreshSurveys, selectProject, selected, selectedProject, selectedSurveyId, selectedWellId, setSelectedSurveyId, setSelectedWellId, surveyBusy, surveys, wells } = context;
  const selectedSurvey = surveys.find(survey => survey.id === selectedSurveyId);
  const selectedWell = wells.find(well => well.id === selectedWellId);
  return (
    <section className="page project-page">
        <div className="page-title"><div><span className="eyebrow">PROJECT CONTEXT</span><h1>{selectedProject?.name ?? (backendStatus === 'Connected' ? 'No project selected' : 'Project workspace')}</h1><p>{selectedProject?.description || (projectId ? 'Manage the current project, its surveys, and wells.' : 'Create or select a project to organize your subsurface data.')}</p></div></div>
        <div className="project-current-strip"><div><span>ACTIVE PROJECT</span><strong>{selectedProject?.name ?? 'None selected'}</strong><small>{selectedProject ? `Created ${new Date(selectedProject.created_at).toLocaleDateString()}` : 'Choose an existing project or create one below.'}</small></div><div><span>DATASETS</span><strong>{loadingInventory ? 'Loading…' : items.length}</strong><small>In the active project</small></div><div><span>SURVEYS</span><strong>{surveys.length}</strong><small>Survey records</small></div><div><span>WELLS</span><strong>{wells.length}</strong><small>Well records</small></div></div>
              <div className="workspace-card theme-panel">
        <span>PROJECT / WORKSPACE · {backendStatus.toUpperCase()}</span>
        {projects.length ? <select aria-label="Backend project" value={projectId} onChange={e => { void selectProject(e.target.value); }} disabled={projectBusy}><option value="" disabled>Select project</option>{projects.map(project => <option key={project.id} value={project.id}>{project.name}</option>)}</select> : <strong>{backendStatus === 'Connected' ? 'No projects yet' : backendStatus === 'Connecting' ? 'Connecting…' : 'Backend offline'}</strong>}
        {selectedProject && <div className="workspace-project-detail theme-inset"><small>{selectedProject.description || 'No description'}</small><small>Created {new Date(selectedProject.created_at).toLocaleDateString()}</small></div>}
        <small><i />{loadingInventory ? 'Loading datasets…' : `${items.length} dataset${items.length === 1 ? '' : 's'} in this project`}</small>
        <div className="workspace-actions theme-action-group">
          <button type="button" className="btn btn-secondary" onClick={openCreateProjectDialog} disabled={backendStatus !== 'Connected' || projectBusy} title="Create project"><Plus />New</button>
          <button type="button" className="btn btn-secondary" onClick={openEditProjectDialog} disabled={!projectId || backendStatus !== 'Connected' || projectBusy} title="Edit project name and description"><Pencil />Edit</button>
          <button type="button" className="btn btn-danger" onClick={() => void handleDeleteProject()} disabled={!projectId || backendStatus !== 'Connected' || projectBusy} title="Delete project"><Trash2 />Delete</button>
          <button type="button" className="btn btn-secondary" onClick={() => void handleRefreshProjects()} disabled={backendStatus !== 'Connected' || projectBusy} title="Refresh projects"><RefreshCw className={projectBusy ? 'spin' : ''}/></button>
        </div>
        <div className="workspace-surveys">
          <span className="survey-label">SURVEYS · {surveys.length}</span>
          {surveys.length ? <select aria-label="Project survey" value={selectedSurveyId} onChange={event => setSelectedSurveyId(event.target.value)} disabled={surveyBusy || projectBusy}><option value="" disabled>Select survey</option>{surveys.map(survey => <option key={survey.id} value={survey.id}>{survey.name}</option>)}</select> : <small className="survey-empty">{projectId ? 'No surveys in this project yet.' : 'Select a project to manage surveys.'}</small>}
          {selectedSurveyId && <small className="survey-detail">{selectedSurvey?.crs || 'CRS not set'} · {selectedSurvey?.description || 'No description'}</small>}
          <div className="workspace-actions survey-actions theme-action-group">
            <button type="button" className="btn btn-secondary" onClick={openCreateSurveyDialog} disabled={!projectId || backendStatus !== 'Connected' || surveyBusy || projectBusy} title="Create survey"><Plus />New</button>
            <button type="button" className="btn btn-secondary" onClick={openEditSurveyDialog} disabled={!selectedSurveyId || backendStatus !== 'Connected' || surveyBusy || projectBusy} title="Edit survey"><Pencil />Edit</button>
            <button type="button" className="btn btn-danger" onClick={() => void handleDeleteSurvey()} disabled={!selectedSurveyId || backendStatus !== 'Connected' || surveyBusy || projectBusy} title="Delete survey"><Trash2 />Delete</button>
            <button type="button" className="btn btn-secondary" onClick={() => void refreshSurveys()} disabled={!projectId || backendStatus !== 'Connected' || surveyBusy || projectBusy} title="Refresh surveys"><RefreshCw className={surveyBusy ? 'spin' : ''}/></button>
          </div>
          <button type="button" className="survey-datasets-button theme-control" onClick={() => void inspectSurveyDatasets()} disabled={!selectedSurveyId || surveyBusy || projectBusy}>Check linked datasets</button>
        </div>
        <div className="workspace-surveys workspace-wells">
          <button type="button" className="btn btn-secondary" onClick={() => wellListInput.current?.click()} disabled={!projectId || backendStatus !== 'Connected' || wellListBusy || projectBusy} title="Import wells from an Excel well list">
            {wellListBusy ? 'Loading well list…' : 'Load well list'}
          </button>
          <input ref={wellListInput} type="file" accept=".xlsx,.xlsm" hidden onChange={handleWellListFile} />
          {wellListMessage && <small role="status" className="survey-detail">{wellListMessage}</small>}
          <span className="survey-label">WELLS · {wells.length}</span>
          <small className="survey-empty">Manage well headers and location metadata for this project.</small>
          {wells.length > 0 && <select aria-label="Project wells" value={selectedWellId} onChange={event => setSelectedWellId(event.target.value)}><option value="">Select well…</option>{wells.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}</select>}
          {selectedWell && <small className="survey-detail">RT: {selectedWell.rt ?? '—'} · TD: {selectedWell.td ?? '—'} · X: {selectedWell.x ?? '—'} · Y: {selectedWell.y ?? '—'}</small>}
          <div className="workspace-actions survey-actions theme-action-group"><button type="button" className="btn btn-secondary" onClick={openCreateWellDialog} disabled={!projectId || backendStatus !== 'Connected' || surveyBusy || projectBusy} title="Create well"><Plus />New</button><button type="button" className="btn btn-secondary" onClick={() => { const w = wells.find(candidate => candidate.id === selectedWellId); if (w)
            openEditWellDialog(w); }} disabled={!selectedWellId || backendStatus !== 'Connected' || surveyBusy} title="Edit selected well"><Pencil />Edit</button><button type="button" className="btn btn-danger" onClick={() => { const w = wells.find(candidate => candidate.id === selectedWellId); if (w)
            void handleDeleteWell(w); }} disabled={!selectedWellId || backendStatus !== 'Connected' || surveyBusy} title="Delete selected well"><Trash2 />Delete</button></div>
        </div>
      </div>
      </section>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactElement } from 'react';
import {
    Activity,
    Code2,
    AlertTriangle,
    Check,
    CircleCheck,
    CircleX,
    Database,
    Layers3,
    MessageSquareText,
    PanelRight,
    PanelLeftClose,
    PanelLeftOpen,
    Pencil,
    Play,
    Plus,
    RefreshCw,
    Server,
    Settings2,
    Moon,
    Sun,
    Trash2,
    UploadCloud,
    Waves,
    Workflow,
} from 'lucide-react';
import type { DataItem, ParsedData } from './data';
import { fromBackendDataset, getWellTableKind, identifyFile, isSegyDataset, isTextTableDataset, isWellDataset, sizeLabel } from './data';
import SettingsPage from './SettingsPage';
import DialogFrame from './DialogFrame';
import { parseGeologicalFile } from '../parsers';
import ProjectTab from './tabs/ProjectTab';
import InputDataTab from './tabs/InputDataTab';
import SeismicViewerTab from './tabs/SeismicViewerTab';
import WellViewerTab from './tabs/WellViewerTab';
import AskGeoTab from './tabs/AskGeoTab';
import EvidenceTab from './tabs/EvidenceTab';
import ProcessingTab from './tabs/ProcessingTab';
import ProcessingWorkflowTab from './tabs/ProcessingWorkflowTab';
import {
    checkBackend,
    createProject,
    deleteGeoChatConversation,
    createSurvey,
    createWell,
    deleteDataset,
    deleteProject,
    deleteSurvey,
    deleteWell,
    downloadDataset,
    getCapabilities,
    getDataset,
    getDatasetCurves,
    getDatasetHeader,
    getDatasetTable,
    getDatasetData,
    getDatasetValidation,
    getCurveSamples,
    getGeoChatConversation,
    getProcessingJob,
    getProject,
    getRootHealth,
    getVersionedHealth,
    getWell,
    listDatasets,
    linkLocalDataset as registerLinkedDataset,
    listGeoChatConversations,
    listProjects,
    listSurveyDatasets,
    listSurveys,
    listWells,
    updateDataset,
    updateProject,
    updateSurvey,
    updateWell,
    sendGeoChat,
    uploadDatasetAdvanced,
    type BackendCapabilities,
    type GeoChatConversation,
    type GeoChatMessage,
    type BackendProject,
    type BackendSurvey,
    type BackendWell,
    type SystemHealth,
} from '../api/backend';

type View = 'Project' | 'Input Data' | 'Seismic Viewer' | 'Well Viewer' | 'Processing' | 'Scripting' | 'Ask Geo' | 'Evidence' | 'System Status' | 'Settings';

function createGeoChatId(): string {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function')
        return crypto.randomUUID();
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, character => {
        const random = Math.floor(Math.random() * 16);
        return (character === 'x' ? random : (random & 0x3) | 0x8).toString(16);
    });
}

function App() {
    const [theme, setTheme] = useState<'dark' | 'light'>(() => {
        try { return localStorage.getItem('apgce-theme') === 'light' ? 'light' : 'dark'; }
        catch { return 'dark'; }
    });
    const [colorScheme, setColorScheme] = useState<'default' | 'blue' | 'purple' | 'red' | 'yellow'>(() => {
        try {
            const saved = localStorage.getItem('apgce-color-scheme');
            return saved === 'blue' || saved === 'purple' || saved === 'red' || saved === 'yellow' ? saved : 'default';
        } catch { return 'default'; }
    });
    useEffect(() => {
        document.documentElement.dataset.theme = theme;
        try { localStorage.setItem('apgce-theme', theme); } catch { /* Theme remains active for this session. */ }
    }, [theme]);
    useEffect(() => {
        document.documentElement.dataset.colorScheme = colorScheme;
        try { localStorage.setItem('apgce-color-scheme', colorScheme); } catch { /* Scheme remains active for this session. */ }
    }, [colorScheme]);
    const [view, setView] = useState<View>('Ask Geo');
    const [requestedProcessingJobId, setRequestedProcessingJobId] = useState('');
    const [requestedProcessingScriptName, setRequestedProcessingScriptName] = useState('');
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [items, setItems] = useState<DataItem[]>([]);
    const [projects, setProjects] = useState<BackendProject[]>([]);
    const [projectId, setProjectId] = useState('');
    const [selectedProject, setSelectedProject] = useState<BackendProject | null>(null);
    const [surveys, setSurveys] = useState<BackendSurvey[]>([]);
    const [wells, setWells] = useState<BackendWell[]>([]);
    const [selectedWellId, setSelectedWellId] = useState('');
    const [wellDialogMode, setWellDialogMode] = useState<'create' | 'edit' | null>(null);
    const [editingWellId, setEditingWellId] = useState('');
    const [wellNameInput, setWellNameInput] = useState('');
    const [wellUwiInput, setWellUwiInput] = useState('');
    const [wellXInput, setWellXInput] = useState('');
    const [wellYInput, setWellYInput] = useState('');
    const [wellCrsInput, setWellCrsInput] = useState('');
    const [wellMetadataInput, setWellMetadataInput] = useState('{}');
    const [selectedSurveyId, setSelectedSurveyId] = useState('');
    const [surveyBusy, setSurveyBusy] = useState(false);
    const [surveyDialogMode, setSurveyDialogMode] = useState<'create' | 'edit' | null>(null);
    const [surveyNameInput, setSurveyNameInput] = useState('');
    const [surveyDescriptionInput, setSurveyDescriptionInput] = useState('');
    const [surveyCrsInput, setSurveyCrsInput] = useState('');
    const [surveyMetadataInput, setSurveyMetadataInput] = useState('{}');
    const [projectBusy, setProjectBusy] = useState(false);
    const [projectDialogMode, setProjectDialogMode] = useState<'create' | 'edit' | null>(null);
    const [projectNameInput, setProjectNameInput] = useState('');
    const [projectDescriptionInput, setProjectDescriptionInput] = useState('');
    const [backendStatus, setBackendStatus] = useState<'Connecting' | 'Connected' | 'Offline'>('Connecting');
    const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
    const [versionedHealth, setVersionedHealth] = useState<SystemHealth | null>(null);
    const [capabilities, setCapabilities] = useState<BackendCapabilities | null>(null);
    const [systemLoading, setSystemLoading] = useState(false);
    const [systemCheckedAt, setSystemCheckedAt] = useState<string | null>(null);
    const [systemError, setSystemError] = useState('');
    const [loadingInventory, setLoadingInventory] = useState(false);
    const [selected, setSelected] = useState('');
    // Keep the seismic viewer's API target independent of the shared Input Data selection.
    const [selectedSeismicDatasetId, setSelectedSeismicDatasetId] = useState('');
    const [dragging, setDragging] = useState(false);
    const [toast, setToast] = useState('');
    const [datasetApiPanel, setDatasetApiPanel] = useState<{
        id: string;
        title: string;
        data: unknown;
    } | null>(null);
    const [datasetApiBusy, setDatasetApiBusy] = useState(false);
    const [query, setQuery] = useState('');
    const [geoChatMessages, setGeoChatMessages] = useState<GeoChatMessage[]>([]);
    const [geoChatConversationId, setGeoChatConversationId] = useState(createGeoChatId);
    const [geoChatBusy, setGeoChatBusy] = useState(false);
    const [geoChatError, setGeoChatError] = useState('');
    const [geoChatConversations, setGeoChatConversations] = useState<GeoChatConversation[]>([]);
    const [geoChatHistoryLoading, setGeoChatHistoryLoading] = useState(false);
    const [geoChatSuggestedQuestions, setGeoChatSuggestedQuestions] = useState<string[]>([]);
    const [activeProcessingJobId, setActiveProcessingJobId] = useState<string | null>(null);
    const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
    const uploadFromChatRef = useRef(false);
    const geoChatProjectRef = useRef(projectId);
    geoChatProjectRef.current = projectId;
    const refreshGeoChatHistory = async (targetProjectId: string) => {
        if (!targetProjectId || backendStatus !== 'Connected') { setGeoChatConversations([]); return; }
        setGeoChatHistoryLoading(true);
        try {
            const conversations = await listGeoChatConversations(targetProjectId);
            if (geoChatProjectRef.current === targetProjectId) setGeoChatConversations(conversations);
        }
        catch (error) { setGeoChatError(error instanceof Error ? error.message : 'Could not load chat history.'); }
        finally { setGeoChatHistoryLoading(false); }
    };
    useEffect(() => {
        setGeoChatMessages([]);
        setGeoChatSuggestedQuestions([]);
        setActiveProcessingJobId(null);
        setGeoChatConversationId(createGeoChatId());
        setGeoChatError('');
        if (projectId && backendStatus === 'Connected') void refreshGeoChatHistory(projectId);
        else setGeoChatConversations([]);
    }, [projectId, backendStatus]);
    useEffect(() => {
        if (!activeProcessingJobId) return;
        let cancelled = false;
        let timer: number | undefined;
        const poll = async () => {
            try {
                const job = await getProcessingJob(activeProcessingJobId);
                if (cancelled) return;
                const status = String(job.status ?? '').toLowerCase();
                if (status === 'succeeded' || status === 'success' || status === 'completed' || status === 'complete') {
                    setGeoChatMessages(current => [...current, {
                        role: 'assistant',
                        content: `Processing job **${activeProcessingJobId}** completed successfully. Its outputs are available in Processing and the Evidence Ledger.`,
                    }]);
                    setGeoChatSuggestedQuestions(['Show the processing job details.', 'Explain the outputs from this job.', 'Compare this job with another run.']);
                    setActiveProcessingJobId(null);
                    return;
                }
                if (['failed', 'submission_failed', 'cancelled'].includes(status)) {
                    const detail = job.error_message ? ` Details: ${job.error_message}` : '';
                    setGeoChatMessages(current => [...current, {
                        role: 'assistant',
                        content: `Processing job **${activeProcessingJobId}** finished with status **${status}**.${detail}`,
                    }]);
                    setGeoChatSuggestedQuestions(['Help me troubleshoot the failed processing job.', 'Show me the processing job details.', 'What processing workflows can I run?']);
                    setActiveProcessingJobId(null);
                    return;
                }
                timer = window.setTimeout(poll, 5000);
            } catch {
                if (!cancelled) timer = window.setTimeout(poll, 10000);
            }
        };
        timer = window.setTimeout(poll, 5000);
        return () => { cancelled = true; if (timer !== undefined) window.clearTimeout(timer); };
    }, [activeProcessingJobId]);

    const loadGeoChatConversation = async (conversationId: string) => {
        if (!projectId) return;
        if (!conversationId) { await refreshGeoChatHistory(projectId); return; }
        setGeoChatHistoryLoading(true);
        setGeoChatError('');
        try {
            const conversation = await getGeoChatConversation(conversationId, projectId);
            setGeoChatConversationId(conversation.conversation_id);
            setGeoChatMessages(conversation.messages);
            setGeoChatSuggestedQuestions(["Explore this project's datasets.", "List the wells and surveys.", "Show completed processing jobs."]);
            setActiveProcessingJobId(null);
        } catch (error) {
            setGeoChatError(error instanceof Error ? error.message : 'Could not load conversation.');
        } finally { setGeoChatHistoryLoading(false); }
    };
    const deleteGeoChatHistory = async (conversationId: string) => {
        if (!projectId || geoChatBusy || geoChatHistoryLoading) return;
        const conversation = geoChatConversations.find(item => item.conversation_id === conversationId);
        if (!window.confirm(`Delete this chat history${conversation ? `?\n\n${conversation.title}` : '?'}`)) return;
        setGeoChatHistoryLoading(true);
        setGeoChatError('');
        try {
            await deleteGeoChatConversation(conversationId, projectId);
            setGeoChatConversations(current => current.filter(item => item.conversation_id !== conversationId));
            if (geoChatConversationId === conversationId) {
                setGeoChatMessages([]);
                setGeoChatSuggestedQuestions([]);
                setActiveProcessingJobId(null);
                setGeoChatConversationId(createGeoChatId());
            }
        } catch (error) {
            setGeoChatError(error instanceof Error ? error.message : 'Could not delete conversation.');
        } finally { setGeoChatHistoryLoading(false); }
    };
    useEffect(() => { if (!toast)
        return; const timer = window.setTimeout(() => setToast(''), 4500); return () => window.clearTimeout(timer); }, [toast]);
    const cancelUploadDialog = () => { uploadFromChatRef.current = false; setUploadDialogOpen(false); };
    const fileRef = useRef<HTMLInputElement>(null);
    const folderRef = useRef<HTMLInputElement>(null);
    // `selected` drives the shared inventory; the seismic API target has its own fallback
    // so canvas/response updates do not unmount its Dataset Operations actions.
    const active = items.find(x => x.id === selected) ?? items[0];
    const activeSeismic = active?.parsed?.type === 'seismic' ? active.parsed : undefined;
    const selectedSeismicDataset = items.find(item => item.id === selected && item.kind === 'Seismic')
        ?? items.find(item => item.id === selectedSeismicDatasetId && item.kind === 'Seismic')
        ?? items.find(item => item.kind === 'Seismic' && item.surveyId === selectedSurveyId)
        ?? items.find(item => item.kind === 'Seismic');
    useEffect(() => {
        if (active?.kind === 'Seismic') setSelectedSeismicDatasetId(active.id);
    }, [active?.id, active?.kind]);
    const activeWell = active?.parsed?.type === 'well' ? active.parsed : undefined;
    const activeTable = active?.parsed?.type === 'table' ? active.parsed : undefined;
    const totals = useMemo(() => ({
        Seismic: items.filter(x => x.kind === 'Seismic').length,
        Wells: items.filter(x => x.kind === 'Well log').length,
        Tables: items.filter(x => x.kind === 'Table').length,
    }), [items]);
    const surveyNameForDataset = useCallback((item: DataItem) => {
        if (item.kind === 'Seismic')
            return item.surveyId ? (surveys.find(survey => survey.id === item.surveyId)?.name ?? 'Survey') : 'Unassigned';
        if (isWellDataset(item))
            return item.wellId ? (wells.find(well => well.id === item.wellId)?.name ?? 'Well') : 'Unassigned';
        return 'Unassigned';
    }, [surveys, wells]);
    const surveySeismicItems = items.filter(item => item.kind === 'Seismic' && item.surveyId === selectedSurveyId);
    const refreshProjectDatasets = async (id: string) => {
        if (!id) {
            setItems([]);
            setSelected('');
            setLoadingInventory(false);
            return;
        }
        setLoadingInventory(true);
        setItems([]);
        setSelected('');
        try {
            const datasets = await listDatasets(id);
            const mapped = datasets.map(fromBackendDataset).filter((item): item is DataItem => item !== null);
            setItems(mapped);
            setSelected(mapped[0]?.id ?? '');
        }
        catch (error) {
            setToast(`Could not load backend datasets: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setLoadingInventory(false);
        }
    };
    const selectProject = async (id: string) => {
        setProjectId(id);
        if (!id) {
            setSelectedProject(null);
            setSurveys([]);
            setWells([]);
            setSelectedSurveyId('');
            setItems([]);
            setSelected('');
            setLoadingInventory(false);
            return;
        }
        setProjectBusy(true);
        try {
            const detail = await getProject(id);
            setSelectedProject(detail);
            const [projectSurveys, projectWells] = await Promise.all([listSurveys(id), listWells(id)]);
            setSurveys(projectSurveys);
            setWells(projectWells);
            setSelectedWellId(current => projectWells.some(well => well.id === current) ? current : (projectWells[0]?.id ?? ''));
            setSelectedSurveyId(current => projectSurveys.some(survey => survey.id === current) ? current : (projectSurveys[0]?.id ?? ''));
            await refreshProjectDatasets(id);
        }
        catch (error) {
            setSelectedProject(null);
            setSurveys([]);
            setWells([]);
            setSelectedSurveyId('');
            setToast(`Could not load project details: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setProjectBusy(false);
        }
    };
    const reloadProjects = async (preferredId?: string) => {
        const availableProjects = await listProjects();
        setProjects(availableProjects);
        const nextId = preferredId && availableProjects.some(project => project.id === preferredId)
            ? preferredId
            : availableProjects[0]?.id ?? '';
        await selectProject(nextId);
    };
    const handleRefreshProjects = async () => {
        if (backendStatus !== 'Connected' || projectBusy)
            return;
        setProjectBusy(true);
        try {
            await reloadProjects(projectId);
            setToast('Project list refreshed.');
        }
        catch (error) {
            setToast(`Could not refresh projects: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setProjectBusy(false);
        }
    };
    const openCreateProjectDialog = () => {
        if (backendStatus !== 'Connected' || projectBusy)
            return;
        setProjectNameInput('');
        setProjectDescriptionInput('');
        setProjectDialogMode('create');
    };
    const openEditProjectDialog = () => {
        if (!selectedProject || backendStatus !== 'Connected' || projectBusy)
            return;
        setProjectNameInput(selectedProject.name);
        setProjectDescriptionInput(selectedProject.description ?? '');
        setProjectDialogMode('edit');
    };
    const handleSaveProject = async () => {
        const name = projectNameInput.trim();
        if (!name) {
            setToast('Project name is required.');
            return;
        }
        const description = projectDescriptionInput.trim() || null;
        setProjectBusy(true);
        try {
            if (projectDialogMode === 'create') {
                const created = await createProject(name, description);
                setProjectDialogMode(null);
                await reloadProjects(created.id);
                setToast(`Project “${created.name}” created.`);
            }
            else if (projectDialogMode === 'edit' && selectedProject) {
                const current = selectedProject;
                if (name === current.name && description === current.description) {
                    setProjectDialogMode(null);
                    return;
                }
                const updated = await updateProject(current.id, { name, description });
                setProjects(old => old.map(project => project.id === updated.id ? updated : project));
                setSelectedProject(updated);
                setProjectDialogMode(null);
                setToast(`Project “${updated.name}” updated.`);
            }
        }
        catch (error) {
            setToast(`Could not ${projectDialogMode === 'create' ? 'create' : 'update'} project: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setProjectBusy(false);
        }
    };
    const handleDeleteProject = async () => {
        const current = selectedProject;
        if (!current) {
            setToast('Select a project to delete.');
            return;
        }
        const confirmed = window.confirm(`Delete project “${current.name}”? This permanently deletes its surveys, wells, datasets, and stored uploaded files. This cannot be undone.`);
        if (!confirmed)
            return;
        setProjectBusy(true);
        try {
            await deleteProject(current.id);
            const remaining = projects.filter(project => project.id !== current.id);
            setProjects(remaining);
            const nextId = remaining[0]?.id ?? '';
            await selectProject(nextId);
            setToast(`Project “${current.name}” deleted.`);
        }
        catch (error) {
            setToast(`Could not delete project: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setProjectBusy(false);
        }
    };
    const openCreateWellDialog = () => {
        if (!projectId || backendStatus !== 'Connected' || surveyBusy)
            return;
        setEditingWellId('');
        setWellNameInput('');
        setWellUwiInput('');
        setWellXInput('');
        setWellYInput('');
        setWellCrsInput('');
        setWellMetadataInput('{}');
        setWellDialogMode('create');
    };
    const openEditWellDialog = (well: BackendWell) => {
        setEditingWellId(well.id);
        setWellNameInput(well.name);
        setWellUwiInput(well.uwi ?? '');
        setWellXInput(well.x == null ? '' : String(well.x));
        setWellYInput(well.y == null ? '' : String(well.y));
        setWellCrsInput(well.crs ?? '');
        setWellMetadataInput(JSON.stringify(well.metadata ?? {}, null, 2));
        setWellDialogMode('edit');
    };
    const handleSaveWell = async () => {
        const name = wellNameInput.trim();
        if (!name || !projectId)
            return;
        let metadata: Record<string, unknown>;
        try {
            const parsed: unknown = JSON.parse(wellMetadataInput || '{}');
            if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed))
                throw new Error('Metadata must be a JSON object.');
            metadata = parsed as Record<string, unknown>;
        }
        catch (error) {
            setToast(error instanceof Error ? error.message : 'Invalid metadata JSON.');
            return;
        }
        const numberOrNull = (value: string) => value.trim() === '' ? null : Number(value);
        const x = numberOrNull(wellXInput), y = numberOrNull(wellYInput);
        if ((x !== null && !Number.isFinite(x)) || (y !== null && !Number.isFinite(y))) {
            setToast('X and Y coordinates must be valid numbers.');
            return;
        }
        setSurveyBusy(true);
        try {
            const mode = wellDialogMode;
            if (!mode)
                return;
            const payload = { name, uwi: wellUwiInput.trim() || null, x, y, crs: wellCrsInput.trim() || null, metadata };
            const saved = mode === 'create' ? await createWell(projectId, payload) : await updateWell(editingWellId, payload);
            // Confirm the write against the API and reload the canonical project list instead of
            // relying only on optimistic React state. This makes persistence/configuration issues visible.
            const confirmed = await getWell(saved.id);
            if (confirmed.project_id !== projectId)
                throw new Error('The API saved the well under a different project than the active project.');
            const refreshedWells = await listWells(projectId);
            if (!refreshedWells.some(well => well.id === confirmed.id)) {
                throw new Error('The API returned success, but the well is missing from the project well list. Check that the API uses one persistent DATABASE_URL.');
            }
            setWells(refreshedWells);
            setSelectedWellId(confirmed.id);
            setWellDialogMode(null);
            setToast(`Well “${confirmed.name}” ${mode === 'create' ? 'created and verified' : 'updated and verified'} in the backend.`);
        }
        catch (error) {
            setToast(`Could not save well: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setSurveyBusy(false);
        }
    };
    const handleDeleteWell = async (well: BackendWell) => {
        if (!window.confirm(`Delete well “${well.name}”? Well-log datasets will be retained but detached from this well.`))
            return;
        setSurveyBusy(true);
        try {
            await deleteWell(well.id);
            const [refreshedWells, refreshedDatasets] = await Promise.all([listWells(projectId), listDatasets(projectId)]);
            if (refreshedWells.some(current => current.id === well.id))
                throw new Error('The API returned success, but the well still appears in the database-backed list.');
            setWells(refreshedWells);
            setSelectedWellId(current => current === well.id ? (refreshedWells[0]?.id ?? '') : current);
            setItems(refreshedDatasets.map(fromBackendDataset).filter((item): item is DataItem => item !== null));
            setToast(`Well “${well.name}” deleted and verified. Linked datasets were retained.`);
        }
        catch (error) {
            setToast(`Could not delete well: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setSurveyBusy(false);
        }
    };
    const handleAssignDatasetWell = async (item: DataItem, wellId: string) => {
        if (!item.id || backendStatus !== 'Connected')
            return;
        setSurveyBusy(true);
        try {
            await updateDataset(item.id, { well_id: wellId || null });
            const [confirmed, refreshedDatasets] = await Promise.all([getDataset(item.id), listDatasets(projectId)]);
            if ((confirmed.well_id ?? null) !== (wellId || null))
                throw new Error('The API did not retain the requested well association.');
            setItems(refreshedDatasets.map(fromBackendDataset).filter((current): current is DataItem => current !== null));
            setToast(confirmed.well_id ? `“${item.name}” linked and verified with ${wells.find(w => w.id === confirmed.well_id)?.name ?? 'well'}.` : `“${item.name}” unassigned and verified.`);
        }
        catch (error) {
            setToast(`Could not update well association: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setSurveyBusy(false);
        }
    };
    const refreshSurveys = async () => {
        if (!projectId || backendStatus !== 'Connected')
            return;
        setSurveyBusy(true);
        try {
            const rows = await listSurveys(projectId);
            setSurveys(rows);
            setSelectedSurveyId(current => rows.some(survey => survey.id === current) ? current : (rows[0]?.id ?? ''));
            setToast('Survey list refreshed.');
        }
        catch (error) {
            setToast(`Could not refresh surveys: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setSurveyBusy(false);
        }
    };
    const openCreateSurveyDialog = () => {
        if (!projectId || backendStatus !== 'Connected' || surveyBusy)
            return;
        setSurveyNameInput('');
        setSurveyDescriptionInput('');
        setSurveyCrsInput('');
        setSurveyMetadataInput('{}');
        setSurveyDialogMode('create');
    };
    const openEditSurveyDialog = () => {
        const current = surveys.find(survey => survey.id === selectedSurveyId);
        if (!current || backendStatus !== 'Connected' || surveyBusy)
            return;
        setSurveyNameInput(current.name);
        setSurveyDescriptionInput(current.description ?? '');
        setSurveyCrsInput(current.crs ?? '');
        setSurveyMetadataInput(JSON.stringify(current.metadata ?? {}, null, 2));
        setSurveyDialogMode('edit');
    };
    const handleSaveSurvey = async () => {
        const name = surveyNameInput.trim();
        if (!name) {
            setToast('Survey name is required.');
            return;
        }
        let metadata: Record<string, unknown>;
        try {
            const parsed: unknown = JSON.parse(surveyMetadataInput || '{}');
            if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object')
                throw new Error('Metadata must be a JSON object.');
            metadata = parsed as Record<string, unknown>;
        }
        catch (error) {
            setToast(`Invalid survey metadata JSON: ${error instanceof Error ? error.message : 'Invalid JSON'}`);
            return;
        }
        setSurveyBusy(true);
        try {
            const payload = { name, description: surveyDescriptionInput.trim() || null, crs: surveyCrsInput.trim() || null, metadata };
            if (surveyDialogMode === 'create') {
                if (!projectId)
                    throw new Error('Select a project first.');
                const created = await createSurvey(projectId, payload);
                const rows = await listSurveys(projectId);
                setSurveys(rows);
                setSelectedSurveyId(created.id);
                setSurveyDialogMode(null);
                setToast(`Survey “${created.name}” created.`);
            }
            else {
                const current = surveys.find(survey => survey.id === selectedSurveyId);
                if (!current)
                    throw new Error('Select a survey to edit.');
                const updated = await updateSurvey(current.id, payload);
                setSurveys(old => old.map(survey => survey.id === updated.id ? updated : survey));
                setSurveyDialogMode(null);
                setToast(`Survey “${updated.name}” updated.`);
            }
        }
        catch (error) {
            setToast(`Could not ${surveyDialogMode === 'create' ? 'create' : 'update'} survey: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setSurveyBusy(false);
        }
    };
    const handleDeleteSurvey = async () => {
        const current = surveys.find(survey => survey.id === selectedSurveyId);
        if (!current) {
            setToast('Select a survey to delete.');
            return;
        }
        const confirmed = window.confirm(`Delete survey “${current.name}”? Datasets linked to this survey will be kept but detached from the survey. This cannot be undone.`);
        if (!confirmed)
            return;
        setSurveyBusy(true);
        try {
            await deleteSurvey(current.id);
            const rows = await listSurveys(projectId);
            setSurveys(rows);
            setSelectedSurveyId(rows[0]?.id ?? '');
            setToast(`Survey “${current.name}” deleted. Linked datasets were retained.`);
        }
        catch (error) {
            setToast(`Could not delete survey: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setSurveyBusy(false);
        }
    };
    const inspectSurveyDatasets = async () => {
        if (!selectedSurveyId) {
            setToast('Select a survey first.');
            return;
        }
        setSurveyBusy(true);
        try {
            const linked = await listSurveyDatasets(selectedSurveyId);
            setToast(`Survey contains ${linked.length} linked dataset${linked.length === 1 ? '' : 's'}.`);
        }
        catch (error) {
            setToast(`Could not load survey datasets: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setSurveyBusy(false);
        }
    };
    const handleAssignDatasetSurvey = async (item: DataItem, surveyId: string) => {
        if (!item.id || backendStatus !== 'Connected' || surveyBusy)
            return;
        setSurveyBusy(true);
        try {
            const updated = await updateDataset(item.id, { survey_id: surveyId || null });
            setItems(old => old.map(current => current.id === item.id ? { ...current, surveyId: updated.survey_id } : current));
            if (updated.survey_id)
                setSelectedSurveyId(updated.survey_id);
            const surveyName = surveys.find(survey => survey.id === updated.survey_id)?.name;
            setToast(updated.survey_id ? `“${item.name}” linked to ${surveyName ?? 'survey'}.` : `“${item.name}” is now unassigned from any survey.`);
        }
        catch (error) {
            setToast(`Could not update dataset survey: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setSurveyBusy(false);
        }
    };
    const handleEditDataset = async (item: DataItem) => {
        if (!item.id || backendStatus !== 'Connected')
            return;
        const name = window.prompt('Edit dataset name', item.name)?.trim();
        if (!name)
            return;
        const currentSurvey = item.surveyId ?? '';
        const surveyChoices = ['Unassigned', ...surveys.map(survey => survey.name)];
        const currentName = currentSurvey ? (surveys.find(survey => survey.id === currentSurvey)?.name ?? 'Unassigned') : 'Unassigned';
        const chosen = window.prompt(`Survey association (enter one of: ${surveyChoices.join(', ')})`, currentName);
        if (chosen === null)
            return;
        const normalized = chosen.trim().toLowerCase();
        const targetSurvey = normalized === '' || normalized === 'unassigned' ? null : surveys.find(survey => survey.name.toLowerCase() === normalized);
        if (normalized !== '' && normalized !== 'unassigned' && !targetSurvey) {
            setToast('Survey not found. Choose an existing survey name or enter Unassigned.');
            return;
        }
        setSurveyBusy(true);
        try {
            const updated = await updateDataset(item.id, { name, survey_id: targetSurvey?.id ?? null });
            setItems(old => old.map(current => current.id === item.id ? { ...current, name: updated.name, surveyId: updated.survey_id } : current));
            setToast(`Dataset “${updated.name}” updated.`);
        }
        catch (error) {
            setToast(`Could not edit dataset: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setSurveyBusy(false);
        }
    };
    const handleDeleteDataset = async (item: DataItem) => {
        if (!item.id || backendStatus !== 'Connected')
            return;
        if (!window.confirm(`Delete “${item.name}” and its stored file permanently? This cannot be undone.`))
            return;
        setSurveyBusy(true);
        try {
            await deleteDataset(item.id);
            const remaining = items.filter(current => current.id !== item.id);
            setItems(remaining);
            if (selected === item.id)
                setSelected(remaining[0]?.id ?? '');
            setToast(`Dataset “${item.name}” deleted.`);
        }
        catch (error) {
            setToast(`Could not delete dataset: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setSurveyBusy(false);
        }
    };
    const showDatasetApiResult = async (item: DataItem, action: 'details' | 'header' | 'validation' | 'curves' | 'table' | 'samples') => {
        if (!item.id || item.id.startsWith('local-')) {
            setToast('This dataset is not saved in the backend yet.');
            return;
        }
        setDatasetApiBusy(true);
        try {
            let title: string = action;
            let data: unknown;
            if (action === 'details') {
                title = 'Dataset record';
                data = await getDataset(item.id);
            }
            else if (action === 'header') {
                title = 'Parsed file header';
                data = await getDatasetHeader(item.id);
            }
            else if (action === 'validation') {
                title = 'Validation report';
                data = await getDatasetValidation(item.id);
            }
            else if (action === 'curves') {
                title = 'LAS curve catalog';
                data = await getDatasetCurves(item.id);
            }
            else if (action === 'table') {
                title = 'Parsed deviation / checkshot table (first 100 rows)';
                data = await getDatasetTable(item.id, 0, 100);
            }
            else {
                title = 'Curve samples (first 100 rows)';
                const curves = await getDatasetCurves(item.id);
                const first = curves.items[0];
                data = first ? await getCurveSamples(first.id, 0, 100) : { message: 'No curves are catalogued for this LAS file.' };
            }
            setDatasetApiPanel({ id: item.id, title, data });
        }
        catch (error) {
            setDatasetApiPanel({ id: item.id, title: 'Dataset API response', data: { error: error instanceof Error ? error.message : String(error) } });
        }
        finally {
            setDatasetApiBusy(false);
        }
    };
    const downloadDatasetFile = async (item: DataItem) => {
        try {
            const blob = await downloadDataset(item.id);
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = item.name;
            anchor.click();
            URL.revokeObjectURL(url);
            setToast(`Download started for ${item.name}.`);
        }
        catch (error) {
            setToast(`Download failed: ${error instanceof Error ? error.message : String(error)}`);
        }
    };
    const editDatasetMetadata = async (item: DataItem) => {
        const current = await getDataset(item.id).catch(() => null);
        if (!current) {
            setToast('Could not retrieve the dataset record.');
            return;
        }
        const name = window.prompt('Dataset display name', current.name);
        if (name === null)
            return;
        const crs = window.prompt('Dataset CRS (leave blank to clear)', current.crs ?? '');
        if (crs === null)
            return;
        const metadataText = window.prompt('Dataset metadata JSON object', JSON.stringify(current.metadata ?? {}));
        if (metadataText === null)
            return;
        let metadata: Record<string, unknown>;
        try {
            const parsed: unknown = JSON.parse(metadataText || '{}');
            if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed))
                throw new Error('Metadata must be a JSON object.');
            metadata = parsed as Record<string, unknown>;
        }
        catch (error) {
            setToast(error instanceof Error ? error.message : 'Invalid metadata JSON.');
            return;
        }
        try {
            const updated = await updateDataset(item.id, { name: name.trim() || current.name, crs: crs.trim() || null, metadata });
            setItems(old => old.map(existing => existing.id === item.id ? { ...existing, name: updated.name, surveyId: updated.survey_id, wellId: updated.well_id } : existing));
            setToast(`Dataset “${updated.name}” updated.`);
            await showDatasetApiResult({ ...item, name: updated.name }, 'details');
        }
        catch (error) {
            setToast(`Dataset update failed: ${error instanceof Error ? error.message : String(error)}`);
        }
    };
    const renderDatasetApiPanel = (item: DataItem | undefined) => item && item.id && !item.id.startsWith('local-') ? <section className="dataset-api-tools theme-panel">
    <div className="dataset-api-heading">
      <div><span className="eyebrow">DATASET API · {item.format}</span><h3>Dataset operations</h3><p>Backend record, file parsing, validation, and original-file access for the selected dataset.</p></div>
      {/* Keep viewer-specific API actions in the shared Dataset Operations action row. */}
      <div className="dataset-api-actions">
        <div className="dataset-api-primary-actions">
          <button className="btn btn-secondary" onClick={() => void showDatasetApiResult(item, 'details')} disabled={datasetApiBusy}>Record</button>
          <button className="btn btn-secondary" onClick={() => void showDatasetApiResult(item, 'validation')} disabled={datasetApiBusy}>Validation</button>
          <button className="btn btn-secondary" onClick={() => void downloadDatasetFile(item)}>Download</button>
          <button className="btn btn-secondary" onClick={() => void editDatasetMetadata(item)} disabled={datasetApiBusy}>Edit metadata</button>
          <button className="btn btn-danger" onClick={() => void handleDeleteDataset(item)} disabled={datasetApiBusy || surveyBusy}>
            <Trash2 /> Delete
          </button>
        </div>

        <div className="dataset-api-parse-actions">
          {isSegyDataset(item) && (
            <button className="btn btn-secondary" onClick={() => void showDatasetApiResult(item, 'header')} disabled={datasetApiBusy || backendStatus !== 'Connected'}>Parse header</button>
          )}
          {item.kind !== 'Seismic' && ['sgy', 'segy', 'las'].includes(item.format.toLowerCase()) && (
            <button className="btn btn-secondary" onClick={() => void showDatasetApiResult(item, 'header')} disabled={datasetApiBusy}>Parsed header</button>
          )}
          {item.format.toLowerCase() === 'las' && (
            <>
              <button className="btn btn-secondary" onClick={() => void showDatasetApiResult(item, 'curves')} disabled={datasetApiBusy}>Curve catalog</button>
              <button className="btn btn-secondary" onClick={() => void showDatasetApiResult(item, 'samples')} disabled={datasetApiBusy}>Curve samples</button>
            </>
          )}
          {isTextTableDataset(item) && (
            <button className="btn btn-secondary" onClick={() => void showDatasetApiResult(item, 'table')} disabled={datasetApiBusy}>Parsed table</button>
          )}
          {datasetApiBusy && <span className="dataset-api-loading">Loading API response…</span>}
        </div>
      </div>
    </div>
    {datasetApiPanel?.id === item.id && <div className="dataset-api-result"><div><strong>{datasetApiPanel.title}</strong><button className="btn-text text-button" onClick={() => setDatasetApiPanel(null)}>Close</button></div><pre>{JSON.stringify(datasetApiPanel.data, null, 2)}</pre></div>}
  </section> : null;
    const refreshSystemStatus = async () => {
        setSystemLoading(true);
        setSystemError('');
        const [rootResult, apiResult, capabilityResult] = await Promise.allSettled([
            getRootHealth(), getVersionedHealth(), getCapabilities(),
        ]);
        if (rootResult.status === 'fulfilled')
            setSystemHealth(rootResult.value);
        else
            setSystemHealth(null);
        if (apiResult.status === 'fulfilled')
            setVersionedHealth(apiResult.value);
        else
            setVersionedHealth(null);
        if (capabilityResult.status === 'fulfilled')
            setCapabilities(capabilityResult.value);
        else
            setCapabilities(null);
        const errors = [rootResult, apiResult, capabilityResult]
            .filter((result): result is PromiseRejectedResult => result.status === 'rejected')
            .map(result => result.reason instanceof Error ? result.reason.message : String(result.reason));
        setSystemError(errors.join(' · '));
        setSystemCheckedAt(new Date().toISOString());
        setSystemLoading(false);
        if (rootResult.status === 'rejected')
            setBackendStatus('Offline');
        else
            setBackendStatus('Connected');
    };
    useEffect(() => {
        void refreshSystemStatus();
    }, []);
    useEffect(() => {
        let cancelled = false;
        const initialize = async () => {
            try {
                await checkBackend();
                const availableProjects = await listProjects();
                if (cancelled)
                    return;
                setProjects(availableProjects);
                setBackendStatus('Connected');
                const initialProjectId = availableProjects[0]?.id ?? '';
                if (initialProjectId) {
                    const detail = await getProject(initialProjectId);
                    if (cancelled)
                        return;
                    setProjectId(initialProjectId);
                    setSelectedProject(detail);
                    const [projectSurveys, projectWells, datasets] = await Promise.all([
                        listSurveys(initialProjectId),
                        listWells(initialProjectId),
                        listDatasets(initialProjectId),
                    ]);
                    if (cancelled)
                        return;
                    setSurveys(projectSurveys);
                    setSelectedSurveyId(projectSurveys[0]?.id ?? '');
                    setWells(projectWells);
                    setSelectedWellId(projectWells[0]?.id ?? '');
                    const mapped = datasets.map(fromBackendDataset).filter((item): item is DataItem => item !== null);
                    setItems(mapped);
                    setSelected(mapped[0]?.id ?? '');
                }
                else {
                    setProjectId('');
                    setSelectedProject(null);
                    setSurveys([]);
                    setSelectedSurveyId('');
                    setItems([]);
                    setSelected('');
                }
            }
            catch (error) {
                if (cancelled)
                    return;
                setBackendStatus('Offline');
                setToast(`Backend unavailable: ${error instanceof Error ? error.message : 'Could not connect to port 8080'}`);
            }
        };
        void initialize();
        return () => { cancelled = true; };
    }, []);
    const loadDatasetPreview = async (item: DataItem) => {
        if (item.parsed || !projectId || item.id.startsWith('local-'))
            return;
        // Deviation/checkshot tables use the backend's typed parser and paginated table API.
        // Avoid downloading the TXT file only to run the generic CSV/TSV preview parser.
        if (getWellTableKind(item))
            return;
        try {
            // The full seismic geometry and sections are served by dedicated random-access endpoints.
            // Keep the initial SEG-Y payload to one trace so opening the viewer stays lightweight.
            const payload = await getDatasetData(item.id, item.kind === 'Seismic' ? 1 : 500);
            setItems(old => old.map(current => current.id === item.id ? {
                ...current,
                parsed: payload as unknown as ParsedData,
                status: 'Ready' as const,
                details: `${current.details} · preview loaded from backend`,
            } : current));
        }
        catch (error) {
            setItems(old => old.map(current => current.id === item.id ? {
                ...current,
                status: 'Review' as const,
                issue: `Could not load preview from backend: ${error instanceof Error ? error.message : 'Unknown error'}`,
            } : current));
        }
    };
    const linkLocalDataset = async (sourcePath: string, kind: 'seismic' | 'well_log' | 'deviation' | 'checkshot', name?: string) => {
        if (!projectId || backendStatus !== 'Connected') throw new Error('Connect to the backend and select a project first.');
        const saved = await registerLinkedDataset({ project_id: projectId, kind, source_path: sourcePath.trim(), name: name?.trim() || undefined });
        const mapped = fromBackendDataset(saved);
        if (mapped) {
            setItems(old => [mapped, ...old.filter(item => item.id !== mapped.id)]);
            setSelected(mapped.id);
        }
        setToast(`Linked ${saved.original_filename} without copying the source file.`);
        return saved;
    };

    const addFiles = async (files: FileList | File[]) => {
        const accepted = Array.from(files).filter(file => identifyFile(file.name));
        if (!accepted.length) {
            setToast('No supported geology files found');
            return;
        }
        const batch = Date.now();
        const scanning: DataItem[] = accepted.map((file, i) => ({
            id: `local-${batch}-${i}`, name: file.name, kind: identifyFile(file.name)!,
            format: 'Detecting…', size: sizeLabel(file.size), status: 'Scanning',
            details: 'Parsing file and saving to backend',
        }));
        setItems(old => [...scanning, ...old]);
        setSelected(scanning[0].id);
        setToast(`Processing ${accepted.length} file${accepted.length > 1 ? 's' : ''}…`);
        const outcomes = await Promise.all(accepted.map(async (file, i) => {
            const localId = scanning[i].id;
            try {
                const identified = identifyFile(file.name);
                if (file.name.toLowerCase().endsWith('.txt') && projectId && backendStatus === 'Connected') {
                    const kindChoice = window.prompt(`Choose the dataset kind for ${file.name}: enter deviation or checkshot`, /checkshot/i.test(file.name) ? 'checkshot' : 'deviation');
                    const datasetKind = kindChoice?.trim().toLowerCase();
                    if (datasetKind !== 'deviation' && datasetKind !== 'checkshot')
                        return { ...scanning[i], status: 'Review' as const, format: 'TXT', details: 'Upload cancelled: choose deviation or checkshot for .txt files.' };
                    // Well-related TXT datasets can be uploaded without an assigned well.
                    // Users can set the association later from the Input Data tab.
                    const saved = await uploadDatasetAdvanced(file, { project_id: projectId, kind: datasetKind });
                    return { id: saved.id, name: saved.original_filename || saved.name, kind: 'Table' as const, format: 'TXT', size: sizeLabel(saved.size_bytes), status: saved.status.toLowerCase() === 'ready' ? 'Ready' as const : 'Review' as const, details: `Saved in backend · ${datasetKind} · well unassigned`, surveyId: saved.survey_id, wellId: saved.well_id };
                }
                const parsed = await parseGeologicalFile(file, localId);
                if (identified === 'Well log' && projectId && backendStatus === 'Connected') {
                    try {
                        // Upload LAS without requiring a well association. Users can assign it later.
                        const saved = await uploadDatasetAdvanced(file, { project_id: projectId, kind: 'well_log' });
                        return {
                            ...parsed,
                            id: saved.id,
                            name: saved.original_filename || saved.name,
                            size: sizeLabel(saved.size_bytes),
                            status: saved.status.toLowerCase() === 'ready' ? 'Ready' as const : 'Review' as const,
                            details: `${parsed.details} · saved to backend · well unassigned`,
                            surveyId: saved.survey_id,
                            wellId: saved.well_id,
                        };
                    }
                    catch (error) {
                        return {
                            ...parsed,
                            details: `${parsed.details} · local only (backend upload failed: ${error instanceof Error ? error.message : 'unknown error'})`,
                        };
                    }
                }
                if (identified === 'Seismic' && projectId && backendStatus === 'Connected') {
                    try {
                        // SEG-Y uploads can remain unassigned until linked in Seismic Viewer.
                        const saved = await uploadDatasetAdvanced(file, { project_id: projectId, kind: 'seismic' });
                        return {
                            ...parsed,
                            id: saved.id,
                            name: saved.original_filename || saved.name,
                            details: `${parsed.details} · saved to backend · survey unassigned`,
                            surveyId: saved.survey_id,
                            wellId: saved.well_id,
                        };
                    }
                    catch (error) {
                        return { ...parsed, details: `${parsed.details} · local only (backend upload failed: ${error instanceof Error ? error.message : 'unknown error'})` };
                    }
                }
                if (identified === 'Table')
                    return { ...parsed, details: `${parsed.details} · browser session only` };
                return { ...parsed, details: `${parsed.details} · local only (backend disconnected)` };
            }
            catch (error) {
                return { ...scanning[i], status: 'Review' as const, format: identifyFile(file.name) ?? 'Unknown', details: 'Could not complete file parsing', issue: error instanceof Error ? error.message : 'Unknown parse error' };
            }
        }));
        setItems(old => [...outcomes, ...old.filter(item => !scanning.some(pending => pending.id === item.id))]);
        setSelected(outcomes[0]?.id ?? '');
        const savedCount = outcomes.filter(item => !item.id.startsWith('local-')).length;
        const localCount = outcomes.length - savedCount;
        const failed = outcomes.filter(item => item.issue || item.details.includes('backend upload failed'));
        const summary = `${savedCount} saved to backend · ${localCount} local/session-only`;
        setToast(failed.length ? `${summary}. ${failed.length} upload/parse issue${failed.length === 1 ? '' : 's'}; see file status.` : summary);
        if (uploadFromChatRef.current) {
            uploadFromChatRef.current = false;
            const completed = outcomes.filter(item => !item.issue && item.status !== 'Review');
            const backendSaved = completed.filter(item => !item.id.startsWith('local-'));
            const sessionOnly = completed.filter(item => item.id.startsWith('local-'));
            const names = completed.map(item => item.name);
            const uploadMessage = completed.length
                ? `Upload processed ${completed.length} file${completed.length === 1 ? '' : 's'}: ${names.join(', ')}. ${backendSaved.length} saved to the backend${sessionOnly.length ? `; ${sessionOnly.length} available in this browser session only` : ''}. ${failed.length ? `${failed.length} item(s) need attention in Input Data.` : 'The inventory has been updated.'}`
                : `No files were uploaded successfully. ${failed.length ? `${failed.length} item(s) need attention in Input Data.` : 'Please check the upload result in Input Data.'}`;
            const eventMessages: GeoChatMessage[] = [...geoChatMessages, { role: 'user', content: uploadMessage }];
            setGeoChatMessages(eventMessages);
            setView('Ask Geo');
            if (projectId) {
                try {
                    const confirmation = await sendGeoChat({ conversation_id: geoChatConversationId, project_id: projectId, messages: eventMessages });
                    setGeoChatConversationId(confirmation.conversation_id);
                    setGeoChatMessages(current => [...current, confirmation.message]);
                    setGeoChatSuggestedQuestions(confirmation.suggested_questions?.length ? confirmation.suggested_questions : ['What datasets, wells, and surveys are available?', 'Show completed processing jobs.', 'What processing workflows can I run?']);
                    void refreshGeoChatHistory(projectId);
                } catch {
                    setGeoChatError('Upload completed, but the confirmation could not be saved to chat history.');
                }
            }
        }
        const firstSaved = outcomes.find(item => !item.id.startsWith('local-') && item.kind === 'Well log');
        if (firstSaved)
            void loadDatasetPreview(firstSaved);
        setTimeout(() => setToast(''), 5000);
    };
    const send = async (editIndex?: number, contentOverride?: string) => {
        const content = (contentOverride ?? query).trim();
        if (!content || geoChatBusy)
            return;
        if (!projectId) {
            setGeoChatError('Select a project before starting a conversation. Chat history is saved per project.');
            return;
        }
        const submittedProjectId = projectId;
        const submittedConversationId = geoChatConversationId;
        // Editing replaces the selected user message and discards every later turn.
        const conversationMessages = editIndex === undefined ? geoChatMessages : geoChatMessages.slice(0, editIndex);

        // Explicit filename-based viewing and short retry follow-ups are deterministic UI actions.
        // A retry reuses the last explicit viewing target in this conversation instead of asking
        // the LLM to infer what "try again" refers to.
        const viewRequestPattern = /(?:view|show|open|display)\s+(.+?\.(?:sgy|segy|las|txt))(?:[.!?]|$)/i;
        const isRetryRequest = /^(?:please\s+)?(?:try again|retry|open it again|show it again)[.!?]*$/i.test(content);
        const previousViewRequest = isRetryRequest
            ? [...conversationMessages].reverse().find(message => message.role === 'user' && viewRequestPattern.test(message.content))?.content
            : undefined;
        const filenameMatch = content.match(viewRequestPattern) ?? previousViewRequest?.match(viewRequestPattern) ?? null;
        if (filenameMatch) {
            setQuery('');
            const requestedName = filenameMatch[1].replace(/\\_/g, '_').replace(/\\/g, '').trim().replace(/^['"`]|['"`]$/g, '');
            const normalizeName = (value: string) => value
                .replace(/\\_/g, '_').replace(/\\/g, '')
                .toLowerCase().replace(/\.[a-z0-9]+$/i, '')
                .replace(/[^a-z0-9]+/g, ' ').trim();
            const requested = normalizeName(requestedName);
            const extension = requestedName.split('.').pop()?.toLowerCase();
            const candidates = items.filter(item => {
                if (extension === 'sgy' || extension === 'segy') return item.kind === 'Seismic' && isSegyDataset(item);
                if (extension === 'las') return item.kind === 'Well log';
                if (extension === 'txt') return item.kind === 'Table' && /deviation|checkshot/i.test(`${item.name} ${item.details}`);
                return false;
            });
            const matches = candidates.filter(item => {
                const candidate = normalizeName(item.name);
                return candidate === requested || candidate.includes(requested) || requested.includes(candidate);
            });
            if (matches.length === 1) {
                const target = matches[0];
                setGeoChatMessages([...conversationMessages, { role: 'user', content }]);
                setGeoChatError('');
                setGeoChatSuggestedQuestions(['Explain the dataset I just opened.', 'Show me another available dataset.', 'What can you do for me?']);
                setSelected(target.id);
                if (target.surveyId) setSelectedSurveyId(target.surveyId);
                if (target.wellId) setSelectedWellId(target.wellId);
                setView(extension === 'sgy' || extension === 'segy' ? 'Seismic Viewer' : 'Well Viewer');
                void loadDatasetPreview(target).catch(() => { const message = `I found “${target.name}”, but could not load its preview. Check that the file is accessible, then say “try again”.`; setToast(message); setGeoChatMessages(current => [...current, { role: 'assistant', content: message }]); setGeoChatSuggestedQuestions(['Retry loading this dataset.', 'Show me another available dataset.', 'What can you do for me?']); });
                return;
            }
            if (matches.length > 1) {
                setGeoChatMessages([...conversationMessages, { role: 'user', content }, { role: 'assistant', content: `I found ${matches.length} matching datasets. Please choose the exact dataset from the Input Data tab.` }]);
                setGeoChatError('');
                setGeoChatSuggestedQuestions(['Which datasets are available in this project?', 'Check the dataset name in Input Data.', 'What can you do for me?']);
                return;
            }
            setGeoChatMessages([...conversationMessages, { role: 'user', content }, { role: 'assistant', content: `I couldn't find “${requestedName}” among this project's registered, viewable datasets.` }]);
            setGeoChatError('');
            setGeoChatSuggestedQuestions(['Show me the available datasets.', 'What file types can I view?', 'What can you do for me?']);
            return;
        }

        const nextMessages: GeoChatMessage[] = [...conversationMessages, { role: 'user', content }];
        setGeoChatMessages(nextMessages);
        setGeoChatSuggestedQuestions([]);
        setQuery('');
        setGeoChatBusy(true);
        setGeoChatError('');
        try {
            const response = await sendGeoChat({ conversation_id: submittedConversationId, project_id: submittedProjectId, messages: nextMessages });
            if (geoChatProjectRef.current === submittedProjectId) {
                setGeoChatConversationId(response.conversation_id);
                setGeoChatSuggestedQuestions(response.suggested_questions?.length ? response.suggested_questions : ['What datasets, wells, and surveys are available?', 'Show completed processing jobs.', 'What processing workflows can I run?']);
                if (response.ui_action === 'watch_processing_job') {
                    const actionData = response.ui_action_data ?? {};
                    const jobId = typeof actionData.job_id === 'string' ? actionData.job_id : '';
                    if (jobId) setActiveProcessingJobId(jobId);
                }
                if (response.ui_action !== 'view_dataset' && response.ui_action !== 'open_seismic_viewer' && response.ui_action !== 'open_well_viewer') {
                    setGeoChatMessages(current => [...current, response.message]);
                }
                if (response.ui_action === 'switch_project') {
                    const actionData = response.ui_action_data ?? {};
                    const newProjectId = typeof actionData.project_id === 'string' ? actionData.project_id : '';
                    if (newProjectId) {
                        await reloadProjects(newProjectId);
                        setView('Project');
                    } else {
                        setGeoChatError('The project was created, but its ID was missing from the response. Refresh the project list.');
                    }
                } else if (response.ui_action === 'open_upload') {
                    uploadFromChatRef.current = true;
                    setView('Input Data');
                    setUploadDialogOpen(true);
                } else if (response.ui_action === 'view_dataset' || response.ui_action === 'open_seismic_viewer' || response.ui_action === 'open_well_viewer') {
                    const actionData = response.ui_action_data ?? {};
                    const requestedName = typeof actionData.dataset_name === 'string' ? actionData.dataset_name.trim() : '';
                    const requestedType = typeof actionData.dataset_type === 'string' ? actionData.dataset_type.trim().toLowerCase() : '';
                    const requestedWell = typeof actionData.well_name === 'string' ? actionData.well_name.trim() : '';
                    const requestedSurvey = typeof actionData.survey_name === 'string' ? actionData.survey_name.trim() : '';
                    const normalizeName = (value: string) => value.replace(/\\_/g, '_').replace(/\\/g, '').toLowerCase().replace(/\.[a-z0-9]+$/i, '').replace(/[^a-z0-9]+/g, ' ').trim();
                    const nameMatches = (item: DataItem) => !requestedName || normalizeName(item.name).includes(normalizeName(requestedName)) || normalizeName(requestedName).includes(normalizeName(item.name));
                    const shouldOpenSeismic = response.ui_action === 'open_seismic_viewer' || requestedType === 'segy' || requestedType === 'sgy' || /\.(sgy|segy)$/i.test(requestedName);
                    if (shouldOpenSeismic) {
                        const candidates = items.filter(item => item.kind === 'Seismic' && isSegyDataset(item));
                        const target = candidates.find(item => nameMatches(item) && (!requestedSurvey || normalizeName(surveys.find(survey => survey.id === item.surveyId)?.name ?? '').includes(normalizeName(requestedSurvey)))) ?? (!requestedName ? candidates.find(item => (requestedType === 'segy' || requestedType === 'sgy' || !requestedType) && (!requestedSurvey || normalizeName(surveys.find(survey => survey.id === item.surveyId)?.name ?? '').includes(normalizeName(requestedSurvey)))) : undefined);
                        if (target) {
                            if (target.surveyId) setSelectedSurveyId(target.surveyId);
                            setSelected(target.id);
                            setView('Seismic Viewer');
                            void loadDatasetPreview(target);
                        } else {
                            setToast(requestedName ? `Could not find a viewable SEG-Y dataset matching “${requestedName}”.` : 'No viewable SEG-Y datasets are available in this project.');
                        }
                    } else {
                        const candidates = items.filter(item => item.kind === 'Well log' || getWellTableKind(item) !== null);
                        const typeMatches = (item: DataItem) => {
                            if (!requestedType) return true;
                            if (requestedType === 'las' || requestedType === 'well_log' || requestedType === 'well log') return item.kind === 'Well log';
                            if (requestedType === 'deviation') return item.kind === 'Table' && /deviation/i.test(`${item.name} ${item.details}`);
                            if (requestedType === 'checkshot') return item.kind === 'Table' && /checkshot/i.test(`${item.name} ${item.details}`);
                            return false;
                        };
                        const target = candidates.find(item => nameMatches(item) && typeMatches(item) && (!requestedWell || normalizeName(wells.find(well => well.id === item.wellId)?.name ?? '').includes(normalizeName(requestedWell)))) ?? (!requestedName ? candidates.find(item => typeMatches(item) && (!requestedWell || normalizeName(wells.find(well => well.id === item.wellId)?.name ?? '').includes(normalizeName(requestedWell)))) : undefined);
                        if (target) {
                            if (target.wellId) setSelectedWellId(target.wellId);
                            setSelected(target.id);
                            setView('Well Viewer');
                            void loadDatasetPreview(target);
                        } else {
                            setToast(requestedName ? `Could not find a supported LAS, deviation, or checkshot dataset matching “${requestedName}”.` : 'No supported well-viewer datasets are available in this project.');
                        }
                    }
                }
                if (response.ui_action !== 'switch_project') void refreshGeoChatHistory(submittedProjectId);
            }
        }
        catch (error) {
            setGeoChatError('Ask Geo is temporarily unavailable. Your message is still here; please try again.');
            setGeoChatSuggestedQuestions(['Retry my last question.', 'Show me the available project data.', 'What can you do for me?']);
            // Keep the user's message in the current chat so they can retry after the service is available.
        }
        finally {
            setGeoChatBusy(false);
        }
    };
    const nav: {
        name: View;
        icon: ReactElement;
    }[] = [
        { name: 'Project', icon: <Layers3 /> }, { name: 'Ask Geo', icon: <MessageSquareText /> }, { name: 'Input Data', icon: <Database /> },
        { name: 'Seismic Viewer', icon: <Waves /> }, { name: 'Well Viewer', icon: <Activity /> }, { name: 'Processing', icon: <Workflow /> }, { name: 'Evidence', icon: <PanelRight /> },
    ];
    const openView = (next: View) => {
        if (next === 'Seismic Viewer') {
            const seismic = items.find(x => x.kind === 'Seismic' && x.surveyId === selectedSurveyId) ?? items.find(x => x.kind === 'Seismic');
            if (seismic) {
                setSelected(seismic.id);
                void loadDatasetPreview(seismic);
            }
        }
        if (next === 'Well Viewer') {
            const log = items.find(x => isWellDataset(x) && x.wellId === selectedWellId) ?? items.find(isWellDataset);
            if (log) {
                setSelected(log.id);
                if (log.wellId)
                    setSelectedWellId(log.wellId);
                void loadDatasetPreview(log);
            }
        }
        setView(next);
    };
    // Shared values and handlers consumed by the extracted tab views.
    const tabContext = {
        AlertTriangle,
        Check,
        Database,
            Pencil,
        Play,
        Plus,
        RefreshCw,
        Trash2,
        UploadCloud,
        uploadDialogOpen,
        setUploadDialogOpen,
        cancelUploadDialog,
        active,
        activeSeismic,
        activeTable,
        activeWell,
        addFiles,
        backendStatus,
        dragging,
        fileRef,
        folderRef,
        handleAssignDatasetSurvey,
        handleAssignDatasetWell,
        handleDeleteDataset,
        handleDeleteProject,
        handleDeleteSurvey,
        handleDeleteWell,
        handleEditDataset,
        handleRefreshProjects,
        inspectSurveyDatasets,
        items,
        loadDatasetPreview,
        linkLocalDataset,
        loadingInventory,
        openCreateProjectDialog,
        openCreateSurveyDialog,
        openCreateWellDialog,
        openEditProjectDialog,
        openEditSurveyDialog,
        openEditWellDialog,
        projectBusy,
        projectId,
        projects,
        refreshSurveys,
        renderDatasetApiPanel,
        selectProject,
        selected,
        selectedProject,
        selectedSeismicDataset,
        selectedSurveyId,
        selectedWellId,
        setDragging,
        setSelected,
        setSelectedSeismicDatasetId,
        setSelectedSurveyId,
        setSelectedWellId,
        setView,
        surveyBusy,
        surveyNameForDataset,
        surveySeismicItems,
        surveys,
        totals,
        wells,
    };

    return <div className={`app-shell${sidebarCollapsed ? ' sidebar-collapsed' : ''}`} data-color-scheme={colorScheme} data-theme={theme}>
    <aside className="sidebar">
      <div className={`sidebar-brand-row${sidebarCollapsed ? ' is-collapsed' : ''}`}>
      <div className="brand" aria-label="Ctrl-Alt-Delineate, Geoscience Intelligence">
        <div className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M7 15.5 21.5 8l19.5 9.5-19.5 9L7 19.2" stroke="currentColor" strokeWidth="2.2" strokeLinejoin="round" />
            <path d="m7 24 14.5 7.2L41 21.8M7 31.5l14.5 7.2L41 29.2" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M28.5 11.4v21.2m-4.2-4.2 4.2 4.2 4.2-4.2" stroke="var(--brand-highlight)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="28.5" cy="11.4" r="3" fill="var(--brand-highlight)" />
          </svg>
        </div>
        <div className="brand-copy">
          <strong><span className="brand-prefix">CTRL <i>·</i> ALT</span><span className="brand-name">DELINEATE</span></strong>
          <span className="brand-tagline">GEOSCIENCE INTELLIGENCE</span>
        </div>
      </div>
      <button
        className="sidebar-collapse-toggle"
        type="button"
        onClick={() => setSidebarCollapsed(current => !current)}
        aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        aria-expanded={!sidebarCollapsed}
        title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {sidebarCollapsed ? <PanelLeftOpen aria-hidden="true" /> : <PanelLeftClose aria-hidden="true" />}
      </button>
      </div>
      <div className="event-brand" aria-label="APGCE Geohackathon 2026">
        <div className="event-logo" role="img" aria-label="APGCE Geohackathon 2026" />
      </div>
      <nav>{nav.map(n => <button key={n.name} aria-label={n.name} className={`btn-nav ${view === n.name ? 'active' : ''}`} onClick={() => openView(n.name)}>{n.icon}<span>{n.name}</span></button>)}</nav>

      <div className="side-bottom">
        <button aria-label="Scripting" title="Scripting" className={`btn-nav ${view === 'Scripting' ? 'active' : ''}`} onClick={() => openView('Scripting')}><Code2 /><span>Scripting</span></button>
        <button aria-label="System status" title="System status" className={`btn-nav ${view === 'System Status' ? 'active' : ''}`} onClick={() => openView('System Status')}><Server /><span>System status</span></button>
        <button aria-label="Settings" title="Settings" className={`btn-nav ${view === 'Settings' ? 'active' : ''}`} onClick={() => openView('Settings')}><Settings2 /><span>Settings</span></button>
      </div>
    </aside>

    <main>
      <header className="topbar">
        <div className="breadcrumb" aria-label="Breadcrumb">
          {view !== 'System Status' && view !== 'Settings' && <>
            <span className="crumb" title={selectedProject?.name ?? 'No project selected'}>{selectedProject?.name ?? 'No project selected'}</span>
            <span className="breadcrumb-separator" aria-hidden="true">/</span>
          </>}
          <strong>{view}</strong>
        </div>
        <div className="header-actions">
          <label className="theme-scheme-control">
            <span>Scheme</span>
            <select aria-label="Color scheme" value={colorScheme} onChange={event => setColorScheme(event.target.value as typeof colorScheme)}>
              <option value="default">Default</option>
              <option value="blue">Blue</option>
              <option value="purple">Purple</option>
              <option value="red">Red</option>
              <option value="yellow">Yellow</option>
            </select>
          </label>
          <button className="btn-theme-toggle" type="button" onClick={() => setTheme(current => current === 'dark' ? 'light' : 'dark')} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
            {theme === 'dark' ? <Sun aria-hidden="true" /> : <Moon aria-hidden="true" />}
            <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
          </button>
          <div className={`api-status-badge ${backendStatus.toLowerCase()}`} role="status" aria-live="polite" aria-label={`API status: ${backendStatus}`}>
            <i aria-hidden="true" />
            <span>API status</span>
            <strong>{backendStatus === 'Connected' ? 'Connected' : backendStatus === 'Connecting' ? 'Connecting…' : 'Offline'}</strong>
          </div>
        </div>
      </header>

      {view === 'System Status' && <section className="page system-page">
        <div className="page-title"><div><span className="eyebrow">SYSTEM · CAPABILITIES</span><h1>Backend status</h1><p>Live health checks and capabilities reported by the Python FastAPI service on port 8080.</p></div><div className="title-actions theme-action-group"><button className="btn btn-primary" onClick={() => void refreshSystemStatus()} disabled={systemLoading}><RefreshCw className={systemLoading ? 'spin' : ''}/>{systemLoading ? 'Checking…' : 'Refresh status'}</button></div></div>
        <div className="system-overview">
          <article className={`system-health-card ${backendStatus.toLowerCase()}`}><div className="system-card-icon">{backendStatus === 'Connected' ? <CircleCheck /> : backendStatus === 'Offline' ? <CircleX /> : <Activity />}</div><div><span>BACKEND CONNECTION</span><h2>{backendStatus}</h2><p>{systemHealth?.service ?? 'geohackathon-data-api'} · port 8080</p></div></article>
          <article className="system-health-card theme-panel"><div className="system-card-icon"><Server /></div><div><span>VERSIONED API HEALTH</span><h2>{versionedHealth?.status ?? 'Unavailable'}</h2><p>{versionedHealth?.service ?? 'GET /api/v1/health'}</p></div></article>
          <article className="system-health-card theme-panel"><div className="system-card-icon"><Layers3 /></div><div><span>CAPABILITIES</span><h2>{capabilities ? 'Loaded' : 'Unavailable'}</h2><p>{capabilities ? `${capabilities.dataset_kinds.length} dataset kinds · ${capabilities.formats_supported_for_upload.length} upload formats` : 'GET /api/v1/capabilities'}</p></div></article>
        </div>
        <div className="system-meta theme-meta"><span>Root health: <code>GET /health</code></span><span>API health: <code>GET /api/v1/health</code></span><span>Capabilities: <code>GET /api/v1/capabilities</code></span><span>Last checked: {systemCheckedAt ? new Date(systemCheckedAt).toLocaleString() : 'Not checked'}</span></div>
        {systemError && <div className="system-error"><AlertTriangle /><div><strong>Some system checks failed</strong><p>{systemError}</p></div></div>}
        <div className="system-section theme-panel"><div className="system-section-heading"><div><span className="eyebrow">INGESTION CONTRACT</span><h2>Upload formats and dataset kinds</h2></div><span>{capabilities?.formats_supported_for_upload.length ?? 0} formats</span></div>
          {capabilities ? <><div className="capability-chip-list">{capabilities.formats_supported_for_upload.map(format => <span key={format}>{format}</span>)}</div><div className="capability-chip-list dataset-kinds">{capabilities.dataset_kinds.map(kind => <span key={kind}>{kind}</span>)}</div></> : <div className="page-empty">Capabilities have not been loaded. Check that the backend is running and refresh.</div>}
        </div>
        <div className="system-section theme-panel"><div className="system-section-heading"><div><span className="eyebrow">PARSER SUPPORT</span><h2>Available parsers</h2></div></div>
          {capabilities ? <div className="system-list">{Object.entries(capabilities.parsers).map(([name, description]) => <article key={name}><div><strong>{name.toUpperCase()}</strong><p>{description}</p></div><span className="availability available">Reported</span></article>)}</div> : <div className="page-empty">Parser information is unavailable.</div>}
        </div>
        <div className="system-section theme-panel"><div className="system-section-heading"><div><span className="eyebrow">PIPELINE READINESS</span><h2>Processing domains</h2></div></div>
          {capabilities ? <div className="system-list">{capabilities.pipeline_domains.map(domain => <article key={domain.id}><div><strong>{domain.name}</strong><small>{domain.id}</small><p>{domain.description}</p></div><span className={`availability ${domain.status === 'available' ? 'available' : 'placeholder'}`}>{domain.status}</span></article>)}</div> : <div className="page-empty">Pipeline readiness is unavailable.</div>}
          {capabilities && <div className="processing-status"><strong>Processing status</strong>{Object.entries(capabilities.processing_status).map(([key, status]) => <span key={key}><b>{key}</b>{status}</span>)}</div>}
        </div>
        <div className="system-links"><a href="/docs" target="_blank" rel="noreferrer">Open API documentation ↗</a><a href="/openapi.json" target="_blank" rel="noreferrer">View OpenAPI schema ↗</a></div>
      </section>}

      {view === 'Processing' && <ProcessingWorkflowTab backendConnected={backendStatus === 'Connected'} onOpenScripting={scriptName => { setRequestedProcessingScriptName(scriptName); setView('Scripting'); }} onAskGeo={question => { setQuery(question); setView('Ask Geo'); }} />}
      {view === 'Scripting' && <ProcessingTab projectId={projectId} backendConnected={backendStatus === 'Connected'} initialJobId={requestedProcessingJobId} initialScriptName={requestedProcessingScriptName}/>}

      {view === 'Settings' && <SettingsPage backendConnected={backendStatus === 'Connected'} setToast={setToast}/>}

      {view === 'Project' && <ProjectTab context={tabContext} />}

      {view === 'Input Data' && <InputDataTab context={tabContext} />}

      {view === 'Seismic Viewer' && <SeismicViewerTab context={tabContext} />}

      {view === 'Well Viewer' && <WellViewerTab context={tabContext} />}

      {view === 'Ask Geo' && <AskGeoTab query={query} setQuery={setQuery} onSend={send} messages={geoChatMessages} busy={geoChatBusy} error={geoChatError} onNewChat={() => { setGeoChatMessages([]); setGeoChatSuggestedQuestions([]); setActiveProcessingJobId(null); setGeoChatConversationId(createGeoChatId()); setGeoChatError(''); }} conversations={geoChatConversations} activeConversationId={geoChatConversationId} onLoadConversation={loadGeoChatConversation} onDeleteConversation={deleteGeoChatHistory} historyLoading={geoChatHistoryLoading} suggestedQuestions={geoChatSuggestedQuestions} onSuggestedQuestion={question => setQuery(question)} datasets={items} onOpenDataset={dataset => { setSelected(dataset.id); if (dataset.surveyId) setSelectedSurveyId(dataset.surveyId); if (dataset.wellId) setSelectedWellId(dataset.wellId); void loadDatasetPreview(dataset); setView(dataset.kind === 'Seismic' ? 'Seismic Viewer' : 'Well Viewer'); }} onOpenProcessingJob={jobId => { setRequestedProcessingJobId(jobId); setView('Scripting'); }} onOpenProcessingTab={scriptName => { setRequestedProcessingScriptName(scriptName); setView('Scripting'); }} />}

      {view === 'Evidence' && <EvidenceTab onAskGeo={question => { setQuery(question); setView('Ask Geo'); }} />}
    </main>
    {projectDialogMode && <DialogFrame
      eyebrow="PROJECT / WORKSPACE"
      title={projectDialogMode === 'create' ? 'Create project' : 'Edit project'}
      description={projectDialogMode === 'create' ? 'Set a name and optional description for this project.' : 'Update the project name or description.'}
      labelledBy="project-dialog-title"
      onClose={() => setProjectDialogMode(null)}
      closeDisabled={projectBusy}
    >
      <form onSubmit={event => { event.preventDefault(); void handleSaveProject(); }}>
        <label htmlFor="project-name">Project name <span>Required</span></label>
        <input id="project-name" autoFocus value={projectNameInput} onChange={event => setProjectNameInput(event.target.value)} placeholder="e.g. APGCE 2026 — Malay Basin" maxLength={200} required/>
        <label htmlFor="project-description">Description <span>Optional</span></label>
        <textarea id="project-description" value={projectDescriptionInput} onChange={event => setProjectDescriptionInput(event.target.value)} placeholder="Describe the study area, objectives, data scope, or project notes…" rows={5} maxLength={5000}/>
        <div className="project-description-help">{projectDescriptionInput.length}/5000 characters · Leave blank for no description.</div>
        <div className="project-dialog-actions theme-action-group"><button type="button" className="btn btn-secondary" disabled={projectBusy} onClick={() => setProjectDialogMode(null)}>Cancel</button><button type="submit" className="btn btn-primary" disabled={projectBusy || !projectNameInput.trim()}>{projectBusy ? 'Saving…' : projectDialogMode === 'create' ? 'Create project' : 'Save changes'}</button></div>
      </form>
    </DialogFrame>}
    {surveyDialogMode && <DialogFrame
      eyebrow={`SURVEY · ${selectedProject?.name ?? 'PROJECT'}`}
      title={surveyDialogMode === 'create' ? 'Create survey' : 'Edit survey'}
      description={surveyDialogMode === 'create' ? 'Add a survey under the selected project.' : 'Update the survey name, description, CRS, or metadata.'}
      labelledBy="survey-dialog-title"
      onClose={() => setSurveyDialogMode(null)}
      closeDisabled={surveyBusy}
      panelClassName="survey-dialog"
    >
      <form onSubmit={event => { event.preventDefault(); void handleSaveSurvey(); }}>
        <label htmlFor="survey-name">Survey name <span>Required</span></label>
        <input id="survey-name" autoFocus value={surveyNameInput} onChange={event => setSurveyNameInput(event.target.value)} placeholder="e.g. Malay Basin 3D Survey" maxLength={200} required/>
        <label htmlFor="survey-description">Description <span>Optional</span></label>
        <textarea id="survey-description" value={surveyDescriptionInput} onChange={event => setSurveyDescriptionInput(event.target.value)} placeholder="Survey scope, acquisition details, or notes…" rows={3} maxLength={5000}/>
        <label htmlFor="survey-crs">Coordinate reference system (CRS) <span>Optional</span></label>
        <input id="survey-crs" value={surveyCrsInput} onChange={event => setSurveyCrsInput(event.target.value)} placeholder="e.g. EPSG:32648" maxLength={200}/>
        <label htmlFor="survey-metadata">Metadata <span>JSON object</span></label>
        <textarea id="survey-metadata" className="survey-metadata-input" value={surveyMetadataInput} onChange={event => setSurveyMetadataInput(event.target.value)} rows={5} spellCheck={false}/>
        <div className="project-description-help">Metadata must be valid JSON with an object at the top level.</div>
        <div className="project-dialog-actions theme-action-group"><button type="button" className="btn btn-secondary" disabled={surveyBusy} onClick={() => setSurveyDialogMode(null)}>Cancel</button><button type="submit" className="btn btn-primary" disabled={surveyBusy || !surveyNameInput.trim()}>{surveyBusy ? 'Saving…' : surveyDialogMode === 'create' ? 'Create survey' : 'Save changes'}</button></div>
      </form>
    </DialogFrame>}
    {wellDialogMode && <DialogFrame
      eyebrow={`WELL · ${selectedProject?.name ?? 'PROJECT'}`}
      title={wellDialogMode === 'create' ? 'Create well' : 'Edit well'}
      description="Manage well identity, coordinates, CRS, and metadata."
      labelledBy="well-dialog-title"
      onClose={() => setWellDialogMode(null)}
      closeDisabled={surveyBusy}
      panelClassName="survey-dialog"
    >
      <form onSubmit={event => { event.preventDefault(); void handleSaveWell(); }}>
        <label htmlFor="well-name">Well name <span>Required</span></label>
        <input id="well-name" autoFocus value={wellNameInput} onChange={event => setWellNameInput(event.target.value)} maxLength={200} required placeholder="e.g. M-1"/>
        <label htmlFor="well-uwi">UWI / Well identifier <span>Optional</span></label>
        <input id="well-uwi" value={wellUwiInput} onChange={event => setWellUwiInput(event.target.value)} placeholder="Unique well identifier"/>
        <div className="well-coordinate-grid">
          <div><label htmlFor="well-x">X coordinate</label><input id="well-x" value={wellXInput} onChange={event => setWellXInput(event.target.value)} inputMode="decimal" placeholder="Easting / longitude"/></div>
          <div><label htmlFor="well-y">Y coordinate</label><input id="well-y" value={wellYInput} onChange={event => setWellYInput(event.target.value)} inputMode="decimal" placeholder="Northing / latitude"/></div>
        </div>
        <label htmlFor="well-crs">Coordinate reference system (CRS)</label>
        <input id="well-crs" value={wellCrsInput} onChange={event => setWellCrsInput(event.target.value)} placeholder="e.g. EPSG:32648"/>
        <label htmlFor="well-metadata">Metadata <span>JSON object</span></label>
        <textarea id="well-metadata" className="survey-metadata-input" value={wellMetadataInput} onChange={event => setWellMetadataInput(event.target.value)} rows={4} spellCheck={false}/>
        <div className="project-description-help">Metadata must be valid JSON with an object at the top level.</div>
        <div className="project-dialog-actions theme-action-group"><button type="button" className="btn btn-secondary" disabled={surveyBusy} onClick={() => setWellDialogMode(null)}>Cancel</button><button type="submit" className="btn btn-primary" disabled={surveyBusy || !wellNameInput.trim()}>{surveyBusy ? 'Saving…' : wellDialogMode === 'create' ? 'Create well' : 'Save changes'}</button></div>
      </form>
    </DialogFrame>}
    {toast && <div className="toast"><Check />{toast}</div>}
  </div>;
}
export default App;

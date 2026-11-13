export const API_BASE = '/api/v1';
export type SystemHealth = {
    status: string;
    service?: string;
};
export type BackendCapabilities = {
    pipeline_domains: {
        id: string;
        name: string;
        status: string;
        description: string;
    }[];
    dataset_kinds: string[];
    formats_supported_for_upload: string[];
    parsers: Record<string, string>;
    processing_status: Record<string, string>;
};
export type BackendRuntimeSettings = {
    ingestion: {
        max_upload_size_mb: number;
        allowed_formats: string[];
        validate_file_format: boolean;
    };
    processing: {
        default_timeout_seconds: number;
        default_runner: string;
        fallback_runner: string;
    };
    storage: {
        data_root: string;
        fallback_data_root: string;
    };
    geochat: {
        base_url: string;
        model: string;
        system_prompt: string;
        temperature: number;
        max_tokens: number;
        top_p: number;
        frequency_penalty: number;
        presence_penalty: number;
        timeout_seconds: number;
    };
    version: number;
};
export function getRuntimeSettings(): Promise<BackendRuntimeSettings> {
    return request<BackendRuntimeSettings>('/settings');
}
export function updateRuntimeSettings(payload: {
    ingestion: BackendRuntimeSettings['ingestion'];
    processing: BackendRuntimeSettings['processing'];
    storage: BackendRuntimeSettings['storage'];
    geochat: BackendRuntimeSettings['geochat'];
    expected_version: number;
}): Promise<BackendRuntimeSettings> {
    return request<BackendRuntimeSettings>('/settings', {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload),
    });
}
export type GeoChatMessage = { role: 'user' | 'assistant'; content: string };
export type GeoChatResponse = {
    conversation_id: string;
    skill?: string | null;
    ui_action?: 'open_upload' | 'view_dataset' | 'open_seismic_viewer' | 'open_well_viewer' | 'watch_processing_job' | 'switch_project' | null;
    ui_action_data?: Record<string, unknown> | null;
    message: GeoChatMessage;
    model: string;
    created_at: string;
    usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
    suggested_questions?: string[];
};
export type GeoChatConversation = {
    conversation_id: string;
    project_id: string;
    title: string;
    created_at: string;
    updated_at: string;
    request_count: number;
    model: string | null;
};
export type GeoChatConversationDetail = GeoChatConversation & { messages: GeoChatMessage[] };
export function listGeoChatConversations(projectId: string): Promise<GeoChatConversation[]> {
    return request<GeoChatConversation[]>(`/geochat/conversations?project_id=${encodeURIComponent(projectId)}`);
}
export function getGeoChatConversation(conversationId: string, projectId: string): Promise<GeoChatConversationDetail> {
    return request<GeoChatConversationDetail>(`/geochat/conversations/${encodeURIComponent(conversationId)}?project_id=${encodeURIComponent(projectId)}`);
}
export function deleteGeoChatConversation(conversationId: string, projectId: string): Promise<void> {
    return request<void>(`/geochat/conversations/${encodeURIComponent(conversationId)}?project_id=${encodeURIComponent(projectId)}`, { method: 'DELETE' });
}
export function sendGeoChat(payload: {
    conversation_id: string;
    project_id: string;
    messages: GeoChatMessage[];
}): Promise<GeoChatResponse> {
    return request<GeoChatResponse>('/geochat', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload),
    });
}
export type PipelineOperationError = {
    detail?: {
        code?: string;
        stage?: string;
        operation?: string;
        message?: string;
    } | string;
};
/** Invoke the backend's reserved QC endpoint. Current backend versions return HTTP 501. */
export function runDatasetQualityChecks(datasetId: string): Promise<unknown> {
    return request<unknown>(`/datasets/${encodeURIComponent(datasetId)}/quality-checks`, { method: 'POST' });
}
/** Invoke the backend's reserved feature-generation endpoint. Current backend versions return HTTP 501. */
export function generateDatasetFeatures(datasetId: string): Promise<unknown> {
    return request<unknown>(`/datasets/${encodeURIComponent(datasetId)}/features`, { method: 'POST' });
}
/** Invoke the backend's reserved ML processing-job endpoint. Current backend versions return HTTP 501. */
export function submitMLProcessingJob(): Promise<unknown> {
    return request<unknown>('/processing-jobs', { method: 'POST' });
}
export type BackendSurvey = {
    id: string;
    project_id: string;
    name: string;
    description: string | null;
    crs: string | null;
    metadata: Record<string, unknown>;
    created_at: string;
};
export type BackendProject = {
    id: string;
    name: string;
    description: string | null;
    created_at: string;
};
export type BackendDataset = {
    id: string;
    project_id: string | null;
    survey_id: string | null;
    well_id: string | null;
    kind: 'seismic' | 'well_log' | 'checkshot' | 'deviation' | string;
    dataset_type: string;
    name: string;
    original_filename: string;
    format: string;
    crs: string | null;
    size_bytes: number;
    checksum_sha256: string;
    storage_mode?: 'managed' | 'linked' | string;
    status: string;
    metadata: Record<string, unknown>;
    validation: Record<string, unknown>;
    created_at: string;
};
type DatasetList = {
    count: number;
    limit: number;
    offset: number;
    items: BackendDataset[];
};
async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, init);
    if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
            const body = await response.json() as {
                detail?: unknown;
                message?: unknown;
            };
            if (typeof body.detail === 'string')
                detail = body.detail;
            else if (typeof body.message === 'string')
                detail = body.message;
            else if (body.detail !== undefined)
                detail = JSON.stringify(body.detail);
        }
        catch { /* response was not JSON */ }
        throw new Error(detail);
    }
    if (response.status === 204)
        return undefined as T;
    return response.json() as Promise<T>;
}
async function systemRequest<T>(path: string): Promise<T> {
    const response = await fetch(path, { headers: { accept: 'application/json' } });
    if (!response.ok)
        throw new Error(`${path} returned ${response.status} ${response.statusText}`);
    return response.json() as Promise<T>;
}
export function getRootHealth(): Promise<SystemHealth> {
    return systemRequest<SystemHealth>('/health');
}
export function getVersionedHealth(): Promise<SystemHealth> {
    return request<SystemHealth>('/health');
}
export function getCapabilities(): Promise<BackendCapabilities> {
    return request<BackendCapabilities>('/capabilities');
}
export async function checkBackend(): Promise<void> {
    const health = await getRootHealth();
    if (health.status.toLowerCase() !== 'ok')
        throw new Error(`Unexpected backend health status: ${health.status}`);
}
export function listProjects(): Promise<BackendProject[]> {
    return request<BackendProject[]>('/projects');
}
export function getProject(projectId: string): Promise<BackendProject> {
    return request<BackendProject>(`/projects/${encodeURIComponent(projectId)}`);
}
export function createProject(name: string, description: string | null = 'frontend workspace'): Promise<BackendProject> {
    return request<BackendProject>('/projects', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name, description }),
    });
}
export function updateProject(projectId: string, patch: {
    name?: string;
    description?: string | null;
}): Promise<BackendProject> {
    return request<BackendProject>(`/projects/${encodeURIComponent(projectId)}`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(patch),
    });
}
export function deleteProject(projectId: string): Promise<void> {
    return request<void>(`/projects/${encodeURIComponent(projectId)}`, { method: 'DELETE' });
}
export function listSurveys(projectId: string): Promise<BackendSurvey[]> {
    return request<BackendSurvey[]>(`/surveys?project_id=${encodeURIComponent(projectId)}`);
}
export function getSurvey(surveyId: string): Promise<BackendSurvey> {
    return request<BackendSurvey>(`/surveys/${encodeURIComponent(surveyId)}`);
}
export function createSurvey(projectId: string, payload: {
    name: string;
    description: string | null;
    crs: string | null;
    metadata: Record<string, unknown>;
}): Promise<BackendSurvey> {
    return request<BackendSurvey>(`/projects/${encodeURIComponent(projectId)}/surveys`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload),
    });
}
export function updateSurvey(surveyId: string, patch: {
    name?: string;
    description?: string | null;
    crs?: string | null;
    metadata?: Record<string, unknown>;
}): Promise<BackendSurvey> {
    return request<BackendSurvey>(`/surveys/${encodeURIComponent(surveyId)}`, {
        method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify(patch),
    });
}
export function deleteSurvey(surveyId: string): Promise<void> {
    return request<void>(`/surveys/${encodeURIComponent(surveyId)}`, { method: 'DELETE' });
}
export function listSurveyDatasets(surveyId: string): Promise<BackendDataset[]> {
    return request<BackendDataset[]>(`/surveys/${encodeURIComponent(surveyId)}/datasets`);
}
export async function listDatasets(projectId: string): Promise<BackendDataset[]> {
    const result = await request<DatasetList>(`/datasets?project_id=${encodeURIComponent(projectId)}&limit=200`);
    return result.items;
}
export function updateDataset(datasetId: string, patch: {
    name?: string;
    project_id?: string | null;
    survey_id?: string | null;
    well_id?: string | null;
    crs?: string | null;
    metadata?: Record<string, unknown>;
}): Promise<BackendDataset> {
    return request<BackendDataset>(`/datasets/${encodeURIComponent(datasetId)}`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(patch),
    });
}
export function deleteDataset(datasetId: string): Promise<void> {
    return request<void>(`/datasets/${encodeURIComponent(datasetId)}`, { method: 'DELETE' });
}
export type BackendWell = {
    id: string;
    project_id: string;
    name: string;
    uwi: string | null;
    x: number | null;
    y: number | null;
    rt: number | null;
    td: number | null;
    crs: string | null;
    metadata: Record<string, unknown>;
    created_at: string;
};
export type WellListImportResult = {
    dataset: BackendDataset;
    summary: {
        rows_read: number;
        created: number;
        updated: number;
        skipped: number;
        errors: { row: number; well: string; error: string }[];
    };
};
export function importWellList(projectId: string, file: File): Promise<WellListImportResult> {
    const form = new FormData();
    form.append('file', file, file.name);
    return request<WellListImportResult>(`/projects/${encodeURIComponent(projectId)}/wells/import`, {
        method: 'POST',
        body: form,
    });
}
export function listWells(projectId: string): Promise<BackendWell[]> {
    return request<BackendWell[]>(`/wells?project_id=${encodeURIComponent(projectId)}`);
}
export function getWell(wellId: string): Promise<BackendWell> {
    return request<BackendWell>(`/wells/${encodeURIComponent(wellId)}`);
}
export function createWell(projectId: string, payload: {
    name: string;
    uwi: string | null;
    x: number | null;
    y: number | null;
    rt?: number | null;
    td?: number | null;
    crs: string | null;
    metadata: Record<string, unknown>;
}): Promise<BackendWell> {
    return request<BackendWell>(`/projects/${encodeURIComponent(projectId)}/wells`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload) });
}
export function updateWell(wellId: string, patch: {
    name?: string;
    uwi?: string | null;
    x?: number | null;
    y?: number | null;
    rt?: number | null;
    td?: number | null;
    crs?: string | null;
    metadata?: Record<string, unknown>;
}): Promise<BackendWell> {
    return request<BackendWell>(`/wells/${encodeURIComponent(wellId)}`, { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify(patch) });
}
export function deleteWell(wellId: string): Promise<void> {
    return request<void>(`/wells/${encodeURIComponent(wellId)}`, { method: 'DELETE' });
}
export function uploadDataset(file: File, projectId: string, kind: 'seismic' | 'well_log', associationId?: string): Promise<BackendDataset> {
    return uploadDatasetAdvanced(file, {
        project_id: projectId,
        kind,
        ...(kind === 'seismic' ? (associationId ? { survey_id: associationId } : {}) : (associationId ? { well_id: associationId } : {})),
    });
}
export async function downloadDataset(datasetId: string): Promise<Blob> {
    const response = await fetch(`${API_BASE}/datasets/${encodeURIComponent(datasetId)}/download`);
    if (!response.ok) {
        let detail = `Could not download dataset (${response.status})`;
        try {
            const body = await response.json() as {
                detail?: string;
            };
            if (body.detail)
                detail = body.detail;
        }
        catch { /* response was not JSON */ }
        throw new Error(detail);
    }
    return response.blob();
}
export type BackendDatasetHeader = {
    dataset_id: string;
    filename: string;
    stored_format: string;
    stored_crs: string | null;
    header: unknown;
};
export type BackendDatasetValidation = {
    dataset_id: string;
    status: string;
    validation: Record<string, unknown>;
};
export type BackendCurve = {
    id: string;
    dataset_id: string;
    mnemonic: string;
    unit: string | null;
    description: string | null;
    position: number;
    is_index: boolean;
    metadata: Record<string, unknown>;
    created_at: string;
};
export type BackendCurveSamples = {
    curve_id: string;
    dataset_id?: string;
    mnemonic?: string;
    unit?: string | null;
    total_samples?: number;
    offset?: number;
    limit?: number;
    returned_samples?: number;
    samples?: {
        index: number;
        value: number | null;
    }[];
};
export type BackendDatasetTable = {
    dataset_id: string;
    kind: 'deviation' | 'checkshot' | string;
    columns: {
        name: string;
        source_name: string;
        unit: string | null;
    }[];
    row_count: number;
    offset: number;
    limit: number;
    returned_rows: number;
    rows: Record<string, number>[];
};
export function getDataset(datasetId: string): Promise<BackendDataset> {
    return request<BackendDataset>(`/datasets/${encodeURIComponent(datasetId)}`);
}
export async function listDatasetsFiltered(filters: {
    project_id?: string;
    kind?: string;
    survey_id?: string;
    well_id?: string;
    limit?: number;
    offset?: number;
} = {}): Promise<DatasetList> {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => { if (value !== undefined && value !== null && value !== '')
        params.set(key, String(value)); });
    return request<DatasetList>(`/datasets?${params.toString()}`);
}
export function getDatasetHeader(datasetId: string): Promise<BackendDatasetHeader> {
    return request<BackendDatasetHeader>(`/datasets/${encodeURIComponent(datasetId)}/header`);
}
export function getDatasetValidation(datasetId: string): Promise<BackendDatasetValidation> {
    return request<BackendDatasetValidation>(`/datasets/${encodeURIComponent(datasetId)}/validation`);
}
export function getDatasetTable(datasetId: string, offset = 0, limit = 100): Promise<BackendDatasetTable> {
    return request<BackendDatasetTable>(`/datasets/${encodeURIComponent(datasetId)}/table?offset=${offset}&limit=${limit}`);
}
export function getDatasetCurves(datasetId: string): Promise<{
    dataset_id: string;
    count: number;
    items: BackendCurve[];
}> {
    return request<{
        dataset_id: string;
        count: number;
        items: BackendCurve[];
    }>(`/datasets/${encodeURIComponent(datasetId)}/curves`);
}
export function getCurveSamples(curveId: string, offset = 0, limit = 100): Promise<BackendCurveSamples> {
    return request<BackendCurveSamples>(`/curves/${encodeURIComponent(curveId)}/data?offset=${offset}&limit=${limit}`);
}
export type LinkLocalBrowserEntry = {
    name: string;
    path: string;
    kind: 'directory' | 'file';
    extension?: string;
};
export type LinkLocalBrowserResult = {
    current_path: string | null;
    parent_path: string | null;
    roots: { name: string; path: string }[];
    entries: LinkLocalBrowserEntry[];
};
export function browseLinkLocalDirectory(path?: string): Promise<LinkLocalBrowserResult> {
    const query = path ? `?path=${encodeURIComponent(path)}` : '';
    return request<LinkLocalBrowserResult>(`/datasets/link-local/browse${query}`);
}

export function linkLocalDataset(payload: {
    project_id: string;
    kind: 'seismic' | 'well_log' | 'deviation' | 'checkshot';
    source_path: string;
    name?: string;
}): Promise<BackendDataset> {
    return request<BackendDataset>('/datasets/link-local', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload),
    });
}
export function uploadDatasetAdvanced(file: File, payload: {
    project_id: string;
    kind: string;
    name?: string;
    survey_id?: string | null;
    well_id?: string | null;
    crs?: string | null;
    metadata?: Record<string, unknown>;
}): Promise<BackendDataset> {
    const form = new FormData();
    form.append('file', file, file.name);
    form.append('project_id', payload.project_id);
    form.append('kind', payload.kind);
    form.append('name', payload.name?.trim() || file.name);
    if (payload.survey_id)
        form.append('survey_id', payload.survey_id);
    if (payload.well_id)
        form.append('well_id', payload.well_id);
    if (payload.crs)
        form.append('crs', payload.crs);
    form.append('metadata_json', JSON.stringify(payload.metadata ?? {}));
    return request<BackendDataset>('/datasets', { method: 'POST', body: form });
}
export function getCurve(curveId: string): Promise<BackendCurve> {
    return request<BackendCurve>(`/curves/${encodeURIComponent(curveId)}`);
}


export type BackendDatasetData = {
    dataset_id: string;
    type: 'seismic' | 'well' | 'table' | string;
    [key: string]: unknown;
};
export function getDatasetData(datasetId: string, maxTraces = 500): Promise<BackendDatasetData> {
    return request<BackendDatasetData>(`/datasets/${encodeURIComponent(datasetId)}/data?max_traces=${maxTraces}`);
}
export type ModelReleaseModel = { task?: string | null; input?: string | null; model?: string | null; model_file?: string; final_params?: string | null; final_threshold?: string | null; scaling_mode?: string | null };
export type ModelReleaseListItem = {
    model_release_id: string; status: string; pipeline?: string | null; pipeline_version?: string | null;
    model_family?: string | null; processing_job_id?: string | null; training_run_id?: string | null;
    model_count: number; models: ModelReleaseModel[];
};
export type ModelReleaseList = { items: ModelReleaseListItem[]; count: number; limit: number; offset: number };
export type ModelReleaseDetail = { model_release_id: string; manifest: Record<string, unknown> };
export type ModelReleaseValidation = { model_release_id: string; valid: boolean; checked_files: number; errors: string[] };
export function listModelReleases(limit = 100, offset = 0): Promise<ModelReleaseList> {
    return request<ModelReleaseList>(`/model-releases?limit=${limit}&offset=${offset}`);
}
export function getModelRelease(releaseId: string): Promise<ModelReleaseDetail> {
    return request<ModelReleaseDetail>(`/model-releases/${encodeURIComponent(releaseId)}`);
}
export function validateModelRelease(releaseId: string): Promise<ModelReleaseValidation> {
    return request<ModelReleaseValidation>(`/model-releases/${encodeURIComponent(releaseId)}/validate`, { method: 'POST' });
}
export function deleteModelRelease(releaseId: string): Promise<void> {
    return request<void>(`/model-releases/${encodeURIComponent(releaseId)}`, { method: 'DELETE' });
}
export type ProcessingJobRequest = {
    python_script: string;
    shell_script?: string;
    dataset_ids: string[];
};
export type ProcessingJobResponse = {
    job_id: string;
    status: string;
    operation?: string;
    execution_mode?: string;
    dataset_ids?: string[];
    error_message?: string | null;
    [key: string]: unknown;
};
export function submitProcessingJob(payload: ProcessingJobRequest): Promise<ProcessingJobResponse> {
    return request<ProcessingJobResponse>('/processing-jobs', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload),
    });
}
export function deleteProcessingJob(jobId: string): Promise<void> {
    return request<void>(`/processing-jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
}

export function getProcessingJob(jobId: string): Promise<ProcessingJobResponse> {
    return request<ProcessingJobResponse>(`/processing-jobs/${encodeURIComponent(jobId)}`);
}

export type ProcessingJobListItem = ProcessingJobResponse & {
    id: string;
    job_id?: string;
    job_dir?: string;
    created_at?: string;
    python_script?: string | null;
    shell_script?: string | null;
};
export type ProcessingJobList = { items: ProcessingJobListItem[]; count: number; limit: number; offset: number };
export type ProcessingJobFile = { name: string; size_bytes: number; updated_at: number };
export type ProcessingJobFiles = { job_id: string; job_dir: string; items: ProcessingJobFile[]; count: number };
export type ProcessingJobFileContent = { job_id: string; name: string; size_bytes: number; content: string };
export type EvidenceArtifact = { name: string; size_bytes: number; updated_at: number; role: string };
export type EvidenceCatalogItem = { id: string; job_id: string; status: string; operation?: string | null; python_script?: string | null; target_well?: string | null; created_at?: string | null; summary?: string | null; artifacts: EvidenceArtifact[]; artifact_count: number; has_result: boolean };
export type EvidenceCatalog = { items: EvidenceCatalogItem[]; count: number; scanned_roots: string[] };
export function listEvidenceCatalog(): Promise<EvidenceCatalog> {
    return request<EvidenceCatalog>('/evidence/catalog');
}
export type EvidenceExplanation = { skill: string; job_id: string; python_script?: string | null; operation?: string | null; target_well?: string | null; status: string; created_at?: string | null; summary: string; llm_explanation?: string | null; findings: string[]; artifacts: { name: string; size_bytes: number; format: string; rows_scanned?: number; truncated?: boolean; columns?: string[]; numeric_fields?: Record<string, { count: number; min: number; max: number; mean: number }>; categorical_fields?: Record<string, Record<string, number>>; top_level_keys?: string[]; line_count?: number; preview?: string; type?: string; note?: string; skipped?: string }[]; skipped: { name: string; reason: string }[]; limitations: string[]; domain_analysis?: Record<string, unknown> | null; domain_interpretation_limits?: string[];
    /** Set when a process-specific output-analysis skill read the run: the same fixed report Ask Geo shows. */
    markdown?: string | null; applicable?: boolean; verdict?: { level?: string; plain?: string } | null; next_steps?: string[] };
export function explainEvidenceItem(jobId: string): Promise<EvidenceExplanation> {
    return request<EvidenceExplanation>(`/evidence/${encodeURIComponent(jobId)}/explain`, { method: 'POST' });
}
export type EvidenceComparisonStat = { job_id: string; count: number; mean: number; median: number; min: number; max: number; stddev: number };
export type EvidenceComparison = { file_name: string; columns: { name: string; runs: EvidenceComparisonStat[] }[] };
export type EvidenceComparisonResult = { python_script: string; runs: { job_id: string; created_at?: string | null }[]; comparisons: EvidenceComparison[]; skill_analysis?: Record<string, unknown> | null; method: string };
export function compareEvidenceResults(pythonScript: string, jobIds: string[]): Promise<EvidenceComparisonResult> {
    return request<EvidenceComparisonResult>('/evidence/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ python_script: pythonScript, job_ids: jobIds }),
    });
}
export function listProcessingJobs(limit = 100, offset = 0): Promise<ProcessingJobList> {
    return request<ProcessingJobList>(`/processing-jobs?limit=${limit}&offset=${offset}`);
}
export function listProcessingJobFiles(jobId: string): Promise<ProcessingJobFiles> {
    return request<ProcessingJobFiles>(`/processing-jobs/${encodeURIComponent(jobId)}/files`);
}
export function getProcessingJobFile(jobId: string, fileName: string): Promise<ProcessingJobFileContent> {
    return request<ProcessingJobFileContent>(`/processing-jobs/${encodeURIComponent(jobId)}/files/${encodeURIComponent(fileName)}`);
}


export type InferenceSliceResponse = {
    job_id: string; attribute: 'lithology' | 'vsh' | 'phit'; plane: 'inline' | 'crossline' | 'time';
    coordinate: number; coordinate_min: number; coordinate_step: number; coordinate_count: number; axes: { x: number[]; y: number[] }; shape: number[]; downsample_stride: number;
    values: number[][]; valid_mask: boolean[][]; valid_count: number; invalid_count: number;
    volume_shape: number[]; summary: Record<string, unknown>; geometry_flags: string[];
};
export function getInferenceSlice(jobId: string, attribute: 'lithology' | 'vsh' | 'phit',
    plane: 'inline' | 'crossline' | 'time', coordinate: number): Promise<InferenceSliceResponse> {
    const query = new URLSearchParams({ attribute, plane, coordinate: String(coordinate) });
    return request<InferenceSliceResponse>(`/processing-jobs/${encodeURIComponent(jobId)}/inference-slice?${query}`);
}

export type BackendScript = {
    id: string;
    name: string;
    size_bytes: number;
    updated_at: string;
    content?: string;
};
export type BackendScriptList = { items: BackendScript[]; count: number };
export function listProcessingScripts(): Promise<BackendScriptList> {
    return request<BackendScriptList>('/scripts');
}
export function getProcessingScript(scriptId: string): Promise<BackendScript> {
    return request<BackendScript>(`/scripts/${encodeURIComponent(scriptId)}`);
}
export async function uploadProcessingScript(file: File): Promise<BackendScript> {
    const form = new FormData();
    form.append('file', file);
    const response = await fetch(`${API_BASE}/scripts`, { method: 'POST', body: form });
    if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try { const body = await response.json() as { detail?: unknown }; if (typeof body.detail === 'string') detail = body.detail; } catch { /* ignore non-JSON response */ }
        throw new Error(detail);
    }
    return response.json() as Promise<BackendScript>;
}
export function updateProcessingScript(scriptId: string, content: string): Promise<BackendScript> {
    return request<BackendScript>(`/scripts/${encodeURIComponent(scriptId)}`, {
        method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ content }),
    });
}
export function deleteProcessingScript(scriptId: string): Promise<void> {
    return request<void>(`/scripts/${encodeURIComponent(scriptId)}`, { method: 'DELETE' });
}


// Stage 12 geobody read-only API.
export type GeobodyRecord = Record<string, string | number | boolean | null>;
export type GeobodyManifest = {
    schema_version?: number;
    stage?: string;
    status?: string;
    interpretation_status?: string;
    source?: Record<string, unknown>;
    geometry?: Record<string, unknown>;
    method?: Record<string, unknown>;
    counts?: Record<string, number>;
    products?: Record<string, unknown>[];
    exports?: {
        segy?: Record<string, unknown>[];
        meshes?: Record<string, unknown>[];
        visualisations?: Record<string, unknown>[];
    };
    api?: Record<string, unknown>;
    limitations?: string[];
};
export type GeobodyListResponse = {
    total: number;
    offset: number;
    limit: number;
    geobodies: GeobodyRecord[];
};
export type GeobodyDetailResponse = {
    geobody: GeobodyRecord;
    mesh: Record<string, unknown> | null;
    figures: { name: string; url: string; [key: string]: unknown }[];
    interpretation_status?: string;
    source?: Record<string, unknown>;
    limitations?: string[];
};
export type GeobodyVoxelPreview = { job_id: string; geobody_id: number; coordinate_order: string[]; voxel_count: number; preview_count: number; sampling_stride: number; bounds_ijk: { min: number[]; max: number[] }; coordinates: [number, number, number][]; is_sampled: boolean };
export function getGeobodyVoxelPreview(jobId: string, geobodyId: number, maxPoints = 30000): Promise<GeobodyVoxelPreview> {
    return request<GeobodyVoxelPreview>(`${geobodyPath(jobId)}/items/${geobodyId}/voxel-preview?max_points=${maxPoints}`);
}
export type GeobodySliceResponse = {
    geobody_id: number;
    plane: 'inline' | 'crossline' | 'time';
    coordinate: number;
    axes: string[];
    downsample_stride: number;
    shape: number[];
    body_cells: number;
    mask: number[][];
    interpretation_status?: string;
};
const geobodyPath = (jobId: string) => `/geobodies/jobs/${encodeURIComponent(jobId)}`;
export function getGeobodyManifest(jobId: string): Promise<GeobodyManifest> {
    return request<GeobodyManifest>(geobodyPath(jobId));
}
export function listGeobodies(jobId: string, limit = 200, offset = 0): Promise<GeobodyListResponse> {
    return request<GeobodyListResponse>(`${geobodyPath(jobId)}/items?limit=${limit}&offset=${offset}`);
}
export function getGeobody(jobId: string, geobodyId: number): Promise<GeobodyDetailResponse> {
    return request<GeobodyDetailResponse>(`${geobodyPath(jobId)}/items/${geobodyId}`);
}
export function getGeobodySlice(jobId: string, geobodyId: number, plane: 'inline' | 'crossline' | 'time', coordinate?: number): Promise<GeobodySliceResponse> {
    const query = new URLSearchParams({ plane });
    if (coordinate !== undefined && Number.isFinite(coordinate)) query.set('coordinate', String(coordinate));
    return request<GeobodySliceResponse>(`${geobodyPath(jobId)}/items/${geobodyId}/slice?${query.toString()}`);
}
export function geobodyAssetUrl(jobId: string, assetPath: string): string {
    return `${API_BASE}${geobodyPath(jobId)}${assetPath}`;
}

export type BackendSeismicGeometry = {
    dataset_id: string;
    trace_count: number;
    samples_per_trace: number;
    sample_interval_us: number;
    format_code: number;
    delay_ms: number;
    amplitude_scale: number;
    inlines: number[];
    crosslines: number[];
    inline_range: [number, number];
    crossline_range: [number, number];
    geometry_confidence: 'High' | 'Medium' | 'Low';
    geometry_candidates: Array<{ label: string; inline_byte: number; crossline_byte: number; score: number; inline_range: [number, number]; crossline_range: [number, number] }>;
    active_inline_byte: number;
    active_crossline_byte: number;
};
export type BackendSeismicSection = {
    dataset_id: string;
    mode: 'inline' | 'crossline' | 'timeslice';
    target: number;
    traces: (number | null)[][];
    x_positions: number[];
    y_positions?: number[];
    present: number;
    total: number;
    duplicates: number;
    sample_count: number;
    sample_stride?: number;
    sample_interval_us: number;
    start_ms: number;
    amplitude_scale: number;
};
export function getSeismicGeometry(datasetId: string, inlineByte?: number, crosslineByte?: number): Promise<BackendSeismicGeometry> {
    const query = inlineByte && crosslineByte ? `?inline_byte=${inlineByte}&crossline_byte=${crosslineByte}` : '';
    return request<BackendSeismicGeometry>(`/datasets/${encodeURIComponent(datasetId)}/seismic/geometry${query}`);
}
export function getSeismicSection(datasetId: string, params: {
    mode: 'inline' | 'crossline' | 'timeslice'; target: number; inlineByte: number; crosslineByte: number;
}): Promise<BackendSeismicSection> {
    const query = new URLSearchParams({ mode: params.mode, target: String(params.target), inline_byte: String(params.inlineByte), crossline_byte: String(params.crosslineByte) });
    return request<BackendSeismicSection>(`/datasets/${encodeURIComponent(datasetId)}/seismic/section?${query.toString()}`);
}


export type ProcessingConfigValue = null | string | number | boolean | ProcessingConfigValue[] | { [key: string]: ProcessingConfigValue };
export type BackendProcessingConfig = {
    id: string;
    name: string;
    content: string;
    config: { [key: string]: ProcessingConfigValue };
    revision: string;
};
export function getProcessingConfig(configId: string): Promise<BackendProcessingConfig> {
    return request<BackendProcessingConfig>(`/processing-configs/${encodeURIComponent(configId)}`);
}
export function updateProcessingConfig(configId: string, config: BackendProcessingConfig['config'], expectedRevision: string): Promise<BackendProcessingConfig> {
    return request<BackendProcessingConfig>(`/processing-configs/${encodeURIComponent(configId)}`, {
        method: 'PUT',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ config, expected_revision: expectedRevision }),
    });
}

export type VolumeProductArtifact = { path: string; name: string; suffix: string; size_bytes: number; category: "summary" | "export" };
export type VolumeProductArtifactList = { job_id: string; items: VolumeProductArtifact[]; count: number };
export type VolumeProductArtifactContent = { job_id: string; path: string; name: string; size_bytes: number; content: string };
export function listVolumeProductArtifacts(jobId: string): Promise<VolumeProductArtifactList> {
    return request<VolumeProductArtifactList>(`/processing-jobs/${encodeURIComponent(jobId)}/volume-products`);
}
export function getVolumeProductArtifact(jobId: string, path: string): Promise<VolumeProductArtifactContent> {
    const query = new URLSearchParams({ path });
    return request<VolumeProductArtifactContent>(`/processing-jobs/${encodeURIComponent(jobId)}/volume-products/file?${query}`);
}
export function volumeProductArtifactUrl(jobId: string, path: string): string {
    const query = new URLSearchParams({ path });
    return `${API_BASE}/processing-jobs/${encodeURIComponent(jobId)}/volume-products/file?${query}`;
}

export type StageResultArtifact = { path: string; name: string; suffix: string; size_bytes: number; category: "summary" | "table" | "figure" | "export" };
export type StageResultArtifactList = { job_id: string; stage: "well-geobody" | "prospect" | "foundations"; items: StageResultArtifact[]; count: number };
export type StageResultArtifactContent = { job_id: string; stage: "well-geobody" | "prospect" | "foundations"; path: string; name: string; size_bytes: number; content: string };
export function listStageResultArtifacts(jobId: string, stage: "well-geobody" | "prospect" | "foundations"): Promise<StageResultArtifactList> {
    return request<StageResultArtifactList>(`/processing-jobs/${encodeURIComponent(jobId)}/stage-results?stage=${stage}`);
}
export function getStageResultArtifact(jobId: string, stage: "well-geobody" | "prospect" | "foundations", path: string): Promise<StageResultArtifactContent> {
    const query = new URLSearchParams({ stage, path });
    return request<StageResultArtifactContent>(`/processing-jobs/${encodeURIComponent(jobId)}/stage-results/file?${query}`);
}
export function stageResultArtifactUrl(jobId: string, stage: "well-geobody" | "prospect" | "foundations", path: string): string {
    const query = new URLSearchParams({ stage, path });
    return `${API_BASE}/processing-jobs/${encodeURIComponent(jobId)}/stage-results/file?${query}`;
}

export type ProspectVoxelPreview = { job_id: string; prospect_id: string; coordinate_order: string[]; voxel_count: number; preview_count: number; sampling_stride: number; bounds_ijk: { min: number[]; max: number[] }; coordinates: number[][]; is_sampled: boolean };
export function getProspectVoxelPreview(jobId: string, prospectId: string): Promise<ProspectVoxelPreview> {
    return request<ProspectVoxelPreview>(`/processing-jobs/${encodeURIComponent(jobId)}/prospect-voxel-preview?prospect_id=${encodeURIComponent(prospectId)}`);
}

export type Well3DEntry = { well: string; html: string[]; images: string[]; tables: string[]; others: string[]; file_count: number };
export type Well3DCatalog = { items: Well3DEntry[]; count: number; available: boolean };
export type Well3DTablePreview = { name: string; columns: string[]; rows: string[][]; offset: number; limit: number; total_rows: number };
export function listWell3DWells(): Promise<Well3DCatalog> { return request<Well3DCatalog>('/evidence/well-3d/wells'); }
export function getWell3DTable(well: string, path: string, offset = 0, limit = 100): Promise<Well3DTablePreview> {
  return request<Well3DTablePreview>(`/evidence/well-3d/${encodeURIComponent(well)}/table?path=${encodeURIComponent(path)}&offset=${offset}&limit=${limit}`);
}
export function well3DAssetUrl(well: string, path: string): string {
  return `${API_BASE}/evidence/well-3d/${encodeURIComponent(well)}/file?path=${encodeURIComponent(path)}`;
}

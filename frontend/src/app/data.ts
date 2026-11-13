import type { BackendDataset } from '../api/backend';

export type DataKind = 'Seismic' | 'Well log' | 'Table';
export type SeismicData = {
    type: 'seismic';
    traces: number[][];
    traceHeaders: Array<{
        inline: number;
        crossline: number;
    }>;
    traceCount: number;
    scannedTraceCount: number;
    samplesPerTrace: number;
    sampleIntervalUs: number;
    formatCode: number;
    inlineRange?: [
        number,
        number
    ];
    crosslineRange?: [
        number,
        number
    ];
    textualHeader: string;
};
export type WellData = {
    type: 'well';
    curves: string[];
    units: string[];
    rows: number[][];
    depthStart?: number;
    depthStop?: number;
    nullValue?: number;
};
export type TableData = {
    type: 'table';
    headers: string[];
    rows: string[][];
    rowCount: number;
    delimiter: string;
};
export type ParsedData = SeismicData | WellData | TableData;
export type DataItem = {
    id: string;
    surveyId?: string | null;
    wellId?: string | null;
    storageMode?: 'managed' | 'linked';
    name: string;
    kind: DataKind;
    format: string;
    size: string;
    status: 'Scanning' | 'Ready' | 'Review';
    details: string;
    parsed?: ParsedData;
    issue?: string;
};

export type WellTableKind = 'deviation' | 'checkshot';

/** True for TXT datasets that use the tabular well-data workflow. */
export const isTextTableDataset = (item: DataItem): boolean =>
    item.kind === 'Table' && item.format.trim().toLowerCase() === 'txt';

/** Well viewers accept LAS logs and TXT deviation/checkshot datasets. */
export const isWellDataset = (item: DataItem): boolean =>
    item.kind === 'Well log' || isTextTableDataset(item);

/** Identify the specialized backend parser used for deviation/checkshot TXT files. */
export const getWellTableKind = (item: DataItem): WellTableKind | null => {
    if (!isTextTableDataset(item)) return null;
    const description = `${item.name} ${item.details}`;
    if (/checkshot/i.test(description)) return 'checkshot';
    if (/deviation/i.test(description)) return 'deviation';
    return null;
};

/** Whether the selected dataset has a supported destination viewer. */
export const canOpenDatasetViewer = (item: DataItem): boolean =>
    item.status === 'Ready' && (item.kind !== 'Table' || isTextTableDataset(item) || ['xlsx', 'xlsm'].includes(item.format.toLowerCase()));
export const identifyFile = (name: string): DataKind | null => {
    const ext = name.toLowerCase().split('.').pop();
    if (ext === 'sgy' || ext === 'segy')
        return 'Seismic';
    if (ext === 'las')
        return 'Well log';
    if (ext === 'csv' || ext === 'tsv' || ext === 'txt' || ext === 'xlsx' || ext === 'xlsm')
        return 'Table';
    return null;
};


/** Format byte counts consistently in dataset inventory rows. */
export const sizeLabel = (bytes: number): string =>
    `${Math.max(0.1, bytes / 1024 / 1024).toFixed(1)} MB`;

/** Convert supported backend dataset records into the frontend inventory model. */
export const fromBackendDataset = (dataset: BackendDataset): DataItem | null => {
    const format = dataset.format.toLowerCase().replace(/^\./, '');
    const kind: DataKind | null = dataset.kind === 'seismic' || dataset.kind === 'inversion'
        ? 'Seismic'
        : dataset.kind === 'well_log'
            ? 'Well log'
            : dataset.kind === 'deviation' || dataset.kind === 'checkshot' || dataset.kind === 'table'
                ? 'Table'
                : null;

    if (!kind || !['sgy', 'segy', 'las', 'txt', 'xlsx', 'xlsm'].includes(format)) return null;

    return {
        id: dataset.id,
        surveyId: dataset.survey_id,
        wellId: dataset.well_id,
        storageMode: dataset.storage_mode === 'linked' ? 'linked' : 'managed',
        name: dataset.original_filename || dataset.name,
        kind,
        format: format.toUpperCase(),
        size: sizeLabel(dataset.size_bytes),
        status: dataset.status.toLowerCase() === 'ready' ? 'Ready' : 'Review',
        details: dataset.kind === 'table' && dataset.metadata?.table_type === 'well_list'
            ? `${dataset.storage_mode === 'linked' ? 'Linked local' : 'Saved in backend'} · Well list (RT / TD / X / Y)`
            : `${dataset.storage_mode === 'linked' ? 'Linked local' : 'Saved in backend'} · ${dataset.kind}`, 
    };
};

/** Identify SEG-Y datasets even when the stored format includes a leading dot. */
export const isSegyDataset = (item: DataItem): boolean => {
    const format = item.format.toLowerCase().replace(/[^a-z0-9]/g, '');
    return item.kind === 'Seismic'
        && (['sgy', 'segy'].includes(format) || /\.(sgy|segy)$/i.test(item.name));
};

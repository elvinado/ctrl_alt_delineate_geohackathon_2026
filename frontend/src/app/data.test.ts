import { describe, expect, it } from 'vitest';
import type { DataItem } from './data';
import { canOpenDatasetViewer, fromBackendDataset, getWellTableKind, identifyFile, isWellDataset } from './data';
describe('geological file identification', () => {
    it('identifies seismic, well and table formats case-insensitively', () => {
        expect(identifyFile('cube.SGY')).toBe('Seismic');
        expect(identifyFile('well.las')).toBe('Well log');
        expect(identifyFile('markers.csv')).toBe('Table');
    });
    it('rejects unsupported formats', () => expect(identifyFile('notes.pdf')).toBeNull());
});



describe('dataset viewer classification', () => {
    const dataset = (overrides: Partial<DataItem> = {}): DataItem => ({
        id: 'dataset-1',
        name: 'well_deviation.txt',
        kind: 'Table',
        format: 'TXT',
        size: '1.0 MB',
        status: 'Ready',
        details: 'Saved in backend · deviation',
        ...overrides,
    });

    it('classifies deviation and checkshot TXT datasets for the well workflow', () => {
        expect(isWellDataset(dataset())).toBe(true);
        expect(getWellTableKind(dataset())).toBe('deviation');
        expect(getWellTableKind(dataset({ name: 'survey_checkshot.TXT', details: 'Saved in backend' }))).toBe('checkshot');
    });

    it('does not classify unrelated tables as specialized well tables', () => {
        expect(getWellTableKind(dataset({ name: 'facies.txt', details: 'Saved in backend' }))).toBeNull();
    });

    it('only enables viewers for ready datasets with supported destinations', () => {
        expect(canOpenDatasetViewer(dataset())).toBe(true);
        expect(canOpenDatasetViewer(dataset({ format: 'CSV' }))).toBe(false);
        expect(canOpenDatasetViewer(dataset({ status: 'Review' }))).toBe(false);
        expect(canOpenDatasetViewer(dataset({ kind: 'Well log', format: 'LAS' }))).toBe(true);
    });
});


describe('registered well-list datasets', () => {
    it('shows backend table XLSX well lists in the inventory and enables preview', () => {
        const item = fromBackendDataset({
            id: 'well-list-1', project_id: 'project-1', kind: 'table', format: 'xlsx',
            name: 'Challenge_WellList.xlsx', original_filename: 'Challenge_WellList.xlsx',
            size_bytes: 1024, status: 'ready', storage_mode: 'managed',
            metadata: { table_type: 'well_list', imported_columns: ['name', 'rt', 'td', 'x', 'y'] },
        } as never);
        expect(item?.kind).toBe('Table');
        expect(item?.details).toContain('Well list');
        expect(item && canOpenDatasetViewer(item)).toBe(true);
    });
});

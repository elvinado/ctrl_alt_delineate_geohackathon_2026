import type { DataItem, SeismicData, TableData, WellData } from '../app/data';
import { identifyFile } from '../app/data';
const MAX_TEXT_BYTES = 16 * 1024 * 1024;
const MAX_SCANNED_TRACES = 6000;
const MAX_SCAN_BYTES = 48 * 1024 * 1024;
const PREVIEW_SAMPLES = 260;
const humanSize = (bytes: number) => bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(2)} GB`
    : bytes >= 1024 ** 2 ? `${(bytes / 1024 ** 2).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
const sampleBytes: Record<number, number> = { 1: 4, 2: 4, 3: 2, 5: 4, 6: 8, 8: 1 };
const ibmFloat = (word: number) => {
    if (word === 0)
        return 0;
    const sign = (word >>> 31) ? -1 : 1;
    const exponent = ((word >>> 24) & 0x7f) - 64;
    const fraction = (word & 0x00ffffff) / 0x01000000;
    return sign * fraction * Math.pow(16, exponent);
};
const readSample = (view: DataView, offset: number, format: number) => {
    if (format === 1)
        return ibmFloat(view.getUint32(offset, false));
    if (format === 2)
        return view.getInt32(offset, false);
    if (format === 3)
        return view.getInt16(offset, false);
    if (format === 5)
        return view.getFloat32(offset, false);
    if (format === 6)
        return view.getFloat64(offset, false);
    if (format === 8)
        return view.getInt8(offset);
    return 0;
};
const printableHeader = (bytes: Uint8Array) => {
    const asciiRatio = bytes.reduce((n, b) => n + ((b >= 32 && b <= 126) || b === 10 || b === 13 ? 1 : 0), 0) / bytes.length;
    if (asciiRatio < .7)
        return 'EBCDIC textual header detected (raw conversion deferred)';
    return new TextDecoder('ascii').decode(bytes).replace(/\0/g, ' ').trim();
};
const normalizeTrace = (values: number[]) => {
    const finite = values.filter(Number.isFinite);
    if (!finite.length)
        return values.map(() => 0);
    const abs = finite.map(Math.abs).sort((a, b) => a - b);
    const scale = abs[Math.floor(abs.length * .98)] || 1;
    return values.map(v => Number.isFinite(v) ? Math.max(-1, Math.min(1, v / scale)) : 0);
};
export async function parseSegyBlob(blob: Blob): Promise<SeismicData> {
    if (blob.size < 3600)
        throw new Error('File is smaller than the 3,600-byte SEG-Y header');
    const header = await blob.slice(0, 3600).arrayBuffer();
    const view = new DataView(header);
    const sampleIntervalUs = view.getUint16(3216, false);
    const samplesPerTrace = view.getUint16(3220, false);
    const formatCode = view.getUint16(3224, false);
    const bps = sampleBytes[formatCode];
    if (!samplesPerTrace || samplesPerTrace > 200000)
        throw new Error('Invalid samples-per-trace in binary header');
    if (!bps)
        throw new Error(`SEG-Y sample format ${formatCode} is not supported yet`);
    const traceBytes = 240 + samplesPerTrace * bps;
    const traceCount = Math.floor((blob.size - 3600) / traceBytes);
    if (traceCount < 1)
        throw new Error('No complete traces found using fixed-length trace geometry');
    const scannedTraceCount = Math.min(MAX_SCANNED_TRACES, traceCount, Math.max(1, Math.floor(MAX_SCAN_BYTES / traceBytes)));
    // Read one contiguous window. This preserves inline/crossline neighbourhoods and
    // avoids thousands of independent file reads while bounding browser memory.
    const traceBuffer = await blob.slice(3600, 3600 + scannedTraceCount * traceBytes).arrayBuffer();
    const traceView = new DataView(traceBuffer);
    const inlineValues: number[] = [];
    const crosslineValues: number[] = [];
    const traceHeaders: Array<{
        inline: number;
        crossline: number;
    }> = [];
    const traces = Array.from({ length: scannedTraceCount }, (_, traceIndex) => {
        const offset = traceIndex * traceBytes;
        const inline = traceView.getInt32(offset + 188, false), crossline = traceView.getInt32(offset + 192, false);
        inlineValues.push(inline);
        crosslineValues.push(crossline);
        traceHeaders.push({ inline, crossline });
        const stride = Math.max(1, Math.floor(samplesPerTrace / PREVIEW_SAMPLES));
        const values: number[] = [];
        for (let sample = 0; sample < samplesPerTrace && values.length < PREVIEW_SAMPLES; sample += stride) {
            values.push(readSample(traceView, offset + 240 + sample * bps, formatCode));
        }
        return normalizeTrace(values);
    });
    const usefulRange = (values: number[]): [
        number,
        number
    ] | undefined => {
        const valid = values.filter(v => v !== 0 && Number.isFinite(v));
        return valid.length ? [Math.min(...valid), Math.max(...valid)] : undefined;
    };
    return {
        type: 'seismic', traces, traceHeaders, traceCount, scannedTraceCount, samplesPerTrace, sampleIntervalUs, formatCode,
        inlineRange: usefulRange(inlineValues), crosslineRange: usefulRange(crosslineValues),
        textualHeader: printableHeader(new Uint8Array(header, 0, 3200)),
    };
}
const splitDelimited = (line: string, delimiter: string) => {
    const cells: string[] = [];
    let cell = '';
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === '"' && line[i + 1] === '"' && quoted) {
            cell += '"';
            i++;
        }
        else if (c === '"')
            quoted = !quoted;
        else if (c === delimiter && !quoted) {
            cells.push(cell.trim());
            cell = '';
        }
        else
            cell += c;
    }
    cells.push(cell.trim());
    return cells;
};
export function parseTableText(text: string): TableData {
    const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter(line => line.trim()).slice(0, 5001);
    if (!lines.length)
        throw new Error('Table is empty');
    const candidates = [',', '\t', ';', '|'];
    const delimiter = candidates.sort((a, b) => lines[0].split(b).length - lines[0].split(a).length)[0];
    const headers = splitDelimited(lines[0], delimiter);
    if (headers.length < 2)
        throw new Error('Could not detect multiple table columns');
    return { type: 'table', headers, rows: lines.slice(1).map(line => splitDelimited(line, delimiter)), rowCount: Math.max(0, text.split(/\r?\n/).filter(Boolean).length - 1), delimiter: delimiter === '\t' ? 'Tab' : delimiter };
}
export function parseLasText(text: string): WellData {
    const lines = text.split(/\r?\n/);
    const curves: string[] = [];
    const units: string[] = [];
    let section = '';
    let nullValue: number | undefined;
    const rows: number[][] = [];
    for (const raw of lines) {
        const line = raw.trim();
        if (!line || line.startsWith('#'))
            continue;
        if (line.startsWith('~')) {
            section = line.slice(1, 2).toUpperCase();
            continue;
        }
        if (section === 'W' && /^NULL\s*\./i.test(line))
            nullValue = Number(line.split(/\s+/).find(x => /^-?\d/.test(x)));
        if (section === 'C') {
            const match = line.match(/^([^\.\s]+)\s*\.\s*([^\s:]*)/);
            if (match) {
                curves.push(match[1]);
                units.push(match[2] || '—');
            }
        }
        if (section === 'A') {
            const values = line.split(/[\s,]+/).map(Number);
            if (values.length && values.every(v => Number.isFinite(v)))
                rows.push(values);
        }
    }
    if (!curves.length || !rows.length)
        throw new Error('LAS curve definitions or ASCII log data are missing');
    const sampled = rows.length > 1600 ? rows.filter((_, i) => i % Math.ceil(rows.length / 1600) === 0) : rows;
    return { type: 'well', curves, units, rows: sampled, depthStart: rows[0]?.[0], depthStop: rows.at(-1)?.[0], nullValue };
}
export async function parseGeologicalFile(file: File, id = `u-${Date.now()}`): Promise<DataItem> {
    const kind = identifyFile(file.name);
    if (!kind)
        throw new Error('Supported formats are SEG-Y, LAS, CSV, and TSV');
    if (kind === 'Seismic') {
        const parsed = await parseSegyBlob(file);
        const ranges = parsed.inlineRange ? `IL ${parsed.inlineRange[0]}–${parsed.inlineRange[1]}` : `${parsed.traceCount.toLocaleString()} traces`;
        return { id, name: file.name, kind, format: `SEG-Y · format ${parsed.formatCode}`, size: humanSize(file.size), status: 'Ready', details: `${ranges} · ${parsed.samplesPerTrace} samples · ${parsed.sampleIntervalUs} µs`, parsed };
    }
    const text = await file.slice(0, MAX_TEXT_BYTES).text();
    if (kind === 'Well log') {
        const parsed = parseLasText(text);
        return { id, name: file.name, kind, format: 'LAS', size: humanSize(file.size), status: 'Ready', details: `MD ${parsed.depthStart ?? '?'}–${parsed.depthStop ?? '?'} · ${parsed.curves.length} curves`, parsed };
    }
    const parsed = parseTableText(text);
    return { id, name: file.name, kind, format: file.name.toLowerCase().endsWith('.tsv') ? 'TSV' : 'CSV', size: humanSize(file.size), status: 'Ready', details: `${parsed.rowCount.toLocaleString()} rows · ${parsed.headers.length} columns`, parsed };
}


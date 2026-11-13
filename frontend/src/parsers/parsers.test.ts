import { describe, expect, it } from 'vitest';
import { parseLasText, parseSegyBlob, parseTableText } from './index';
describe('real local parsers', () => {
    it('reads SEG-Y binary headers and IEEE trace samples', async () => {
        const bytes = new Uint8Array(3600 + 240 + 16);
        const view = new DataView(bytes.buffer);
        view.setUint16(3216, 2000, false);
        view.setUint16(3220, 4, false);
        view.setUint16(3224, 5, false);
        view.setInt32(3600 + 188, 1200, false);
        view.setInt32(3600 + 192, 2400, false);
        [.2, -.4, .8, -.1].forEach((v, i) => view.setFloat32(3600 + 240 + i * 4, v, false));
        const parsed = await parseSegyBlob(new Blob([bytes]));
        expect(parsed.traceCount).toBe(1);
        expect(parsed.samplesPerTrace).toBe(4);
        expect(parsed.inlineRange).toEqual([1200, 1200]);
        expect(parsed.traces[0]).toHaveLength(4);
    });
    it('keeps the SEG-Y preview traces contiguous', async () => {
        const traceCount = 100, traceBytes = 244;
        const bytes = new Uint8Array(3600 + traceCount * traceBytes);
        const view = new DataView(bytes.buffer);
        view.setUint16(3216, 2000, false);
        view.setUint16(3220, 1, false);
        view.setUint16(3224, 5, false);
        for (let i = 0; i < traceCount; i++) {
            const start = 3600 + i * traceBytes;
            view.setInt32(start + 188, 1200, false);
            view.setInt32(start + 192, 1000 + i, false);
            view.setFloat32(start + 240, i, false);
        }
        const parsed = await parseSegyBlob(new Blob([bytes]));
        expect(parsed.traces).toHaveLength(100);
        expect(parsed.crosslineRange).toEqual([1000, 1099]);
    });
    it('reads LAS curves and samples', () => {
        const parsed = parseLasText('~Curve\nDEPT.M : Depth\nGR.API : Gamma\n~Ascii\n1000 45\n1001 47');
        expect(parsed.curves).toEqual(['DEPT', 'GR']);
        expect(parsed.rows).toHaveLength(2);
    });
    it('reads quoted CSV cells', () => {
        const parsed = parseTableText('well,note\nA,"good, tied"\nB,review');
        expect(parsed.headers).toEqual(['well', 'note']);
        expect(parsed.rows[0][1]).toBe('good, tied');
    });
});


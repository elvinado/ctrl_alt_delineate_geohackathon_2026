import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import MarkdownMessage from './MarkdownMessage';
import type { DataItem } from '../data';

const checkshot: DataItem = {
    id: 'dataset-checkshot',
    name: 'm-1.txt',
    kind: 'Table',
    format: 'TXT',
    size: '0.1 MB',
    status: 'Ready',
    details: 'Saved in backend · checkshot',
};

describe('MarkdownMessage dataset viewer actions', () => {
    it('keeps a filename visible and adds its viewer action in an escaped Markdown table', () => {
        const content = [
            '\\*\\*Dataset inventory\\*\\*',
            '',
            '\\| Dataset | Type | Status |',
            '\\|---|---|---|',
            '\\| \\`m-1.txt\\` | Checkshot | Ready |',
        ].join('\n');

        const html = renderToStaticMarkup(
            <MarkdownMessage content={content} datasets={[checkshot]} onOpenDataset={() => undefined}/>,
        );

        expect(html).toContain('<table>');
        expect(html).toContain('<code>m-1.txt</code>');
        expect(html).toContain('class="inline-dataset-link"');
        expect(html).toContain('Open Well Viewer');
    });
});


describe('MarkdownMessage UTF-8 rendering', () => {
    it('decodes UTF-8 byte tokens into Unicode emoji', () => {
        const content = 'Processing result: <0xF0><0x9F><0x9B><0xA2>️';
        const html = renderToStaticMarkup(<MarkdownMessage content={content}/>);

        expect(html).toContain('🛢️');
        expect(html).not.toContain('<0xF0>');
    });

    it('preserves malformed UTF-8 byte tokens instead of corrupting the message', () => {
        const content = 'Unknown token: <0xFF><0xFE>';
        const html = renderToStaticMarkup(<MarkdownMessage content={content}/>);

        expect(html).toContain('&lt;0xFF&gt;&lt;0xFE&gt;');
    });
});


describe('MarkdownMessage escaped line-break rendering', () => {
    it('renders headings, lists and paragraphs when line breaks arrive as literal \\n sequences', () => {
        const content = String.raw`**Processing results — np-1**\\n\\n- **Processing job:** \`18d24542-811c-4863-af3b-94850fd4fbe0\` (succeeded)\\n- **Prediction bins:** 260\\n\\n**Interpretation limits:** These are model predictions, not verified lithology.`;
        const html = renderToStaticMarkup(<MarkdownMessage content={content}/>);

        expect(html).toContain('<h3>');
        expect(html).toContain('<ul>');
        expect(html).toContain('<li>');
        expect(html).toContain('<strong>Processing job:</strong>');
        expect(html).toContain('<p><strong>Interpretation limits:</strong>');
        expect(html).not.toContain('\\\\n');
    });
});


describe('MarkdownMessage processing job actions', () => {
    it('renders an inline Scripting action beside a code-formatted UUID job ID', () => {
        const jobId = '18d24542-811c-4863-af3b-94850fd4fbe0';
        const onOpenProcessingJob = () => undefined;
        const html = renderToStaticMarkup(
            <MarkdownMessage
                content={`- **Processing job:** \`${jobId}\` (succeeded)`}
                onOpenProcessingJob={onOpenProcessingJob}
            />,
        );

        expect(html).toContain(`<code>${jobId}</code>`);
        expect(html).toContain('class="inline-dataset-link"');
        expect(html).toContain('Open in Scripting');
    });

    it('adds a Processing navigation action beside a Python script name', () => {
        const html = renderToStaticMarkup(
            <MarkdownMessage content="Script: `run_inference.py`" onOpenProcessingTab={() => undefined}/>,
        );

        expect(html).toContain('<code>run_inference.py</code>');
        expect(html).toContain('Go to Scripting');
    });

    it('adds the Processing navigation action to bare Python filenames in workflow tables', () => {
        const content = ['| Scripts | Description |', '|---|---|', '| `run_model.py` | Run the selected workflow |'].join('\n');
        const html = renderToStaticMarkup(
            <MarkdownMessage content={content} onOpenProcessingTab={() => undefined}/>,
        );

        expect(html).toContain('<code>run_model.py</code>');
        expect(html).toContain('Go to Scripting');
    });
});


describe('MarkdownMessage bare processing job IDs in tables', () => {
    it('renders a code-formatted ID and adjacent Scripting action for a plain UUID table cell', () => {
        const jobId = '18d24542-811c-4863-af3b-94850fd4fbe0';
        const content = [
            '**Completed processing jobs**',
            '',
            '| Python file | ID | Status |',
            '|---|---|---|',
            `| run_inference.py | ${jobId} | succeeded |`,
        ].join('\\n');
        const html = renderToStaticMarkup(
            <MarkdownMessage content={content} onOpenProcessingJob={() => undefined}/>,
        );

        expect(html).toContain(`<code>${jobId}</code>`);
        expect(html).toContain('Open in Scripting');
        expect(html).toContain('class="inline-dataset-link"');
    });
});


describe('MarkdownMessage process result reports', () => {
    it('keeps underscores inside identifiers and still renders word-edge emphasis', () => {
        const html = renderToStaticMarkup(
            <MarkdownMessage content={'Status COMPLETED_WITH_GRV_BLOCKERS from geogg_prospect_summary.json is _provisional_.'}/>,
        );

        expect(html).toContain('COMPLETED_WITH_GRV_BLOCKERS');
        expect(html).toContain('geogg_prospect_summary.json');
        expect(html).toContain('<em>provisional</em>');
    });

    it('renders the fixed report layout: title, verdict quote, table and job action', () => {
        const content = [
            '# Process D · Geobody extraction (stage 12) — results',
            '',
            'Job `8243c6e5-3cc3-4d42-ae62-64f609c7dd4c` · status **succeeded** · view **overview**',
            '',
            '> 🟡 **Verdict: Usable with caveats.** 248 bodies were kept.',
            '',
            '## At a glance',
            '',
            '| Measure | Value |',
            '|---|---|',
            '| Run status | `COMPLETED_WITH_GRV_BLOCKERS` |',
        ].join('\n');

        const html = renderToStaticMarkup(<MarkdownMessage content={content} onOpenProcessingJob={() => undefined}/>);

        expect(html).toContain('<h3>Process D · Geobody extraction (stage 12) — results</h3>');
        expect(html).toContain('<blockquote>');
        expect(html).toContain('<strong>Verdict: Usable with caveats.</strong>');
        expect(html).toContain('<code>COMPLETED_WITH_GRV_BLOCKERS</code>');
        expect(html).toContain('Open in Scripting');
    });
});

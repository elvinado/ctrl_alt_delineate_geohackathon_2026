import type { ReactNode } from 'react';
import MathExpression from './MathExpression';
import type { DataItem } from '../data';
import { Activity, Waves, Workflow } from 'lucide-react';

function decodeUtf8ByteEscapes(value: string): string {
    // Some model/API paths expose UTF-8 bytes as printable tokens such as
    // <0xF0><0x9F><0x9B><0xA2>. Decode each contiguous byte run before Markdown parsing.
    return value.replace(/(?:<0x[0-9a-f]{2}>)+/gi, sequence => {
        const bytes = Array.from(sequence.matchAll(/<0x([0-9a-f]{2})>/gi), match => Number.parseInt(match[1], 16));
        try {
            return new TextDecoder('utf-8', { fatal: true }).decode(new Uint8Array(bytes));
        } catch {
            // Preserve malformed or non-UTF-8 tokens rather than silently corrupting text.
            return sequence;
        }
    });
}

function normalizeFilename(value: string): string {
    return value.trim().replace(/\\([_*`])/g, '$1').replace(/\\/g, '/').split('/').pop()?.toLocaleLowerCase() ?? '';
}

function findDataset(value: string, datasets: DataItem[]): DataItem | undefined {
    const normalized = normalizeFilename(value);
    return datasets.find(item => normalizeFilename(item.name) === normalized);
}

function datasetAction(dataset: DataItem, onOpenDataset: (dataset: DataItem) => void, key: string): ReactNode {
    return <button key={key} type="button" className="inline-dataset-link" onClick={() => onOpenDataset(dataset)} title={`Open ${dataset.name} in its viewer`}>
        {dataset.kind === 'Seismic' ? <Waves size={12}/> : <Activity size={12}/>}
        {dataset.kind === 'Seismic' ? 'Open Seismic Viewer' : 'Open Well Viewer'}
    </button>;
}

function processingJobAction(jobId: string, onOpenProcessingJob: (jobId: string) => void, key: string): ReactNode {
    return <button key={key} type="button" className="inline-dataset-link" onClick={() => onOpenProcessingJob(jobId)} title={`Open processing job ${jobId} in Scripting`}>
        <Workflow size={12}/>Open in Scripting
    </button>;
}

function processingTabAction(scriptName: string, onOpenProcessingTab: (scriptName: string) => void, key: string): ReactNode {
    return <button key={key} type="button" className="inline-dataset-link" onClick={() => onOpenProcessingTab(scriptName)} title={`Select ${scriptName} in Scripting`}>
        <Workflow size={12}/>Go to Scripting
    </button>;
}

const PROCESSING_JOB_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function inlineMarkdown(text: string, datasets: DataItem[], onOpenDataset?: (dataset: DataItem) => void, onOpenProcessingJob?: (jobId: string) => void, onOpenProcessingTab?: (scriptName: string) => void): ReactNode[] {
    const escapeRegex = (value: string) => value.split('').map(char => char === '_' ? String.raw`\\?_` : char.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('');
    const filenames = datasets.map(item => escapeRegex(item.name)).sort((a, b) => b.length - a.length);
    // Parse Markdown wrappers before bare filenames so code/bold formatting is retained.
    const tokens = [
        // Underscores only open or close emphasis at a word edge, so identifiers such as
        // geogg_prospect_summary or NOT_VALIDATED keep their underscores when they are not in code.
        String.raw`\*\*[^*]+\*\*`, String.raw`(?<![A-Za-z0-9_])__[^_]+__(?![A-Za-z0-9_])`, String.raw`\*[^*]+\*`,
        String.raw`(?<![A-Za-z0-9_])_[^_\n]+_(?![A-Za-z0-9_])`,
        String.raw`\`[^\`]+\``, PROCESSING_JOB_ID.source, String.raw`[\w./-]+\.py`, ...filenames,
        String.raw`\\\([\s\S]+?\\\)`, String.raw`\$\$[^$]+\$\$`, String.raw`(?<!\\)\$(?!\s)[^$\n]+(?<!\s)\$`,
        String.raw`\[[^\]]+\]\(https?:\/\/[^)]+\)`
    ];
    const pattern = new RegExp(`(${tokens.join('|')})`, 'g');
    const nodes: ReactNode[] = [];
    let last = 0;
    for (const match of text.matchAll(pattern)) {
        const value = match[0];
        const index = match.index ?? 0;
        if (index > last) nodes.push(text.slice(last, index));
        if ((value.startsWith('**') || value.startsWith('__')) && value.length >= 4) {
            nodes.push(<strong key={index}>{inlineMarkdown(value.slice(2, -2), datasets, onOpenDataset, onOpenProcessingJob, onOpenProcessingTab)}</strong>);
        } else if ((value.startsWith('*') || value.startsWith('_')) && value.length >= 2) {
            nodes.push(<em key={index}>{inlineMarkdown(value.slice(1, -1), datasets, onOpenDataset, onOpenProcessingJob, onOpenProcessingTab)}</em>);
        } else if (value.startsWith('`')) {
            const filename = value.slice(1, -1);
            const dataset = findDataset(filename, datasets);
            nodes.push(<code key={`code-${index}`}>{filename}</code>);
            if (dataset && onOpenDataset) nodes.push(datasetAction(dataset, onOpenDataset, `dataset-${index}-${dataset.id}`));
            else if (/\.py$/i.test(filename) && onOpenProcessingTab) {
                nodes.push(processingTabAction(filename, onOpenProcessingTab, `processing-tab-${index}-${filename}`));
            } else if (PROCESSING_JOB_ID.test(filename) && onOpenProcessingJob) {
                nodes.push(processingJobAction(filename, onOpenProcessingJob, `processing-job-${index}-${filename}`));
            }
        } else if (/^[\w./-]+\.py$/i.test(value) && onOpenProcessingTab) {
            nodes.push(<code key={`script-code-${index}`}>{value}</code>);
            nodes.push(processingTabAction(value, onOpenProcessingTab, `processing-tab-${index}-${value}`));
        } else if (PROCESSING_JOB_ID.test(value)) {
            // Job-history tables often return UUIDs as plain text, not Markdown code.
            // Render the ID as code and place its Processing action immediately after it.
            nodes.push(<code key={`job-code-${index}`}>{value}</code>);
            if (onOpenProcessingJob) nodes.push(processingJobAction(value, onOpenProcessingJob, `processing-job-${index}-${value}`));
        } else {
            const dataset = findDataset(value, datasets);
            if (dataset && onOpenDataset) {
                // Keep the filename visible; the action is an additional adjacent control.
                nodes.push(value, datasetAction(dataset, onOpenDataset, `dataset-${index}-${dataset.id}`));
            } else if (value.startsWith('$$')) nodes.push(<MathExpression key={index} expression={value.slice(2, -2)} display/>);
            else if (value.startsWith('\\(')) nodes.push(<MathExpression key={index} expression={value.slice(2, -2)}/>);
            else if (value.startsWith('$')) nodes.push(<MathExpression key={index} expression={value.slice(1, -1)}/>);
            else {
                const link = value.match(/^\[([^\]]+)\]\((https?:\/\/[^)]+)\)$/);
                if (link) nodes.push(<a key={index} href={link[2]} target="_blank" rel="noreferrer">{link[1]}</a>);
                else nodes.push(value);
            }
        }
        last = index + value.length;
    }
    if (last < text.length) nodes.push(text.slice(last));
    return nodes;
}

export default function MarkdownMessage({ content, datasets = [], onOpenDataset, onOpenProcessingJob, onOpenProcessingTab }: { content: string; datasets?: DataItem[]; onOpenDataset?: (dataset: DataItem) => void; onOpenProcessingJob?: (jobId: string) => void; onOpenProcessingTab?: (scriptName: string) => void }) {
    // Some model responses escape Markdown punctuation (for example \| and \`).
    // Normalize those escapes before detecting tables and inline code so filenames
    // still resolve to their dataset and receive the adjacent viewer action.
    const decodedContent = decodeUtf8ByteEscapes(content);
    // Some backend/model responses contain JSON-style escaped line breaks
    // as literal characters (\\n) instead of actual newline characters.
    // Decode those before Markdown block parsing so headings, lists and
    // paragraphs render consistently with normally formatted responses.
    const withRealLineBreaks = decodedContent
        .replace(/\\r\\n/g, '\n')
        .replace(/\\n/g, '\n')
        .replace(/\\r/g, '\n');
    const normalizedContent = withRealLineBreaks.replace(/\\([|*_`])/g, '$1');
    const lines = normalizedContent.replace(/\r\n?/g, '\n').split('\n');
    const blocks: ReactNode[] = [];
    let i = 0;
    while (i < lines.length) {
        const line = lines[i];
        if (!line.trim()) { i++; continue; }
        if (line.startsWith('```')) {
            const language = line.slice(3).trim();
            const code: string[] = [];
            i++;
            while (i < lines.length && !lines[i].startsWith('```')) code.push(lines[i++]);
            if (i < lines.length) i++;
            blocks.push(<pre key={`code-${i}`}><code className={language ? `language-${language}` : undefined}>{code.join('\n')}</code></pre>);
            continue;
        }
        const trimmedLine = line.trim();
        const environmentStart = trimmedLine.match(/^\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|flalign\*?|cases)\}/);
        if (trimmedLine.startsWith('$$') || trimmedLine.startsWith('\\[') || environmentStart) {
            const delimiter = trimmedLine.startsWith('$$') ? '$$' : trimmedLine.startsWith('\\[') ? '\\]' : `\\end{${environmentStart?.[1]}}`;
            const openingLength = trimmedLine.startsWith('$$') || trimmedLine.startsWith('\\[') ? 2 : 0;
            const inlineEnd = openingLength ? trimmedLine.indexOf(delimiter, openingLength) : -1;
            if (inlineEnd >= 0) {
                blocks.push(<MathExpression key={`math-${i}`} expression={trimmedLine.slice(openingLength, inlineEnd)} display/>);
                i++;
            } else {
                const equation: string[] = [];
                const firstLine = openingLength ? trimmedLine.slice(openingLength) : trimmedLine;
                if (firstLine) equation.push(firstLine);
                i++;
                while (i < lines.length && !lines[i].trim().includes(delimiter)) equation.push(lines[i++]);
                if (i < lines.length) {
                    const closingLine = lines[i].trim();
                    const closingIndex = closingLine.indexOf(delimiter);
                    const trailing = environmentStart ? closingLine : closingLine.slice(0, closingIndex);
                    if (trailing) equation.push(trailing);
                    i++;
                }
                blocks.push(<MathExpression key={`math-${i}`} expression={equation.join('\n')} display/>);
            }
            continue;
        }
        const heading = line.match(/^(#{1,6})\s+(.+)$/);
        if (heading) {
            const level = heading[1].length;
            const children = inlineMarkdown(heading[2], datasets, onOpenDataset, onOpenProcessingJob, onOpenProcessingTab);
            blocks.push(level === 1 ? <h3 key={i}>{children}</h3> : level === 2 ? <h4 key={i}>{children}</h4> : <h5 key={i}>{children}</h5>);
            i++; continue;
        }
        if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
            const cells = (value: string) => value.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim());
            const headers = cells(line);
            i += 2;
            const rows: string[][] = [];
            while (i < lines.length && lines[i].includes('|') && lines[i].trim()) rows.push(cells(lines[i++]));
            blocks.push(<div className="markdown-table-wrap" key={`table-${i}`}><table><thead><tr>{headers.map((cell, index) => <th key={index}>{inlineMarkdown(cell, datasets, onOpenDataset, onOpenProcessingJob, onOpenProcessingTab)}</th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{headers.map((_, index) => <td key={index}>{inlineMarkdown(row[index] ?? '', datasets, onOpenDataset, onOpenProcessingJob, onOpenProcessingTab)}</td>)}</tr>)}</tbody></table></div>);
            continue;
        }
        if (/^\s*(---+|\*\*\*|___)\s*$/.test(line)) { blocks.push(<hr key={`hr-${i}`}/>); i++; continue; }
        if (/^\s*([-*+] |\d+\. )/.test(line)) {
            const ordered = /^\s*\d+\. /.test(line);
            const items: ReactNode[] = [];
            while (i < lines.length && /^\s*([-*+] |\d+\. )/.test(lines[i])) {
                const item = lines[i].replace(/^\s*(?:[-*+] |\d+\. )/, '');
                items.push(<li key={i}>{inlineMarkdown(item, datasets, onOpenDataset, onOpenProcessingJob, onOpenProcessingTab)}</li>); i++;
            }
            blocks.push(ordered ? <ol key={`list-${i}`}>{items}</ol> : <ul key={`list-${i}`}>{items}</ul>);
            continue;
        }
        if (line.startsWith('> ')) {
            const quote: string[] = [];
            while (i < lines.length && lines[i].startsWith('> ')) quote.push(lines[i++].slice(2));
            blocks.push(<blockquote key={`quote-${i}`}>{quote.map((item, j) => <p key={j}>{inlineMarkdown(item, datasets, onOpenDataset, onOpenProcessingJob, onOpenProcessingTab)}</p>)}</blockquote>);
            continue;
        }
        const paragraph = [line]; i++;
        while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|```|> )/.test(lines[i]) && !/^\s*([-*+] |\d+\. )/.test(lines[i])) paragraph.push(lines[i++]);
        blocks.push(<p key={`p-${i}`}>{paragraph.map((item, j) => <span key={j}>{j > 0 && <br/>}{inlineMarkdown(item, datasets, onOpenDataset, onOpenProcessingJob, onOpenProcessingTab)}</span>)}</p>);
    }
    return <div className="markdown-content">{blocks}</div>;
}

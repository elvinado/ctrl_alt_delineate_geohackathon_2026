import { useState } from 'react';
import { Pencil } from 'lucide-react';
import type { DataItem } from '../data';
import type { GeoChatMessage } from '../../api/backend';
import MarkdownMessage from './MarkdownMessage';

/** A per-message switch between the source text and its rendered presentation. */
export default function MessageBody({ message, onEdit, datasets = [], onOpenDataset, onOpenProcessingJob, onOpenProcessingTab }: {
    message: GeoChatMessage; onEdit?: () => void; datasets?: DataItem[];
    onOpenDataset?: (dataset: DataItem) => void; onOpenProcessingJob?: (jobId: string) => void; onOpenProcessingTab?: (scriptName: string) => void;
}) {
    const [view, setView] = useState<'rendered' | 'raw'>('rendered');

    const isRaw = view === 'raw';

    return <div className="message-body-content">
        {isRaw
            ? <pre className="message-raw-content">{message.content}</pre>
            : message.role === 'assistant'
                ? <MarkdownMessage content={message.content} datasets={datasets.filter(item => (!item.status || item.status.toLowerCase().includes('ready')) && (item.kind !== 'Table' || item.format?.toLowerCase() === 'txt'))} onOpenDataset={onOpenDataset} onOpenProcessingJob={onOpenProcessingJob} onOpenProcessingTab={onOpenProcessingTab}/>
                : <p className="message-rendered-user">{message.content}</p>}
        <div className="message-body-actions">
            {message.role === 'user' && onEdit && <div className="message-edit-action">
                <button type="button" onClick={onEdit} aria-label="Edit message">
                    <Pencil size={12}/> Edit
                </button>
            </div>}
            <div className="message-view-toggle">
            <button
                type="button"
                aria-label={isRaw ? 'Switch to rendered view' : 'Switch to raw Markdown view'}
                aria-pressed={isRaw}
                onClick={() => setView(isRaw ? 'rendered' : 'raw')}
            >
                {isRaw ? 'View rendered' : 'View raw Markdown'}
            </button>
            </div>
        </div>
    </div>;
}

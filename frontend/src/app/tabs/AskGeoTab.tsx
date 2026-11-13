import type { Dispatch, SetStateAction } from 'react';
import { useState } from 'react';
import { MessageSquare, Plus, Send, UserRound, RefreshCw, MessageCircle, X } from 'lucide-react';
import type { GeoChatConversation, GeoChatMessage } from '../../api/backend';
import MessageBody from './MessageBody';
import type { DataItem } from '../data';

const starterQuestions = [
  'List my projects or switch to another project.',
  'What datasets are available in this project?',
  'List the wells and surveys in this project.',
  'What processing workflows can I run?',
  'Show my completed processing jobs and outputs.',
  'Explain the results of each process.',
  'Explain the results of Process A.',
];

function formatWorkflowCatalogue(content: string): string {
  const rows = new Map<string, string>();
  for (const line of content.split(/\r?\n/)) {
    const scripts = [...line.matchAll(/(?:`)?([\w./-]+\.py)(?:`)?/gi)].map(match => match[1].split('/').pop() || match[1]);
    if (!scripts.length) continue;
    let description = line
      .replace(/^\s*[-*+>\d.)|]+\s*/, '')
      .replace(/\|/g, ' ')
      .replace(/`/g, '')
      .replace(/(?:[\w./-]+\.py)/gi, '')
      .replace(/\s+/g, ' ')
      .replace(/^[-:–—\s]+|[-:–—\s]+$/g, '')
      .trim();
    description = description.replace(/^(scripts?|python scripts?)\s*/i, '').trim();
    for (const script of scripts) rows.set(script, description || 'Available processing script.');
  }
  return ['| Scripts | Description |', '|---|---|', ...[...rows].map(([script, description]) => `| \`${script}\` | ${description.replace(/\|/g, '\\|')} |`)].join('\n');
}

function isWorkflowCatalogueQuestion(content: string): boolean {
  return /processing workflows?.*(available|run|project)|(?:available|list|show).*(?:processing scripts|processing workflows)/i.test(content);
}

export default function AskGeoTab({ query, setQuery, onSend, messages, busy, error, onNewChat, conversations, activeConversationId, onLoadConversation, onDeleteConversation, historyLoading, suggestedQuestions, onSuggestedQuestion, datasets, onOpenDataset, onOpenProcessingJob, onOpenProcessingTab }: {
  query: string; setQuery: Dispatch<SetStateAction<string>>; onSend: (editIndex?: number, contentOverride?: string) => void | Promise<void>;
  messages: GeoChatMessage[]; busy: boolean; error: string; onNewChat: () => void;
  conversations: GeoChatConversation[]; activeConversationId: string;
  onLoadConversation: (conversationId: string) => void; onDeleteConversation: (conversationId: string) => void; historyLoading: boolean;
  suggestedQuestions: string[]; onSuggestedQuestion: (question: string) => void;
  datasets: DataItem[]; onOpenDataset: (dataset: DataItem) => void; onOpenProcessingJob: (jobId: string) => void; onOpenProcessingTab: (scriptName: string) => void;
}) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState('');

  const startEditing = (index: number, content: string) => {
    setEditingIndex(index);
    setEditDraft(content);
  };

  const cancelEditing = () => {
    setEditingIndex(null);
    setEditDraft('');
  };

  const resendEditedMessage = async () => {
    if (editingIndex === null || !editDraft.trim() || busy) return;
    const index = editingIndex;
    await onSend(index, editDraft);
    setEditingIndex(null);
    setEditDraft('');
  };

  return <section className="page chat-page">
    <div className="page-title chat-title"><div><span className="eyebrow">GEOSCIENCE Q&A</span><h1>Ask Geo</h1><p>Project-specific conversations are saved and can be resumed later.</p></div></div>
    <div className="chat-layout">
      <div className="chat-sidebar">
        <button className="btn btn-secondary chat-new-button" onClick={onNewChat} disabled={busy}><Plus size={15}/> New chat</button>
        <aside className="chat-history theme-panel">
        <div className="chat-history-heading"><strong>Chat history</strong><button className="btn-icon icon-btn theme-control" onClick={() => onLoadConversation('')} title="Refresh chat history" disabled={historyLoading}><RefreshCw size={14}/></button></div>
        {historyLoading && <small className="history-empty">Loading conversations…</small>}
        {!historyLoading && !conversations.length && <small className="history-empty">No saved conversations for this project.</small>}
        {conversations.map(item => <div key={item.conversation_id} className={`chat-history-row ${activeConversationId === item.conversation_id ? 'active' : ''}`}>
          <button className="btn-chat-history chat-history-item" onClick={() => onLoadConversation(item.conversation_id)} disabled={busy || historyLoading}>
            <MessageCircle size={14}/><span><strong>{item.title}</strong><small>{item.updated_at ? new Date(item.updated_at).toLocaleString() : 'Saved conversation'}</small></span>
          </button>
          <button className="btn-icon btn-icon-danger chat-history-delete" onClick={() => onDeleteConversation(item.conversation_id)} disabled={busy || historyLoading} title="Delete chat history" aria-label={`Delete conversation: ${item.title}`}><X size={14}/></button>
        </div>)}
        </aside>
      </div>
      <div className="chat-window theme-panel">
        <div className="chat-thread" aria-live="polite">
          {!messages.length && !busy && <>
            <div className="chat-welcome"><MessageSquare/><strong>What would you like to understand?</strong><p>Ask a geoscience question or explore the data and processing workflows in the selected project.</p></div>
            <div className="chat-suggested-questions chat-starters" aria-label="Ask Geo starter questions">
              <strong>Try asking</strong>
              <div>{starterQuestions.map(question => <button type="button" className="btn-chat-suggestion" key={question} onClick={() => onSuggestedQuestion(question)}>{question}</button>)}</div>
            </div>
          </>}
          {messages.map((message, index) => {
            const previous = messages[index - 1];
            const workflowReply = message.role === 'assistant' && previous?.role === 'user' && isWorkflowCatalogueQuestion(previous.content);
            const displayMessage = workflowReply ? { ...message, content: formatWorkflowCatalogue(message.content) } : message;
            return <article className={`message ${message.role}`} key={`${index}-${message.role}`}>
            {message.role === 'user' && <div aria-hidden="true"><UserRound/></div>}
            <div className="message-body">{editingIndex === index ? <div className="message-inline-editor">
              <textarea aria-label="Edit message" value={editDraft} onChange={event => setEditDraft(event.target.value)} rows={Math.max(2, Math.min(8, editDraft.split('\n').length + 1))} disabled={busy} autoFocus/>
              <div className="message-edit-warning" role="status">Resending replaces this message and deletes all subsequent messages in the conversation.</div>
              <div className="message-edit-actions"><button type="button" className="message-edit-cancel" onClick={cancelEditing} disabled={busy}>Cancel</button><button type="button" className="message-edit-resend" onClick={() => void resendEditedMessage()} disabled={!editDraft.trim() || busy}><RefreshCw size={12}/> Resend</button></div>
            </div> : <MessageBody message={displayMessage} datasets={datasets} onOpenDataset={onOpenDataset} onOpenProcessingJob={onOpenProcessingJob} onOpenProcessingTab={onOpenProcessingTab} onEdit={!busy && message.role === 'user' ? () => startEditing(index, message.content) : undefined}/>}</div>
          </article>;
          })}
          {busy && <article className="message assistant"><div className="message-body"><p className="typing">Thinking…</p></div></article>}
          {suggestedQuestions.length > 0 && <div className="chat-suggested-questions" aria-label="Suggested follow-up questions">
            <strong>Suggested questions</strong>
            <div>{suggestedQuestions.map(question => <button type="button" className="btn-chat-suggestion" key={question} onClick={() => onSuggestedQuestion(question)}>{question}</button>)}</div>
          </div>}
          {error && <div className="chat-error" role="alert"><strong>Could not get a response</strong><p>{error}</p><small>Your message remains in this chat. You can send another message after resolving the issue.</small></div>}
        </div>
        <form className="composer" onSubmit={event => { event.preventDefault(); void onSend(); }}>
          <textarea aria-label="Message Ask Geo" value={query} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void onSend(); } }} placeholder="Ask a geoscience question…" rows={2} disabled={busy}/>
          <button type="submit" className="btn-icon btn-icon-primary composer-send" aria-label="Send message" title="Send message" disabled={!query.trim() || busy}><Send/></button>
        </form>
        <small className="grounded">Enter to send · Shift+Enter for a new line · History is saved per project.</small>
      </div>
    </div>
  </section>;
}

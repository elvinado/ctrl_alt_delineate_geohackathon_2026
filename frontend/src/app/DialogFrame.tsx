import type { CSSProperties, ReactNode } from 'react';
import { createPortal } from 'react-dom';

type Props = {
  eyebrow: string;
  title: string;
  description: string;
  labelledBy: string;
  onClose: () => void;
  closeDisabled?: boolean;
  panelClassName?: string;
  children: ReactNode;
};

/** Shared accessible shell for project, survey, well, and upload dialogs. */
export default function DialogFrame({
  eyebrow,
  title,
  description,
  labelledBy,
  onClose,
  closeDisabled = false,
  panelClassName = '',
  children,
}: Props) {
  const appShell = document.querySelector('.app-shell');
  const themeStyle: CSSProperties = {};
  if (appShell) {
    const computed = window.getComputedStyle(appShell);
    for (let index = 0; index < computed.length; index += 1) {
      const property = computed.item(index);
      if (property.startsWith('--')) {
        (themeStyle as Record<string, string>)[property] = computed.getPropertyValue(property);
      }
    }
    (themeStyle as Record<string, string>)['colorScheme'] = computed.colorScheme;
  }

  return createPortal(
    <div
      className="project-dialog-backdrop"
      style={themeStyle}
      role="presentation"
      onMouseDown={event => {
        if (event.target === event.currentTarget && !closeDisabled) onClose();
      }}
    >
      <section
        className={`project-dialog theme-panel ${panelClassName}`.trim()}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
      >
        <div className="project-dialog-heading">
          <div>
            <span className="eyebrow">{eyebrow}</span>
            <h2 id={labelledBy}>{title}</h2>
            <p>{description}</p>
          </div>
          <button
            type="button"
            className="btn-icon project-dialog-close"
            aria-label="Close dialog"
            disabled={closeDisabled}
            onClick={onClose}
          >
            ×
          </button>
        </div>
        {children}
      </section>
    </div>,
    document.body,
  );
}

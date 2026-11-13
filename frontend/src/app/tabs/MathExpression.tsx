import { useEffect, useRef } from 'react';

type MathJaxWindow = Window & {
    MathJax?: {
        startup?: { promise?: Promise<void> };
        typesetPromise?: (elements?: HTMLElement[]) => Promise<void>;
        typesetClear?: (elements?: HTMLElement[]) => void;
    };
};

/** Render TeX with MathJax, including expressions mounted before its CDN script loads. */
export default function MathExpression({ expression, display = false }: { expression: string; display?: boolean }) {
    const inlineRef = useRef<HTMLSpanElement>(null);
    const displayRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const container = display ? displayRef.current : inlineRef.current;
        if (!container) return;

        let active = true;
        let typeset = false;

        const render = () => {
            const mathJax = (window as MathJaxWindow).MathJax;
            if (!active || !mathJax?.typesetPromise) return;

            void (mathJax.startup?.promise ?? Promise.resolve())
                .then(async () => {
                    if (!active || !mathJax.typesetPromise) return;
                    await mathJax.typesetPromise([container]);
                    typeset = true;
                })
                .catch((error: unknown) => console.error('MathJax typesetting failed:', error));
        };

        // The component can mount before the deferred CDN script has finished loading.
        render();
        window.addEventListener('mathjax-ready', render);

        return () => {
            active = false;
            window.removeEventListener('mathjax-ready', render);
            const mathJax = (window as MathJaxWindow).MathJax;
            if (typeset) mathJax?.typesetClear?.([container]);
        };
    }, [expression, display]);

    const source = display ? `\\[${expression}\\]` : `\\(${expression}\\)`;
    return display
        ? <div ref={displayRef} className="markdown-math-block">{source}</div>
        : <span ref={inlineRef} className="markdown-math-inline">{source}</span>;
}

import React, { useMemo } from 'react';
import katex from 'katex';

interface MathRendererProps {
  content: string;
  className?: string;
}

export const MathRenderer: React.FC<MathRendererProps> = ({ content, className = '' }) => {
  const renderedElements = useMemo(() => {
    if (!content) return null;

    // Split text into tokens based on LaTeX delimiters:
    // 1. $$ ... $$ (display math)
    // 2. \[ ... \] (display math)
    // 3. $ ... $ (inline math)
    // 4. \( ... \) (inline math)
    
    // Regular expression matching block math first, then inline math
    const mathRegex = /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\$(?:\\.|[^\$\\\n])+\$|\\\([\s\S]*?\\\))/g;
    const parts = content.split(mathRegex);

    return parts.map((part, index) => {
      if (!part) return null;

      // Check for block math $$...$$
      if (part.startsWith('$$') && part.endsWith('$$') && part.length >= 4) {
        const formula = part.slice(2, -2).trim();
        try {
          const html = katex.renderToString(formula, {
            displayMode: true,
            throwOnError: false,
            output: 'html',
          });
          return (
            <div
              key={index}
              className="my-3 overflow-x-auto py-1 text-stone-900 dark:text-stone-100"
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        } catch {
          return <pre key={index} className="my-2 p-2 bg-stone-100 dark:bg-stone-900 rounded font-mono text-xs">{part}</pre>;
        }
      }

      // Check for block math \[...\]
      if (part.startsWith('\\[') && part.endsWith('\\]') && part.length >= 4) {
        const formula = part.slice(2, -2).trim();
        try {
          const html = katex.renderToString(formula, {
            displayMode: true,
            throwOnError: false,
            output: 'html',
          });
          return (
            <div
              key={index}
              className="my-3 overflow-x-auto py-1 text-stone-900 dark:text-stone-100"
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        } catch {
          return <pre key={index} className="my-2 p-2 bg-stone-100 dark:bg-stone-900 rounded font-mono text-xs">{part}</pre>;
        }
      }

      // Check for inline math $...$
      if (part.startsWith('$') && part.endsWith('$') && part.length >= 2) {
        const formula = part.slice(1, -1).trim();
        try {
          const html = katex.renderToString(formula, {
            displayMode: false,
            throwOnError: false,
            output: 'html',
          });
          if (html.includes('katex-error')) {
            return (
              <span key={index} className="inline-block px-1 font-mono text-xs text-stone-700 dark:text-stone-300">
                {formula}
              </span>
            );
          }
          return (
            <span
              key={index}
              className="inline-block px-0.5 text-stone-900 dark:text-stone-100"
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        } catch {
          return <span key={index} className="inline-block px-1 font-mono text-xs">{formula}</span>;
        }
      }

      // Check for inline math \(...\)
      if (part.startsWith('\\(') && part.endsWith('\\)') && part.length >= 4) {
        const formula = part.slice(2, -2).trim();
        try {
          const html = katex.renderToString(formula, {
            displayMode: false,
            throwOnError: false,
            output: 'html',
          });
          if (html.includes('katex-error')) {
            return (
              <span key={index} className="inline-block px-1 font-mono text-xs text-stone-700 dark:text-stone-300">
                {formula}
              </span>
            );
          }
          return (
            <span
              key={index}
              className="inline-block px-0.5 text-stone-900 dark:text-stone-100"
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        } catch {
          return <span key={index} className="inline-block px-1 font-mono text-xs">{formula}</span>;
        }
      }

      // Plain text formatting (preserving newlines and paragraphs)
      return (
        <span key={index} className="whitespace-pre-wrap leading-relaxed">
          {part}
        </span>
      );
    });
  }, [content]);

  return <div className={`math-content ${className}`}>{renderedElements}</div>;
};

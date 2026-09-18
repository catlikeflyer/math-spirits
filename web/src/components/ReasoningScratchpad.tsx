import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { MathRenderer } from './MathRenderer';

interface ReasoningScratchpadProps {
  thoughtContent: string;
  isStreaming?: boolean;
  spiritName?: string;
}

export const ReasoningScratchpad: React.FC<ReasoningScratchpadProps> = ({
  thoughtContent,
  isStreaming = false,
  spiritName,
}) => {
  const [isOpen, setIsOpen] = useState(true);

  // Automatically keep open during streaming so the user sees live thoughts
  useEffect(() => {
    if (isStreaming) {
      setIsOpen(true);
    }
  }, [isStreaming]);

  if (!thoughtContent.trim() && !isStreaming) {
    return null;
  }

  // Count reasoning steps (paragraphs or numbered items)
  const steps = thoughtContent
    .split(/\n\s*\n/)
    .filter((s) => s.trim().length > 0);

  return (
    <div className="my-3 overflow-hidden rounded border border-stone-200 dark:border-[#282b32] bg-stone-50/50 dark:bg-[#181a1e] transition-colors">
      {/* Accordion Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 bg-stone-100/70 dark:bg-[#1d1f24] hover:bg-stone-200/50 dark:hover:bg-[#23262c] transition-colors text-left border-b border-stone-200 dark:border-[#282b32]"
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-5 h-5 rounded bg-stone-200 dark:bg-stone-800 text-stone-600 dark:text-stone-300 text-xs font-serif font-medium select-none">
            ⊢
          </div>

          <div>
            <div className="flex items-center gap-2">
              <span className="font-serif font-medium text-xs sm:text-sm text-stone-800 dark:text-stone-200">
                Derivation & Deductive Reasoning
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-stone-200/80 dark:bg-stone-800 text-stone-600 dark:text-stone-400 border border-stone-300 dark:border-stone-700">
                {isStreaming ? 'Synthesizing...' : `${steps.length} steps`}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-stone-500 dark:text-stone-400">
          <span className="text-[11px] hidden sm:inline-block">
            {isOpen ? 'Hide' : 'Show'}
          </span>
          {isOpen ? (
            <ChevronUp className="w-3.5 h-3.5" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5" />
          )}
        </div>
      </button>

      {/* Accordion Body */}
      {isOpen && (
        <div className="p-4 border-l-2 border-stone-300 dark:border-stone-700 ml-4 my-3 bg-transparent text-stone-700 dark:text-stone-300 text-sm leading-relaxed space-y-2 font-serif">
          {thoughtContent ? (
            <MathRenderer content={thoughtContent} className="text-stone-700 dark:text-stone-300" />
          ) : (
            <div className="flex items-center gap-2 text-stone-400 italic text-xs py-1">
              <span>Generating mathematical derivation...</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

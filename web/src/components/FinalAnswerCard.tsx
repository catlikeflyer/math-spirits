import React, { useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { MathRenderer } from './MathRenderer';

interface FinalAnswerCardProps {
  finalAnswer: string;
}

export const FinalAnswerCard: React.FC<FinalAnswerCardProps> = ({ finalAnswer }) => {
  const [copied, setCopied] = useState(false);

  if (!finalAnswer || !finalAnswer.trim()) {
    return null;
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(finalAnswer.trim());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // If finalAnswer has LaTeX syntax (e.g. \frac, \sqrt, \pm, \sum) but isn't wrapped in math delimiters ($ or \[), wrap in $$...$$
  const hasLatexMacros = /\\(frac|sqrt|pm|times|div|cdot|sum|int|infty|boxed|mathbf|text|alpha|beta|gamma|delta|pi|theta)|\^|_/.test(finalAnswer);
  const displayContent = hasLatexMacros && !finalAnswer.includes('$') && !finalAnswer.includes('\\[')
    ? `$$${finalAnswer}$$`
    : finalAnswer;

  return (
    <div className="relative mt-3 p-3.5 rounded border border-stone-300 dark:border-stone-700 bg-stone-50 dark:bg-[#16181c]">
      <div className="flex items-center justify-between gap-2 mb-2 pb-1.5 border-b border-stone-200 dark:border-stone-800">
        <div className="flex items-center gap-1.5">
          <span className="text-stone-400 dark:text-stone-500 font-serif font-bold text-xs select-none">
            ■
          </span>
          <span className="font-serif font-semibold text-xs tracking-wide uppercase text-stone-800 dark:text-stone-200">
            Result
          </span>
        </div>

        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-2 py-0.5 text-xs rounded bg-white dark:bg-stone-800 hover:bg-stone-100 dark:hover:bg-stone-700 text-stone-700 dark:text-stone-300 border border-stone-300 dark:border-stone-700 transition-colors"
          title="Copy Answer"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
              <span className="text-emerald-700 dark:text-emerald-300 text-[11px]">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3 text-stone-500" />
              <span className="text-[11px]">Copy</span>
            </>
          )}
        </button>
      </div>

      <div className="text-sm md:text-base font-serif text-stone-900 dark:text-stone-100 leading-relaxed pl-1">
        <MathRenderer content={displayContent} />
      </div>
    </div>
  );
};

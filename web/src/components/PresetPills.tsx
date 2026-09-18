import React from 'react';
import { Sparkles, Calculator, BookOpen, Compass, Layers } from 'lucide-react';

interface Preset {
  id: string;
  label: string;
  prompt: string;
  icon: React.ReactNode;
}

interface PresetPillsProps {
  onSelectPrompt: (prompt: string) => void;
  disabled?: boolean;
}

export const PRESET_PROBLEMS: Preset[] = [
  {
    id: 'rate-problem',
    label: 'Rate & Proportion',
    prompt: 'A car travels 240 miles in 4 hours at a constant speed. How many hours will it take to travel 420 miles at the same rate?',
    icon: <Calculator className="w-3.5 h-3.5 text-stone-500 dark:text-stone-400" />,
  },
  {
    id: 'quadratic-derivation',
    label: 'Quadratic Formula Derivation',
    prompt: 'Derive the quadratic formula step-by-step by completing the square on the general equation ax^2 + bx + c = 0.',
    icon: <BookOpen className="w-3.5 h-3.5 text-stone-500 dark:text-stone-400" />,
  },
  {
    id: 'circle-area',
    label: 'Circle Area via Limits',
    prompt: 'Explain step-by-step why the area of a circle is A = \\pi r^2 by slicing the circle into infinite triangular sectors.',
    icon: <Compass className="w-3.5 h-3.5 text-stone-500 dark:text-stone-400" />,
  },
  {
    id: 'linear-system',
    label: 'Linear System (2 eq)',
    prompt: 'Solve the linear system of equations step-by-step: 3x + 2y = 16 and 2x - y = 6. Find both x and y.',
    icon: <Layers className="w-3.5 h-3.5 text-stone-500 dark:text-stone-400" />,
  },
  {
    id: 'prime-factorization',
    label: 'Prime Factorization',
    prompt: 'Find the prime factorization of 1008 using step-by-step division and express the final result in exponential form.',
    icon: <BookOpen className="w-3.5 h-3.5 text-stone-500 dark:text-stone-400" />,
  },
];

export const PresetPills: React.FC<PresetPillsProps> = ({ onSelectPrompt, disabled }) => {
  return (
    <div className="w-full my-4">
      <div className="flex items-center gap-1.5 mb-2.5 text-xs font-serif tracking-wider uppercase text-stone-500 dark:text-stone-400">
        <span>Sample Exercises</span>
      </div>
      <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-thin">
        {PRESET_PROBLEMS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => onSelectPrompt(preset.prompt)}
            disabled={disabled}
            className="flex-shrink-0 flex items-center gap-2 px-3 py-1.5 rounded border border-stone-300 dark:border-stone-700 bg-white dark:bg-[#1a1c20] hover:bg-stone-100 dark:hover:bg-[#23262d] text-stone-700 dark:text-stone-300 text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {preset.icon}
            <span>{preset.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

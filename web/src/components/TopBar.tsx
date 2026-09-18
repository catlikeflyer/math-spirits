import React from 'react';
import {
  Server,
  Cpu,
  Download,
  RotateCcw,
  Sun,
  Moon,
  ChevronDown,
  Activity,
} from 'lucide-react';
import { ModelInfo, SystemStatus, EngineMode } from '../types';

interface TopBarProps {
  models: ModelInfo[];
  activeModelId: string;
  onSelectModel: (modelId: string) => void;
  systemStatus: SystemStatus | null;
  engineMode: EngineMode;
  onToggleEngineMode: (mode: EngineMode) => void;
  isBackendHealthy: boolean;
  onExportMarkdown: () => void;
  onResetChat: () => void;
  isDarkMode: boolean;
  onToggleDarkMode: () => void;
  isSwitchingModel?: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({
  models,
  activeModelId,
  onSelectModel,
  systemStatus,
  engineMode,
  onToggleEngineMode,
  isBackendHealthy,
  onExportMarkdown,
  onResetChat,
  isDarkMode,
  onToggleDarkMode,
  isSwitchingModel = false,
}) => {
  const activeModel = models.find((m) => m.id === activeModelId) || models[0];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-stone-200 dark:border-[#282b32] bg-[#fcfbf9]/95 dark:bg-[#131416]/95 backdrop-blur-sm">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        {/* Logo & Title */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-7 h-7 rounded border border-stone-300 dark:border-stone-700 bg-stone-100 dark:bg-stone-800 text-stone-800 dark:text-stone-200 font-serif font-bold text-sm select-none">
            ∑
          </div>

          <div>
            <div className="flex items-center gap-2">
              <span className="font-serif text-sm sm:text-base font-semibold tracking-tight text-stone-900 dark:text-stone-100">
                Math Spirits
              </span>
              <span className="hidden sm:inline-block px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider bg-stone-100 dark:bg-stone-800 text-stone-600 dark:text-stone-400 border border-stone-200 dark:border-stone-700">
                Inference
              </span>
            </div>
            <p className="text-[11px] text-stone-500 dark:text-stone-400 hidden sm:block font-sans">
              Modular Mathematical Reasoning Engine
            </p>
          </div>
        </div>

        {/* Dynamic Model Selector & Badges */}
        <div className="flex items-center gap-2 md:gap-3 flex-1 justify-center max-w-lg">
          {/* Model Dropdown */}
          <div className="relative">
            <select
              value={activeModelId}
              onChange={(e) => onSelectModel(e.target.value)}
              disabled={isSwitchingModel}
              className="appearance-none pl-3 pr-8 py-1.5 rounded bg-stone-50 dark:bg-[#1a1c20] hover:bg-stone-100 dark:hover:bg-[#22252c] text-xs sm:text-sm font-medium text-stone-800 dark:text-stone-200 border border-stone-300 dark:border-stone-700 focus:outline-none focus:border-stone-500 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id} className="bg-white dark:bg-stone-900 text-stone-900 dark:text-stone-100">
                  {m.display_name || m.name} ({m.parameters})
                </option>
              ))}
              {models.length === 0 && (
                <option value={activeModelId || "math-spirits"} className="bg-white dark:bg-stone-900 text-stone-900 dark:text-stone-100">
                  Math Spirits
                </option>
              )}
            </select>
            <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-stone-500 pointer-events-none" />
          </div>

          {/* Model Status Badges */}
          <div className="hidden lg:flex items-center gap-1.5 text-[11px] font-mono">
            {/* Quantization */}
            <span
              className="px-2 py-0.5 rounded bg-stone-100 dark:bg-stone-800/80 text-stone-700 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80"
              title="Model Quantization"
            >
              {activeModel?.quantization || 'Q4_K_M'}
            </span>

            {/* Parameter Count */}
            <span
              className="px-2 py-0.5 rounded bg-stone-100 dark:bg-stone-800/80 text-stone-700 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80"
              title="Parameters"
            >
              {activeModel?.parameters || '1.5B'}
            </span>

            {/* Hardware Telemetry Badge */}
            {systemStatus && (
              <div
                className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-stone-100 dark:bg-stone-800/80 text-stone-700 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80"
                title={`Apple Metal VRAM: ${systemStatus.metal_active_gb} GB, System RAM: ${systemStatus.ram_used_gb}/${systemStatus.ram_total_gb} GB (${systemStatus.ram_percent}%)`}
              >
                <Activity className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                <span>
                  {systemStatus.metal_active_gb > 0
                    ? `${systemStatus.metal_active_gb.toFixed(1)} GB Metal`
                    : `${systemStatus.ram_used_gb.toFixed(1)} GB RAM`}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Right Section: Dual-Mode Toggle & Session Actions */}
        <div className="flex items-center gap-2">
          {/* Dual-Mode Toggle */}
          <div className="flex items-center p-0.5 rounded bg-stone-100 dark:bg-stone-800/80 border border-stone-200 dark:border-stone-700/80">
            <button
              onClick={() => onToggleEngineMode('backend')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                engineMode === 'backend'
                  ? 'bg-white dark:bg-stone-700 text-stone-900 dark:text-stone-100 shadow-xs'
                  : 'text-stone-500 dark:text-stone-400 hover:text-stone-800 dark:hover:text-stone-200'
              }`}
              title="Connect to local FastAPI + Apple Silicon MLX engine"
            >
              <Server className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Backend</span>
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  isBackendHealthy ? 'bg-emerald-500' : 'bg-red-500'
                }`}
                title={isBackendHealthy ? 'Backend Online' : 'Backend Offline'}
              />
            </button>

            <button
              onClick={() => onToggleEngineMode('webgpu')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                engineMode === 'webgpu'
                  ? 'bg-white dark:bg-stone-700 text-stone-900 dark:text-stone-100 shadow-xs'
                  : 'text-stone-500 dark:text-stone-400 hover:text-stone-800 dark:hover:text-stone-200'
              }`}
              title="Run in-browser via WebGPU (@mlc-ai/web-llm)"
            >
              <Cpu className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">WebGPU</span>
            </button>
          </div>

          {/* Export to Markdown */}
          <button
            onClick={onExportMarkdown}
            className="p-1.5 rounded bg-white dark:bg-stone-800/80 hover:bg-stone-100 dark:hover:bg-stone-700 text-stone-600 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80 transition-colors"
            title="Export session to Markdown (.md)"
            aria-label="Export Markdown"
          >
            <Download className="w-3.5 h-3.5" />
          </button>

          {/* Reset Chat */}
          <button
            onClick={onResetChat}
            className="p-1.5 rounded bg-white dark:bg-stone-800/80 hover:bg-stone-100 dark:hover:bg-stone-700 text-stone-600 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80 transition-colors"
            title="Reset conversation"
            aria-label="Reset conversation"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          {/* Dark / Light Toggle */}
          <button
            onClick={onToggleDarkMode}
            className="p-1.5 rounded bg-white dark:bg-stone-800/80 hover:bg-stone-100 dark:hover:bg-stone-700 text-stone-600 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80 transition-colors"
            title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label="Toggle theme"
          >
            {isDarkMode ? (
              <Sun className="w-3.5 h-3.5 text-stone-300" />
            ) : (
              <Moon className="w-3.5 h-3.5 text-stone-700" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
};

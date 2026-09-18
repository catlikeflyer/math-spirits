import React from 'react';
import { Zap, Clock, Hash, HardDrive } from 'lucide-react';
import { Telemetry } from '../types';

interface TelemetryBadgeProps {
  telemetry?: Telemetry;
  isStreaming?: boolean;
}

export const TelemetryBadge: React.FC<TelemetryBadgeProps> = ({ telemetry, isStreaming }) => {
  if (!telemetry && !isStreaming) {
    return null;
  }

  if (isStreaming) {
    return (
      <div className="flex items-center gap-2 mt-2.5 pt-2 border-t border-stone-200 dark:border-stone-800 text-[11px] font-mono text-stone-500 dark:text-stone-400">
        <span className="w-1.5 h-1.5 rounded-full bg-stone-500 dark:bg-stone-400 animate-pulse" />
        <span>Synthesizing mathematical response...</span>
      </div>
    );
  }

  if (!telemetry) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 md:gap-3 mt-3 pt-2 border-t border-stone-200 dark:border-stone-800/80 text-[11px] font-mono text-stone-500 dark:text-stone-400">
      {/* Tokens / Sec */}
      <div className="flex items-center gap-1" title="Tokens per second">
        <Zap className="w-3 h-3 text-stone-500" />
        <span>{telemetry.tokens_per_second.toFixed(1)} t/s</span>
      </div>
      <span>·</span>

      {/* TTFT */}
      <div className="flex items-center gap-1" title="Time to first token">
        <Clock className="w-3 h-3 text-stone-500" />
        <span>TTFT {telemetry.time_to_first_token_ms.toFixed(0)}ms</span>
      </div>
      <span>·</span>

      {/* Total Tokens & Duration */}
      <div className="flex items-center gap-1" title="Generated token count & total time">
        <Hash className="w-3 h-3 text-stone-500" />
        <span>{telemetry.total_tokens} tokens ({telemetry.total_duration_s.toFixed(1)}s)</span>
      </div>

      {/* Memory Footprint */}
      {telemetry.peak_memory_gb !== undefined && telemetry.peak_memory_gb > 0 && (
        <>
          <span>·</span>
          <div className="flex items-center gap-1" title="Peak Metal VRAM">
            <HardDrive className="w-3 h-3 text-stone-500" />
            <span>{telemetry.peak_memory_gb.toFixed(2)} GB VRAM</span>
          </div>
        </>
      )}
    </div>
  );
};

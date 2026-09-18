export interface ModelInfo {
  id: string;
  name: string;
  display_name: string;
  base_model: string;
  adapter_path: string;
  quantization: string;
  parameters: string;
  context_window: number;
  type: string;
  description: string;
  system_prompt?: string;
  is_active: boolean;
}

export interface Telemetry {
  tokens_per_second: number;
  time_to_first_token_ms: number;
  total_tokens: number;
  total_duration_s: number;
  active_memory_gb?: number;
  peak_memory_gb?: number;
  model_id?: string;
}

export interface SystemStatus {
  active_model_id: string;
  active_model_name: string;
  quantization: string;
  parameters: string;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_percent: number;
  metal_active_gb: number;
  metal_peak_gb: number;
  is_loaded: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'system' | 'user' | 'assistant';
  content: string;
  thought?: string;
  final_answer?: string;
  telemetry?: Telemetry;
  isStreaming?: boolean;
  timestamp: number;
}

export type EngineMode = 'backend' | 'webgpu';

export interface WebGPULoadProgress {
  progress: number;
  text: string;
}

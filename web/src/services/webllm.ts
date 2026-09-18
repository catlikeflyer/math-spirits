import {
  CreateMLCEngine,
  MLCEngineInterface,
  InitProgressReport,
  prebuiltAppConfig,
  AppConfig,
} from '@mlc-ai/web-llm';
import { Telemetry, WebGPULoadProgress } from '../types';

// ─── Fine-tuned model definitions ────────────────────────────────────────────
// HuggingFace repos containing MLC-compiled fine-tuned weights.
// The .wasm model library is reused from the prebuilt mlc-ai registry —
// same Qwen2 architecture means the compiled kernel is identical;
// only the weight shards differ (these are OUR fine-tuned weights).
//
// TODO: Replace <HF_USERNAME> with your actual HuggingFace username
// after running scripts/upload_hf.sh
const HF_USERNAME = 'catlikeflyer';

const MATH_GHOST_MLC_ID   = 'math-ghost-1-q4f16_1-MLC';
const MATH_SPECTRE_MLC_ID = 'math-spectre-1-q4f16_1-MLC';

// Prebuilt wasm libraries from mlc-ai (Qwen2 architecture, same version as @mlc-ai/web-llm)
const GHOST_BASE_ID   = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC';
const SPECTRE_BASE_ID = 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC';

function getPrebuiltModelLib(baseId: string): string {
  const entry = prebuiltAppConfig.model_list.find((m) => m.model_id === baseId);
  if (!entry?.model_lib) {
    throw new Error(
      `Cannot find prebuilt model_lib for ${baseId}. ` +
      `Ensure @mlc-ai/web-llm version matches the compiled artifacts.`
    );
  }
  return entry.model_lib as string;
}

// Custom AppConfig that loads YOUR fine-tuned weights but reuses prebuilt wasm.
// Falls back to prebuilt base models if HF_USERNAME is not yet set.
function buildAppConfig(): AppConfig {
  const isConfigured = (HF_USERNAME as string) !== '<HF_USERNAME>' && (HF_USERNAME as string).trim() !== '';

  if (!isConfigured) {
    // Fallback: use base models from mlc-ai until HF_USERNAME is configured.
    // This is clearly labeled in the UI via isFinetuned flag below.
    return prebuiltAppConfig;
  }

  return {
    model_list: [
      {
        model: `https://huggingface.co/${HF_USERNAME}/math-ghost-1-q4f16_1-MLC/resolve/main/`,
        model_id: MATH_GHOST_MLC_ID,
        model_lib: getPrebuiltModelLib(GHOST_BASE_ID),
      },
      {
        model: `https://huggingface.co/${HF_USERNAME}/math-spectre-1-q4f16_1-MLC/resolve/main/`,
        model_id: MATH_SPECTRE_MLC_ID,
        model_lib: getPrebuiltModelLib(SPECTRE_BASE_ID),
      },
      // Also include the full prebuilt list so other models still work if needed.
      ...prebuiltAppConfig.model_list,
    ],
  };
}

// ─── Model map ────────────────────────────────────────────────────────────────
export interface WebGPUModelEntry {
  mlcModelId: string;
  displayName: string;
  /** True when running our fine-tuned weights (not stock base model) */
  isFinetuned: boolean;
}

const isConfigured = (HF_USERNAME as string) !== '<HF_USERNAME>' && (HF_USERNAME as string).trim() !== '';

export const WEBGPU_MODEL_MAP: Record<string, WebGPUModelEntry> = {
  'math-ghost-1': {
    mlcModelId: isConfigured ? MATH_GHOST_MLC_ID : GHOST_BASE_ID,
    displayName: isConfigured ? 'Ghost 1 (Fine-tuned WebGPU)' : 'Ghost 1 (Base · WebGPU)',
    isFinetuned: isConfigured,
  },
  'math-spectre-1': {
    mlcModelId: isConfigured ? MATH_SPECTRE_MLC_ID : SPECTRE_BASE_ID,
    displayName: isConfigured ? 'Spectre 1 (Fine-tuned WebGPU)' : 'Spectre 1 (Base · WebGPU)',
    isFinetuned: isConfigured,
  },
};

export const DEFAULT_WEBGPU_MODEL = isConfigured ? MATH_GHOST_MLC_ID : GHOST_BASE_ID;

export function getWebGPUModelForId(modelId?: string): string {
  if (modelId && WEBGPU_MODEL_MAP[modelId]) {
    return WEBGPU_MODEL_MAP[modelId].mlcModelId;
  }
  if (modelId?.toLowerCase().includes('spectre')) {
    return isConfigured ? MATH_SPECTRE_MLC_ID : SPECTRE_BASE_ID;
  }
  return DEFAULT_WEBGPU_MODEL;
}

/** Returns true when running actual fine-tuned weights in WebGPU mode */
export function isWebGPUFinetuned(modelId?: string): boolean {
  if (!modelId) return false;
  return WEBGPU_MODEL_MAP[modelId]?.isFinetuned ?? false;
}

// ─── Engine state ─────────────────────────────────────────────────────────────
let engineInstance: MLCEngineInterface | null = null;
let currentLoadedModel: string | null = null;

export function isWebGPUSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export function getCurrentWebGPUModel(): string | null {
  return currentLoadedModel;
}

// ─── Engine init ──────────────────────────────────────────────────────────────
export async function initWebGPUEngine(
  modelId: string = DEFAULT_WEBGPU_MODEL,
  onProgress?: (progress: WebGPULoadProgress) => void
): Promise<MLCEngineInterface> {
  if (engineInstance && currentLoadedModel === modelId) {
    return engineInstance;
  }

  if (!isWebGPUSupported()) {
    throw new Error(
      'WebGPU is not supported in this browser. Please use Chrome, Edge, or enable WebGPU flags.'
    );
  }

  // Destroy previous engine before loading a new model to free VRAM
  if (engineInstance) {
    try {
      await (engineInstance as any).unload?.();
    } catch {
      // unload may not exist in all versions — ignore
    }
    engineInstance = null;
    currentLoadedModel = null;
  }

  const progressCallback = (report: InitProgressReport) => {
    if (onProgress) {
      onProgress({
        progress: Math.round(report.progress * 100),
        text: report.text,
      });
    }
  };

  const appConfig = buildAppConfig();

  try {
    engineInstance = await CreateMLCEngine(modelId, {
      appConfig,
      initProgressCallback: progressCallback,
      logLevel: 'INFO',
    });
  } catch (err) {
    // If custom fine-tuned model failed to load (e.g. not yet uploaded to HF),
    // fallback gracefully to prebuilt base model so WebGPU still functions
    if (modelId === MATH_GHOST_MLC_ID || modelId === MATH_SPECTRE_MLC_ID) {
      console.warn(
        `[WebLLM] Failed to load custom weights for ${modelId} from Hugging Face. Falling back to base prebuilt model. Error:`,
        err
      );
      const fallbackId = modelId === MATH_GHOST_MLC_ID ? GHOST_BASE_ID : SPECTRE_BASE_ID;
      engineInstance = await CreateMLCEngine(fallbackId, {
        appConfig: prebuiltAppConfig,
        initProgressCallback: progressCallback,
        logLevel: 'INFO',
      });
      currentLoadedModel = fallbackId;
      return engineInstance;
    }
    throw err;
  }

  currentLoadedModel = modelId;
  return engineInstance;
}

// ─── Streaming chat ───────────────────────────────────────────────────────────
export async function streamWebGPUChat(
  messages: { role: string; content: string }[],
  options: { temperature?: number; max_tokens?: number; systemPrompt?: string },
  onToken: (token: string) => void,
  onDone: (telemetry: Telemetry) => void,
  onError: (error: Error) => void
): Promise<void> {
  try {
    if (!engineInstance) {
      throw new Error('WebGPU engine is not initialized. Please load the model first.');
    }

    const startTime = performance.now();
    let firstTokenTime: number | null = null;
    let generatedTokens = 0;

    // Inject system prompt if provided and not already present
    let formattedMessages = [...messages];
    if (options.systemPrompt && !formattedMessages.some((m) => m.role === 'system')) {
      formattedMessages = [
        { role: 'system', content: options.systemPrompt },
        ...formattedMessages,
      ];
    }

    const stream = await engineInstance.chat.completions.create({
      messages: formattedMessages as any,
      temperature: options.temperature ?? 0.7,
      max_tokens: options.max_tokens ?? 1024,
      stream: true,
    });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content || '';
      if (delta) {
        if (firstTokenTime === null) {
          firstTokenTime = performance.now();
        }
        generatedTokens += 1;
        onToken(delta);
      }
    }

    const endTime = performance.now();
    const totalDuration = (endTime - startTime) / 1000.0;
    const ttftMs = firstTokenTime ? firstTokenTime - startTime : 0;
    const genDuration = firstTokenTime ? (endTime - firstTokenTime) / 1000.0 : totalDuration;
    const tps = genDuration > 0 && generatedTokens > 0 ? generatedTokens / genDuration : 0;

    onDone({
      tokens_per_second: Math.round(tps * 100) / 100,
      time_to_first_token_ms: Math.round(ttftMs),
      total_tokens: generatedTokens,
      total_duration_s: Math.round(totalDuration * 100) / 100,
      model_id: currentLoadedModel || 'webgpu-local',
    });
  } catch (err: any) {
    onError(err);
  }
}

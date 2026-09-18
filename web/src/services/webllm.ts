import { CreateMLCEngine, MLCEngineInterface, InitProgressReport } from '@mlc-ai/web-llm';
import { Telemetry, WebGPULoadProgress } from '../types';

// Model mapping for WebGPU
export const WEBGPU_MODEL_MAP: Record<string, { mlcModelId: string; displayName: string }> = {
  'math-ghost-1': {
    mlcModelId: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    displayName: 'Ghost 1 (0.5B WebGPU)',
  },
  'math-spectre-1': {
    mlcModelId: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
    displayName: 'Spectre 1 (1.5B WebGPU)',
  },
};

export const DEFAULT_WEBGPU_MODEL = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC';

export function getWebGPUModelForId(modelId?: string): string {
  if (modelId && WEBGPU_MODEL_MAP[modelId]) {
    return WEBGPU_MODEL_MAP[modelId].mlcModelId;
  }
  if (modelId?.toLowerCase().includes('spectre')) {
    return 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC';
  }
  return 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC';
}

let engineInstance: MLCEngineInterface | null = null;
let currentLoadedModel: string | null = null;

export function isWebGPUSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export function getCurrentWebGPUModel(): string | null {
  return currentLoadedModel;
}

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

  const progressCallback = (report: InitProgressReport) => {
    if (onProgress) {
      onProgress({
        progress: Math.round(report.progress * 100),
        text: report.text,
      });
    }
  };

  engineInstance = await CreateMLCEngine(modelId, {
    initProgressCallback: progressCallback,
    logLevel: 'INFO',
  });

  currentLoadedModel = modelId;
  return engineInstance;
}

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

import { ModelInfo, SystemStatus, Telemetry } from '../types';

const API_BASE = ''; // Proxy in vite handles /api -> http://localhost:8000/api

export async function fetchModels(): Promise<{ models: ModelInfo[]; active_model_id: string }> {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) {
    throw new Error(`Failed to fetch models: ${res.statusText}`);
  }
  return res.json();
}

export async function selectModel(modelId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/api/models/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || 'Failed to select model');
  }
  return res.json();
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/api/system/status`);
  if (!res.ok) {
    throw new Error(`Failed to fetch system status: ${res.statusText}`);
  }
  return res.json();
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

export interface StreamChatOptions {
  model?: string;
  temperature?: number;
  max_tokens?: number;
  system_prompt?: string;
  signal?: AbortSignal;
}

export async function streamChatCompletion(
  messages: { role: string; content: string }[],
  options: StreamChatOptions,
  onToken: (token: string) => void,
  onDone: (telemetry: Telemetry) => void,
  onError: (error: Error) => void
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE}/api/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages,
        model: options.model,
        temperature: options.temperature ?? 0.7,
        max_tokens: options.max_tokens ?? 1024,
        system_prompt: options.system_prompt,
        stream: true,
      }),
      signal: options.signal,
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Server returned status ${response.status}: ${errText}`);
    }

    if (!response.body) {
      throw new Error('ReadableStream not supported by response');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;

        const jsonStr = trimmed.slice(5).trim();
        try {
          const data = JSON.parse(jsonStr);
          if (data.type === 'token') {
            onToken(data.content);
          } else if (data.type === 'done') {
            onDone(data.telemetry);
          } else if (data.type === 'error') {
            throw new Error(data.error || 'Unknown streaming error');
          }
        } catch (e) {
          // Ignore JSON parse errors for incomplete chunks
        }
      }
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      return;
    }
    onError(err);
  }
}

import React, { useState, useEffect, useRef } from 'react';
import {
  AlertTriangle,
  RefreshCw,
  Cpu,
} from 'lucide-react';
import { ModelInfo, SystemStatus, ChatMessage, EngineMode, WebGPULoadProgress } from './types';
import {
  fetchModels,
  selectModel,
  fetchSystemStatus,
  checkBackendHealth,
  streamChatCompletion,
} from './services/api';
import {
  isWebGPUSupported,
  initWebGPUEngine,
  streamWebGPUChat,
  DEFAULT_WEBGPU_MODEL,
  getWebGPUModelForId,
  getCurrentWebGPUModel,
  isWebGPUFinetuned,
} from './services/webllm';
import { TopBar } from './components/TopBar';
import { ChatCanvas } from './components/ChatCanvas';

export const DEFAULT_FALLBACK_MODELS: ModelInfo[] = [
  {
    id: 'math-ghost-1',
    name: 'Math Ghost 1',
    display_name: 'Ghost 1 (Ultra-Light)',
    base_model: 'Qwen/Qwen2.5-0.5B-Instruct',
    adapter_path: './models/math-ghost-1/adapters',
    quantization: 'Q4_K_M',
    parameters: '0.5B',
    context_window: 4096,
    type: 'mlx',
    system_prompt: 'You are Ghost, an ultra-fast, lightweight math tutor spirit. Always think through the problem step-by-step inside <thought> tags before providing the final answer.',
    description: 'Ultra-lightweight 0.5B parameter fine-tuned model for sub-second mathematical deduction and reasoning on low-spec hardware and mobile devices.',
    is_active: true,
  },
  {
    id: 'math-spectre-1',
    name: 'Math Spectre 1',
    display_name: 'Spectre 1 (Math Spirit)',
    base_model: 'Qwen/Qwen2.5-Math-1.5B',
    adapter_path: './models/math-spectre-1/adapters',
    quantization: 'Q4_K_M',
    parameters: '1.5B',
    context_window: 4096,
    type: 'mlx',
    system_prompt: 'You are Spectre, an expert math tutor. Always think through the solution step-by-step inside <thought> tags before giving the final answer.',
    description: 'Fine-tuned Qwen2.5-Math-1.5B with step-by-step chain-of-thought reasoning scratchpad on GSM8K and NuminaMath.',
    is_active: false,
  },
];

export const App: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>(DEFAULT_FALLBACK_MODELS);
  const [activeModelId, setActiveModelId] = useState<string>('math-ghost-1');
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [engineMode, setEngineMode] = useState<EngineMode>('backend');
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean>(true);
  const [isSwitchingModel, setIsSwitchingModel] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isDarkMode, setIsDarkMode] = useState<boolean>(true);

  // WebGPU state
  const [webgpuLoading, setWebgpuLoading] = useState<boolean>(false);
  const [webgpuProgress, setWebgpuProgress] = useState<WebGPULoadProgress | null>(null);
  const [webgpuReady, setWebgpuReady] = useState<boolean>(false);
  const [webgpuIsFinetuned, setWebgpuIsFinetuned] = useState<boolean>(isWebGPUFinetuned('math-ghost-1'));

  const abortControllerRef = useRef<AbortController | null>(null);
  const activeModel = models.find((m) => m.id === activeModelId) || models[0] || null;

  // Initialize theme
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
      setIsDarkMode(false);
      document.documentElement.classList.remove('dark');
      document.documentElement.classList.add('light');
    } else {
      setIsDarkMode(true);
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    }
  }, []);

  const handleToggleDarkMode = () => {
    if (isDarkMode) {
      setIsDarkMode(false);
      localStorage.setItem('theme', 'light');
      document.documentElement.classList.remove('dark');
      document.documentElement.classList.add('light');
    } else {
      setIsDarkMode(true);
      localStorage.setItem('theme', 'dark');
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    }
  };

  // Poll backend health & system telemetry
  const refreshBackendData = async () => {
    try {
      const healthy = await checkBackendHealth();
      setIsBackendHealthy(healthy);

      if (healthy) {
        const { models: discovered, active_model_id } = await fetchModels();
        if (discovered && discovered.length > 0) {
          setModels(discovered);
        }
        if (active_model_id) {
          setActiveModelId(active_model_id);
        }

        const status = await fetchSystemStatus();
        setSystemStatus(status);
      }
    } catch {
      setIsBackendHealthy(false);
    }
  };

  useEffect(() => {
    refreshBackendData();
    const interval = setInterval(refreshBackendData, 6000);
    return () => clearInterval(interval);
  }, []);

  // Handle Model Selection
  const handleSelectModel = async (modelId: string) => {
    setIsSwitchingModel(true);
    try {
      setActiveModelId(modelId);
      if (engineMode === 'backend') {
        await selectModel(modelId);
        const status = await fetchSystemStatus();
        setSystemStatus(status);
      } else {
        // WebGPU Mode: switch WebGPU model if changed
        const targetWebgpuModel = getWebGPUModelForId(modelId);
        if (targetWebgpuModel !== getCurrentWebGPUModel()) {
          setWebgpuReady(false);
          setWebgpuIsFinetuned(false);
          setWebgpuLoading(true);
          await initWebGPUEngine(targetWebgpuModel, (progress) => {
            setWebgpuProgress(progress);
          });
          setWebgpuReady(true);
          setWebgpuIsFinetuned(isWebGPUFinetuned(modelId));
        }
      }
    } catch (err: any) {
      alert(`Failed to switch model: ${err.message}`);
    } finally {
      setIsSwitchingModel(false);
      setWebgpuLoading(false);
    }
  };

  // Handle Engine Mode Switch
  const handleToggleEngineMode = async (mode: EngineMode) => {
    setEngineMode(mode);
    if (mode === 'webgpu') {
      const targetWebgpuModel = getWebGPUModelForId(activeModelId);
      if (getCurrentWebGPUModel() !== targetWebgpuModel || !webgpuReady) {
        if (!isWebGPUSupported()) {
          alert('WebGPU is not supported by your browser. Please use Chrome/Edge or stick to Backend Mode.');
          setEngineMode('backend');
          return;
        }
        setWebgpuLoading(true);
        try {
          await initWebGPUEngine(targetWebgpuModel, (progress) => {
            setWebgpuProgress(progress);
          });
          setWebgpuReady(true);
          setWebgpuIsFinetuned(isWebGPUFinetuned(activeModelId));
        } catch (err: any) {
          alert(`WebGPU Initialization Error: ${err.message}`);
          setEngineMode('backend');
        } finally {
          setWebgpuLoading(false);
        }
      }
    }
  };

  // Stop Generation
  const handleStopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
    setMessages((prev) =>
      prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
    );
  };

  // Send Message
  const handleSendMessage = async (userPrompt: string) => {
    if (!userPrompt.trim() || isGenerating) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: userPrompt.trim(),
      timestamp: Date.now(),
    };

    const assistantMsgId = `assistant-${Date.now()}`;
    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsGenerating(true);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const conversationHistory = [...messages, userMsg].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    const onToken = (token: string) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId ? { ...m, content: m.content + token } : m
        )
      );
    };

    const onDone = (telemetry: any) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? { ...m, isStreaming: false, telemetry }
            : m
        )
      );
      setIsGenerating(false);
      abortControllerRef.current = null;
      // Refresh memory telemetry
      fetchSystemStatus().then((st) => setSystemStatus(st)).catch(() => {});
    };

    const onError = (err: Error) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? {
                ...m,
                content:
                  m.content +
                  `\n\n> ⚠️ **Inference Error**: ${err.message}. If the backend is offline, switch to WebGPU Mode.`,
                isStreaming: false,
              }
            : m
        )
      );
      setIsGenerating(false);
      abortControllerRef.current = null;
    };

    if (engineMode === 'backend') {
      await streamChatCompletion(
        conversationHistory,
        {
          model: activeModelId,
          temperature: 0.7,
          max_tokens: 1024,
          signal: abortController.signal,
        },
        onToken,
        onDone,
        onError
      );
    } else {
      // WebGPU Mode
      try {
        const targetWebgpuModel = getWebGPUModelForId(activeModelId);
        if (!webgpuReady || getCurrentWebGPUModel() !== targetWebgpuModel) {
          setWebgpuLoading(true);
          await initWebGPUEngine(targetWebgpuModel, (progress) => {
            setWebgpuProgress(progress);
          });
          setWebgpuReady(true);
          setWebgpuLoading(false);
        }
        await streamWebGPUChat(
          conversationHistory,
          {
            temperature: 0.7,
            max_tokens: 1024,
            systemPrompt: activeModel?.system_prompt,
          },
          onToken,
          onDone,
          onError
        );
      } catch (err: any) {
        onError(err);
      }
    }
  };

  // Reset Chat
  const handleResetChat = () => {
    if (isGenerating) {
      handleStopGeneration();
    }
    setMessages([]);
  };

  // Export to Markdown
  const handleExportMarkdown = () => {
    if (messages.length === 0) {
      alert('No messages in conversation to export.');
      return;
    }

    let md = `# Math Spirits Session\n`;
    md += `*Generated: ${new Date().toLocaleString()}*\n`;
    md += `*Model: ${activeModel?.display_name || activeModelId || 'Math Spirits'} (${engineMode.toUpperCase()} Mode)*\n\n---\n\n`;

    messages.forEach((msg) => {
      if (msg.role === 'user') {
        md += `### 🧑 User\n\n${msg.content}\n\n`;
      } else if (msg.role === 'assistant') {
        md += `### Math Spirits\n\n`;
        md += `${msg.content}\n\n`;
        if (msg.telemetry) {
          md += `> **Telemetry**: ${msg.telemetry.tokens_per_second} t/s | TTFT: ${msg.telemetry.time_to_first_token_ms}ms | Total tokens: ${msg.telemetry.total_tokens} | Duration: ${msg.telemetry.total_duration_s}s\n\n`;
        }
        md += `---\n\n`;
      }
    });

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `math-spirits-session-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="relative flex flex-col min-h-screen bg-[#fcfbf9] dark:bg-[#131416] text-[#1c1917] dark:text-[#eceae3] selection:bg-stone-300 dark:selection:bg-stone-800 transition-colors duration-150">
      {/* Top Bar Header */}
      <TopBar
        models={models}
        activeModelId={activeModelId}
        onSelectModel={handleSelectModel}
        systemStatus={systemStatus}
        engineMode={engineMode}
        onToggleEngineMode={handleToggleEngineMode}
        isBackendHealthy={isBackendHealthy}
        onExportMarkdown={handleExportMarkdown}
        onResetChat={handleResetChat}
        isDarkMode={isDarkMode}
        onToggleDarkMode={handleToggleDarkMode}
        isSwitchingModel={isSwitchingModel}
        isWebGPUFinetuned={webgpuIsFinetuned}
      />

      {/* Backend Offline Fallback Banner */}
      {!isBackendHealthy && engineMode === 'backend' && (
        <div className="relative z-40 bg-amber-50 dark:bg-amber-950/30 border-b border-amber-200 dark:border-amber-800/40 px-4 py-2 text-xs text-amber-800 dark:text-amber-200/90 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 flex-shrink-0" />
            <span>
              <strong>Backend Offline:</strong> Local server not responding. Run{' '}
              <code className="bg-amber-100 dark:bg-black/40 px-1 py-0.5 rounded font-mono text-[11px] text-amber-900 dark:text-amber-300">
                python3 server/app.py
              </code>
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleToggleEngineMode('webgpu')}
              className="flex items-center gap-1 px-2 py-1 rounded bg-stone-800 hover:bg-stone-700 text-stone-100 font-medium text-xs transition-colors"
            >
              <Cpu className="w-3 h-3" />
              <span>Switch to WebGPU</span>
            </button>
            <button
              onClick={refreshBackendData}
              className="flex items-center gap-1 px-2 py-1 rounded bg-stone-200 dark:bg-stone-800 hover:bg-stone-300 dark:hover:bg-stone-700 text-stone-800 dark:text-stone-200 text-xs transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Retry</span>
            </button>
          </div>
        </div>
      )}

      {/* WebGPU Loading Overlay Banner */}
      {webgpuLoading && webgpuProgress && (
        <div className="relative z-40 bg-stone-100 dark:bg-stone-900 border-b border-stone-200 dark:border-stone-800 px-4 py-2 text-xs text-stone-700 dark:text-stone-300 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-stone-600 dark:text-stone-400 animate-spin" />
            <span>Loading WebGPU model in-browser: {webgpuProgress.text}</span>
          </div>
          <div className="w-32 bg-stone-200 dark:bg-stone-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-stone-600 dark:bg-stone-400 h-full transition-all duration-200"
              style={{ width: `${webgpuProgress.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Chat & Math Canvas Container */}
      <main className="relative z-10 flex-1 flex flex-col">
        <ChatCanvas
          messages={messages}
          onSendMessage={handleSendMessage}
          onStopGeneration={handleStopGeneration}
          isGenerating={isGenerating}
          activeModel={models.find((m) => m.id === activeModelId) || models[0] || null}
        />
      </main>
    </div>
  );
};

export default App;

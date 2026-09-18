import React, { useRef, useEffect, useState } from 'react';
import {
  Send,
  Square,
  User,
  Code2,
} from 'lucide-react';
import { ModelInfo, ChatMessage } from '../types';
import { ReasoningScratchpad } from './ReasoningScratchpad';
import { FinalAnswerCard } from './FinalAnswerCard';
import { TelemetryBadge } from './TelemetryBadge';
import { MathRenderer } from './MathRenderer';
import { PresetPills } from './PresetPills';

interface ChatCanvasProps {
  messages: ChatMessage[];
  onSendMessage: (content: string) => void;
  onStopGeneration: () => void;
  isGenerating: boolean;
  activeModel?: ModelInfo | null;
}

// Helper to extract LaTeX \\boxed{...} with arbitrary nested brace depth
export function extractBoxedContent(text: string): string | null {
  const marker = '\\boxed{';
  const idx = text.lastIndexOf(marker);
  if (idx === -1) return null;
  let depth = 0;
  let end = -1;
  for (let i = idx + marker.length - 1; i < text.length; i++) {
    if (text[i] === '{') {
      depth++;
    } else if (text[i] === '}') {
      depth--;
      if (depth === 0) {
        end = i;
        break;
      }
    }
  }
  if (end !== -1) {
    return text.substring(idx, end + 1);
  }
  return null;
}

export function parseThoughtAndAnswer(rawContent: string) {
  let thought = '';
  let answer = '';
  let solutionExplanation = '';
  let finalAnswer = '';

  const thoughtStartIdx = rawContent.indexOf('<thought>');
  const thoughtEndIdx = rawContent.indexOf('</thought>');

  if (thoughtStartIdx !== -1) {
    if (thoughtEndIdx !== -1) {
      thought = rawContent.slice(thoughtStartIdx + 9, thoughtEndIdx).trim();
      answer = rawContent.slice(thoughtEndIdx + 10).trim();
    } else {
      thought = rawContent.slice(thoughtStartIdx + 9).trim();
      answer = '';
    }
  } else {
    answer = rawContent.trim();
  }

  // Detect and truncate repeated hallucinated "Final Answer:" loops
  const lowerAnswer = answer.toLowerCase();
  const firstFaIdx = lowerAnswer.indexOf('final answer:');
  if (firstFaIdx !== -1) {
    const secondFaIdx = lowerAnswer.indexOf('final answer:', firstFaIdx + 13);
    if (secondFaIdx !== -1) {
      answer = answer.slice(0, secondFaIdx).trim();
    }
  }

  // Strip hallucinated image tags, </latex>, or web OCR artifacts
  answer = answer.split(/<\/latex>|!\[/i)[0].trim();
  answer = answer.replace(/<\|im_end\|>|<\|endoftext\|>|alignSelf/gi, '').trim();

  // Check for LaTeX boxed answer anywhere in the answer
  const boxedAnswer = extractBoxedContent(answer);

  // Parse out explicit final answer if present
  const finalAnswerRegex = /Final Answer:\s*([\s\S]*)/i;
  const match = answer.match(finalAnswerRegex);
  if (match) {
    solutionExplanation = answer.slice(0, match.index).trim();
    const rawFa = match[1].trim();
    // Only take the first non-empty line of the final answer
    const faLines = rawFa.split('\n').map((l) => l.trim()).filter(Boolean);
    let firstValidLine = faLines.find((l) => !l.startsWith('!') && !l.startsWith('[')) || faLines[0] || '';

    // Clean up hallucinated crawler artifacts, OCR tags, image markdowns
    firstValidLine = firstValidLine
      .replace(/!\[.*?\]\([^\)]*\)/g, '')
      .replace(/!.*$/, '') // any trailing !VIP, !竞赛, etc.
      .replace(/<\/latex>|<\|im_end\|>|<\|endoftext\|>|alignSelf/gi, '')
      // Strip any non-ASCII / non-Greek / non-Math Unicode noise (Korean, Thai, Chinese web crawler spam)
      .replace(/[^\x20-\x7E\u0370-\u03FF\u2190-\u22FF]/g, '')
      .replace(/[;]+$/, '') // trailing semicolons
      .trim();

    // Check brace balance and incomplete \frac
    const openBraces = (firstValidLine.match(/\{/g) || []).length;
    const closeBraces = (firstValidLine.match(/\}/g) || []).length;
    const isIncompleteFrac = firstValidLine.includes('\\frac') && (firstValidLine.match(/\{/g) || []).length < 2;

    // If a boxed answer exists in the solution, prefer the boxed answer as it is the canonical golden answer
    if (boxedAnswer) {
      const unwrapped = boxedAnswer.startsWith('\\boxed{') && boxedAnswer.endsWith('}')
        ? boxedAnswer.slice(7, -1).trim()
        : boxedAnswer;
      finalAnswer = unwrapped;
      // Remove duplicate boxed formula from the end of solutionExplanation
      solutionExplanation = solutionExplanation
        .replace(boxedAnswer, '')
        .replace(/(?:We can box the final answer as follows|The final answer is|Hence the final answer is)[:\s]*$/i, '')
        .trim();
    } else {
      // Auto-balance braces if open > close
      const openBraces = (firstValidLine.match(/\{/g) || []).length;
      const closeBraces = (firstValidLine.match(/\}/g) || []).length;
      if (openBraces > closeBraces) {
        firstValidLine += '}'.repeat(openBraces - closeBraces);
      }
      finalAnswer = firstValidLine;
    }
  } else if (boxedAnswer) {
    const unwrapped = boxedAnswer.startsWith('\\boxed{') && boxedAnswer.endsWith('}')
      ? boxedAnswer.slice(7, -1).trim()
      : boxedAnswer;
    finalAnswer = unwrapped;
    solutionExplanation = answer
      .replace(boxedAnswer, '')
      .replace(/(?:We can box the final answer as follows|The final answer is|Hence the final answer is)[:\s]*$/i, '')
      .trim();
  } else {
    solutionExplanation = answer;
  }

  return { thought, answer, solutionExplanation, finalAnswer };
}

export const ChatCanvas: React.FC<ChatCanvasProps> = ({
  messages,
  onSendMessage,
  onStopGeneration,
  isGenerating,
  activeModel,
}) => {
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (!inputText.trim() || isGenerating) return;
    onSendMessage(inputText.trim());
    setInputText('');
  };

  const insertMathSnippet = (snippet: string) => {
    setInputText((prev) => prev + snippet);
    textareaRef.current?.focus();
  };

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-3.5rem)] max-w-4xl mx-auto w-full px-4 sm:px-6">
      {/* Conversation Scroll Area */}
      <div className="flex-1 overflow-y-auto py-6 space-y-6">
        {/* Welcome Empty State */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center min-h-[50vh] text-center px-4 max-w-2xl mx-auto">
            <div className="w-11 h-11 rounded border border-stone-300 dark:border-stone-700 bg-stone-100 dark:bg-stone-800/80 flex items-center justify-center text-stone-800 dark:text-stone-200 font-serif font-bold text-lg mb-3 select-none">
              ∑
            </div>

            <h2 className="text-xl sm:text-2xl font-serif font-semibold tracking-tight text-stone-900 dark:text-stone-100">
              Mathematical Reasoning & Deduction
            </h2>
            <p className="mt-2 text-xs sm:text-sm text-stone-600 dark:text-stone-400 leading-relaxed font-sans max-w-lg">
              A minimalist small language model environment designed for deductive proof exploration, stepwise derivations, and formal problem solving.
            </p>

            {/* Quick Test Presets */}
            <div className="w-full mt-5">
              <PresetPills onSelectPrompt={onSendMessage} disabled={isGenerating} />
            </div>
          </div>
        )}

        {/* Message Thread */}
        {messages.map((message) => {
          const isUser = message.role === 'user';
          const { thought, answer, solutionExplanation, finalAnswer } = parseThoughtAndAnswer(message.content);

          return (
            <div
              key={message.id}
              className={`flex gap-3 ${
                isUser ? 'justify-end' : 'justify-start'
              }`}
            >
              {/* Assistant Avatar */}
              {!isUser && (
                <div className="flex-shrink-0 w-7 h-7 rounded border border-stone-300 dark:border-stone-700 bg-stone-100 dark:bg-stone-800 flex items-center justify-center text-stone-700 dark:text-stone-300 text-xs font-serif font-bold select-none">
                  ∑
                </div>
              )}

              {/* Message Bubble */}
              <div
                className={`max-w-[88%] sm:max-w-[80%] rounded p-4 transition-colors ${
                  isUser
                    ? 'bg-stone-100 dark:bg-[#1a1c22] text-stone-900 dark:text-stone-100 border border-stone-200 dark:border-stone-800'
                    : 'bg-white dark:bg-[#16181c] text-stone-900 dark:text-stone-100 border border-stone-200 dark:border-[#282b32]'
                }`}
              >
                {isUser ? (
                  <div className="text-sm font-sans font-medium whitespace-pre-wrap leading-relaxed">
                    <MathRenderer content={message.content} />
                  </div>
                ) : (
                  <div>
                    {/* Spirits Header */}
                    <div className="flex items-center justify-between gap-2 mb-2 pb-1.5 border-b border-stone-100 dark:border-stone-800/80">
                      <div className="flex items-center gap-2">
                        <span className="font-serif font-semibold text-xs tracking-wider uppercase text-stone-700 dark:text-stone-300">
                          Math Spirits
                        </span>
                        {message.isStreaming && (
                          <span className="text-[10px] font-mono text-stone-500">
                            Deducing...
                          </span>
                        )}
                      </div>
                      <span className="text-[10px] font-mono text-stone-400">
                        {new Date(message.timestamp).toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </span>
                    </div>

                    {/* Step-by-Step Chain of Thought Scratchpad */}
                    {(thought || message.isStreaming) && (
                      <ReasoningScratchpad
                        thoughtContent={thought}
                        isStreaming={message.isStreaming}
                        spiritName="Spirits"
                      />
                    )}

                    {/* Explanation / Solution text */}
                    {solutionExplanation ? (
                      <div className="mt-2.5 text-sm sm:text-base leading-relaxed text-stone-800 dark:text-stone-200 font-serif">
                        <MathRenderer content={solutionExplanation} />
                      </div>
                    ) : (
                      !finalAnswer && answer && (
                        <div className="mt-2.5 text-sm sm:text-base leading-relaxed text-stone-800 dark:text-stone-200 font-serif">
                          <MathRenderer content={answer} />
                        </div>
                      )
                    )}

                    {/* Final Answer Summary Card */}
                    {finalAnswer && <FinalAnswerCard finalAnswer={finalAnswer} />}

                    {/* Telemetry Dashboard Stats */}
                    <TelemetryBadge
                      telemetry={message.telemetry}
                      isStreaming={message.isStreaming}
                    />
                  </div>
                )}
              </div>

              {/* User Avatar */}
              {isUser && (
                <div className="flex-shrink-0 w-7 h-7 rounded border border-stone-300 dark:border-stone-700 bg-stone-100 dark:bg-stone-800 flex items-center justify-center text-stone-500 dark:text-stone-400">
                  <User className="w-3.5 h-3.5" />
                </div>
              )}
            </div>
          );
        })}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Section */}
      <div className="pb-4 pt-1">
        {/* Math notation shortcuts */}
        <div className="flex items-center gap-1.5 mb-2 overflow-x-auto pb-1 text-xs text-stone-500 dark:text-stone-400">
          <span className="text-[11px] font-mono text-stone-400 dark:text-stone-500 mr-1 flex items-center gap-1">
            <Code2 className="w-3 h-3" /> LaTeX:
          </span>
          <button
            onClick={() => insertMathSnippet('$\\frac{a}{b}$ ')}
            className="px-2 py-0.5 rounded bg-stone-100 dark:bg-[#1a1c20] hover:bg-stone-200 dark:hover:bg-[#252830] text-stone-700 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80 transition-colors font-mono text-xs"
          >
            \frac&#123;a&#125;&#123;b&#125;
          </button>
          <button
            onClick={() => insertMathSnippet('$\\sqrt{x}$ ')}
            className="px-2 py-0.5 rounded bg-stone-100 dark:bg-[#1a1c20] hover:bg-stone-200 dark:hover:bg-[#252830] text-stone-700 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80 transition-colors font-mono text-xs"
          >
            \sqrt&#123;x&#125;
          </button>
          <button
            onClick={() => insertMathSnippet('$x^2$ ')}
            className="px-2 py-0.5 rounded bg-stone-100 dark:bg-[#1a1c20] hover:bg-stone-200 dark:hover:bg-[#252830] text-stone-700 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80 transition-colors font-mono text-xs"
          >
            x^2
          </button>
          <button
            onClick={() => insertMathSnippet('$\\pi$ ')}
            className="px-2 py-0.5 rounded bg-stone-100 dark:bg-[#1a1c20] hover:bg-stone-200 dark:hover:bg-[#252830] text-stone-700 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80 transition-colors font-mono text-xs"
          >
            \pi
          </button>
          <button
            onClick={() => insertMathSnippet('$\\sum_{i=1}^{n}$ ')}
            className="px-2 py-0.5 rounded bg-stone-100 dark:bg-[#1a1c20] hover:bg-stone-200 dark:hover:bg-[#252830] text-stone-700 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80 transition-colors font-mono text-xs"
          >
            \sum
          </button>
          <button
            onClick={() => insertMathSnippet('$\\boxed{answer}$ ')}
            className="px-2 py-0.5 rounded bg-stone-100 dark:bg-[#1a1c20] hover:bg-stone-200 dark:hover:bg-[#252830] text-stone-700 dark:text-stone-300 border border-stone-200 dark:border-stone-700/80 transition-colors font-mono text-xs"
          >
            \boxed&#123;&#125;
          </button>
        </div>

        {/* Input Bar */}
        <div className="relative rounded border border-stone-300 dark:border-[#2a2d35] bg-white dark:bg-[#17191d] focus-within:border-stone-500 dark:focus-within:border-stone-500 transition-colors">
          <textarea
            ref={textareaRef}
            rows={2}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter a mathematical problem, proof request, or calculation..."
            disabled={isGenerating}
            className="w-full bg-transparent px-3.5 py-2.5 text-sm sm:text-base text-stone-900 dark:text-stone-100 placeholder-stone-400 dark:placeholder-stone-500 focus:outline-none resize-none font-sans"
          />

          <div className="flex items-center justify-between px-3 py-2 border-t border-stone-100 dark:border-stone-800 bg-stone-50/50 dark:bg-black/10">
            <span className="text-[11px] text-stone-400 hidden sm:inline font-sans">
              Press <kbd className="px-1 py-0.5 rounded bg-stone-200 dark:bg-stone-800 text-stone-600 dark:text-stone-300 font-mono text-[10px]">Enter</kbd> to solve, <kbd className="px-1 py-0.5 rounded bg-stone-200 dark:bg-stone-800 text-stone-600 dark:text-stone-300 font-mono text-[10px]">Shift + Enter</kbd> for newline
            </span>

            <div className="flex items-center gap-2 ml-auto">
              {isGenerating ? (
                <button
                  onClick={onStopGeneration}
                  className="flex items-center gap-1.5 px-3 py-1 rounded bg-stone-800 hover:bg-stone-700 text-white text-xs font-medium transition-colors"
                >
                  <Square className="w-3 h-3 fill-current" />
                  <span>Halt</span>
                </button>
              ) : (
                <button
                  onClick={handleSubmit}
                  disabled={!inputText.trim()}
                  className="flex items-center gap-1.5 px-3.5 py-1 rounded bg-stone-900 dark:bg-stone-100 text-white dark:text-stone-900 hover:bg-stone-800 dark:hover:bg-white text-xs font-medium transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <span>Solve</span>
                  <Send className="w-3 h-3" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

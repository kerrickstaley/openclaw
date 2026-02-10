import { appendFileSync } from 'node:fs';
import { Type } from '@sinclair/typebox';
import { createSubsystemLogger } from '../logging/subsystem.js';
import { extractToolResultText } from './pi-embedded-subscribe.tools.js';
import type { AnyAgentTool } from './pi-tools.types.js';
import {
  createRedactedToolResult,
  PROMPT_INJECTION_THRESHOLD,
  scoreForPromptInjection,
} from './prompt-injection-monitor.js';
import { jsonResult } from './tools/common.js';

const log = createSubsystemLogger('agents/prompt-injection-monitor');

const MIN_TEXT_LENGTH = 50;
const DEBUG_LOG = '/tmp/openclaw-pi-monitor.log';

function debugLog(msg: string) {
  const ts = new Date().toISOString();
  const line = `[${ts}] ${msg}\n`;
  try {
    appendFileSync(DEBUG_LOG, line);
  } catch {
    // ignore write errors
  }
}

export type MonitorState = { skipNext: boolean };

export function createMonitorState(): MonitorState {
  return { skipNext: false };
}

export function wrapToolWithPromptInjectionMonitor(tool: AnyAgentTool, state: MonitorState): AnyAgentTool {
  if (tool.name === 'disable_pi_monitor') {
    return tool;
  }
  if (!process.env.OPENAI_API_KEY) {
    debugLog(`SKIP wrapping tool "${tool.name}" — OPENAI_API_KEY not set`);
    return tool;
  }
  const execute = tool.execute;
  if (!execute) {
    debugLog(`SKIP wrapping tool "${tool.name}" — no execute method`);
    return tool;
  }
  const toolName = tool.name || 'tool';
  debugLog(`WRAPPED tool "${toolName}"`);
  return {
    ...tool,
    execute: async (toolCallId, params, signal, onUpdate) => {
      debugLog(`EXECUTE tool="${toolName}" toolCallId=${toolCallId}`);
      const result = await execute(toolCallId, params, signal, onUpdate);

      if (state.skipNext) {
        state.skipNext = false;
        debugLog(`SKIP monitoring tool="${toolName}" — disabled by disable_pi_monitor`);
        return result;
      }

      const text = extractToolResultText(result);
      if (!text || text.length < MIN_TEXT_LENGTH) {
        debugLog(`SKIP scoring tool="${toolName}" — text too short (${text?.length ?? 0} chars)`);
        return result;
      }

      debugLog(`SCORING tool="${toolName}" text=${text.length} chars, preview: ${JSON.stringify(text.slice(0, 200))}`);

      try {
        const { score, reasoning } = await scoreForPromptInjection(text, toolName);
        debugLog(`SCORED tool="${toolName}" score=${score}/100 reasoning=${JSON.stringify(reasoning)}`);
        if (score >= PROMPT_INJECTION_THRESHOLD) {
          log.warn(
            `Prompt injection detected in tool "${toolName}" (score: ${score}/100): ${reasoning}`,
          );
          debugLog(`REDACTED tool="${toolName}" score=${score}`);
          return createRedactedToolResult(toolName, score) as typeof result;
        }
        debugLog(`PASSED tool="${toolName}" score=${score}`);
        return result;
      } catch (err) {
        log.warn(`Prompt injection scoring failed for tool "${toolName}": ${String(err)}`);
        debugLog(`ERROR tool="${toolName}" err=${String(err)} — REDACTING (fail closed)`);
        return createRedactedToolResult(toolName, -1) as typeof result;
      }
    },
  };
}

export function createDisablePiMonitorTool(state: MonitorState): AnyAgentTool {
  return {
    name: 'disable_pi_monitor',
    label: 'Disable PI Monitor',
    description:
      'Disables prompt injection monitoring for the next tool call only. Use when the user has reviewed a redacted result and confirmed it is safe. The bypass is consumed after one tool execution.',
    parameters: Type.Object({}),
    execute: async () => {
      state.skipNext = true;
      debugLog('disable_pi_monitor called — skipNext set to true');
      return jsonResult({ ok: true, message: 'Prompt injection monitoring disabled for the next tool call.' });
    },
  };
}

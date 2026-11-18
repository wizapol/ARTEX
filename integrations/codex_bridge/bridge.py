#!/usr/bin/env python3
"""Local ARTEX Chat Completions adapter using the signed-in official Codex CLI.

Tools are returned to ARTEX as data. They are never executed by this adapter.
Each completion uses an ephemeral Codex session with a structured output schema.
"""
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import signal
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jsonschema

CODE_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('ARTEX_BRIDGE_RUNTIME', CODE_DIR))
MODEL = 'gpt-6.2-sol'
KEY = (ROOT / 'bridge-key').read_text().strip()
GATE = threading.BoundedSemaphore(1)
MAX_BODY = 4 * 1024 * 1024


def command(output):
    args = ['/usr/lib/chatgpt/resources/codex', 'exec', '--ephemeral',
            '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
            '--sandbox', 'read-only', '--cd', str(ROOT / 'empty'),
            '--model', MODEL, '--json', '--color', 'never']
    for feature in ['shell_tool', 'unified_exec', 'apps', 'code_mode',
                    'multi_agent', 'multi_agent_v2', 'hooks', 'browser_use',
                    'computer_use', 'tool_suggest', 'shell_snapshot', 'shell_snapshot_v2']:
        args += ['--disable', feature]
    for setting in ['plugins._default.enabled=false', 'agents.enabled=false',
                    'web_search="disabled"', 'features.skip_host_skill_discovery=true',
                    'tools.update_plan.enabled=false', 'model_reasoning_effort="low"']:
        args += ['-c', setting]
    return args + ['--output-schema', str(CODE_DIR / 'response-schema.json'),
                   '--output-last-message', str(output), '-']


def prepare(body):
    if body.get('model') != MODEL:
        raise ValueError('Select the supported model: ' + MODEL)
    if body.get('stream'):
        raise ValueError('This adapter requires streaming=false in the LLM profile')
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a non-empty array')
    tools = {}
    entries = body.get('tools') or []
    if not isinstance(entries, list):
        raise ValueError('tools must be an array')
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('function'), dict):
            raise ValueError('Each tool must contain a function definition')
        if entry.get('type') != 'function':
            raise ValueError('Only function tools are supported')
        fn = entry['function']
        if not isinstance(fn.get('name'), str) or not fn['name']:
            raise ValueError('Function name must be a non-empty string')
        if fn['name'] in tools:
            raise ValueError('Duplicate tool name')
        tools[fn['name']] = fn
    choice = body.get('tool_choice', 'auto')
    if isinstance(choice, dict):
        if choice.get('type') != 'function' or not isinstance(choice.get('function'), dict):
            raise ValueError('Invalid tool_choice function definition')
        if choice['function'].get('name') not in tools:
            raise ValueError('tool_choice must name a supplied function')
    elif choice not in ['auto', 'none', 'required']:
        raise ValueError('Invalid tool_choice')
    if choice == 'required' and not tools:
        raise ValueError('tool_choice required needs at least one tool')
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ['system', 'developer', 'user', 'assistant', 'tool']:
            raise ValueError('Invalid conversation message')
    prompt = (
        'You are the inference backend for ARTEX. Answer the next ASSISTANT turn. '
        'The function definitions in this request describe tools executed by ARTEX in its container. '
        'The local CLI tool settings apply only to this inference process. They do not disable '
        'the ARTEX tools. Request ARTEX tools using the structured tool_calls response. '
        'Diagnose connectivity only from actual ARTEX tool results. A WebFetch refusal to '
        'download a static resource describes that tool limitation, not a network outage. '
        'Preserve all applicable authorization, approval and safety requirements. '
        'If a tool is needed, return its exact name and JSON object arguments as a STRING in '
        'tool_calls. ARTEX will execute it and send the actual result in the next request. '
        'Never invent tool results. Use only the function definitions provided. '
        'Use completed tool results already present in the conversation. Do not repeat a '
        'completed tool request unless new evidence or the user request requires it. '
        'Respect tool_choice (none forbids calls; required requires a call). '
        'Return only the response schema.\n\nARTEX REQUEST:\n'
        + json.dumps({k: body[k] for k in ['messages', 'tools', 'tool_choice'] if k in body},
                     ensure_ascii=False)
    )
    return prompt, tools


def convert(result, body, tools):
    jsonschema.validate(result, json.loads((CODE_DIR / 'response-schema.json').read_text()))
    calls = []
    choice = body.get('tool_choice', 'auto')
    for call in result['tool_calls']:
        name = call['name']
        if choice == 'none' or name not in tools:
            raise ValueError('Model returned a tool that was not allowed')
        if isinstance(choice, dict) and choice.get('function', {}).get('name') != name:
            raise ValueError('Model did not select the required function')
        try:
            arguments = json.loads(call['arguments'])
        except json.JSONDecodeError as exc:
            raise ValueError('Invalid JSON in tool arguments') from exc
        if not isinstance(arguments, dict):
            raise ValueError('Tool arguments must be a JSON object')
        jsonschema.validate(arguments, tools[name].get('parameters', {'type': 'object'}))
        calls.append({'id': 'call_' + secrets.token_hex(12), 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(arguments)}})
    if (choice == 'required' or isinstance(choice, dict)) and not calls:
        raise ValueError('Model did not return the required tool call')
    msg = {'role': 'assistant', 'content': result['content'] or None}
    if calls:
        msg['tool_calls'] = calls
    if not calls and not result['content']:
        raise ValueError('Empty model response')
    return {'id': 'chatcmpl-' + secrets.token_hex(12), 'object': 'chat.completion',
            'created': int(time.time()), 'model': MODEL,
            'choices': [{'index': 0, 'message': msg,
                         'finish_reason': 'tool_calls' if calls else 'stop'}]}


def complete_once(body):
    prompt, tools = prepare(body)
    env = {'HOME': '/home/wizapol', 'PATH': '/usr/local/bin:/usr/bin:/bin',
           'LANG': 'C.UTF-8'}
    # This is the existing host login; no OAuth tokens are copied into ARTEX.
    env['CODEX_HOME'] = '/home/wizapol/.codex'
    with tempfile.TemporaryDirectory(prefix='completion-', dir=ROOT) as tmp:
        output = Path(tmp) / 'answer.json'
        # Logs can contain prompts. Keep them private and remove them on completion.
        with (Path(tmp) / 'events.jsonl').open('w') as stdout, (Path(tmp) / 'error.log').open('w') as stderr:
            proc = subprocess.Popen(command(output), stdin=subprocess.PIPE, stdout=stdout,
                                    stderr=stderr, text=True, env=env,
                                    cwd=ROOT / 'empty', start_new_session=True)
            try:
                proc.communicate(prompt, timeout=80)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise RuntimeError('Codex timed out after 80s; no tool was executed')
        if proc.returncode:
            # Do not return stderr or credentials in HTTP responses.
            raise RuntimeError('Codex request failed; check the host login or subscription limits')
        if not output.exists():
            raise RuntimeError('Codex did not produce a structured response')
        answer = convert(json.loads(output.read_text()), body, tools)
        # Report actual CLI usage when available; never fabricate zero-token usage.
        for line in (Path(tmp) / 'events.jsonl').read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'turn.completed' and 'usage' in event:
                usage = event['usage']
                inp, out = usage.get('input_tokens', 0), usage.get('output_tokens', 0)
                answer['usage'] = {'prompt_tokens': inp, 'completion_tokens': out,
                                   'total_tokens': inp + out,
                                   'prompt_tokens_details': {'cached_tokens': usage.get('cached_input_tokens', 0)}}
        return answer


def complete(body):
    try:
        return complete_once(body)
    except (ValueError, jsonschema.ValidationError):
        logging.warning('Invalid structured response; retrying inference once')
    retry_body = dict(body)
    retry_body['messages'] = list(body['messages']) + [{
        'role': 'system',
        'content': 'The previous response failed JSON or function-schema validation. '
                   'Return a valid structured response for the same request. Encode function '
                   'arguments as a valid JSON object string matching the supplied schema. '
                   'Preserve tool_choice and all authorization and safety requirements. '
                   'Previously completed tool results remain valid.'}]
    return complete_once(retry_body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # No prompts, auth headers or response bodies in persistent logs.
        logging.info('%s %s', self.client_address[0], fmt % args)

    def send_json(self, status, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        if status == 429:
            self.send_header('Retry-After', '5')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            logging.info('client disconnected while writing response')

    def authenticated(self):
        return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + KEY)

    def do_GET(self):
        if self.path == '/health':
            return self.send_json(200, {'ok': True, 'adapter': 'codex-cli'})
        if not self.authenticated():
            return self.send_json(401, {'error': {'message': 'Unauthorized'}})
        if self.path == '/v1/models':
            return self.send_json(200, {'object': 'list', 'data': [
                {'id': MODEL, 'object': 'model', 'owned_by': 'local-codex-adapter'}]})
        self.send_json(404, {'error': {'message': 'Not found'}})

    def do_POST(self):
        if not self.authenticated():
            return self.send_json(401, {'error': {'message': 'Unauthorized'}})
        if self.path != '/v1/chat/completions':
            return self.send_json(404, {'error': {'message': 'Not found'}})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_BODY:
                return self.send_json(413, {'error': {'message': 'Invalid request size'}})
            self.connection.settimeout(200)
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object')
            prepare(body)
        except (ValueError, KeyError, TypeError) as exc:
            return self.send_json(400, {'error': {'message': str(exc)}})
        if not GATE.acquire(blocking=False):
            return self.send_json(429, {'error': {'message': 'Codex adapter is busy; retry later'}})
        start = time.monotonic()
        try:
            self.send_json(200, complete(body))
            logging.info('completion ok duration=%.1fs', time.monotonic() - start)
        except (ValueError, jsonschema.ValidationError) as exc:
            logging.warning('invalid model response: %s', type(exc).__name__)
            detail = 'Invalid JSON in tool arguments' if str(exc) == 'Invalid JSON in tool arguments' else 'Invalid structured model response'
            self.send_json(502, {'error': {'message': detail}})
        except RuntimeError as exc:
            logging.warning('%s', exc)
            self.send_json(502, {'error': {'message': str(exc)}})
        except (BrokenPipeError, ConnectionResetError):
            logging.info('client disconnected')
        except Exception:
            logging.exception('adapter failure')
            self.send_json(502, {'error': {'message': 'Local adapter failure'}})
        finally:
            GATE.release()


if __name__ == '__main__':
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    ThreadingHTTPServer(('172.20.0.1', 8789), Handler).serve_forever()

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


CODE_DIR = Path(__file__).resolve().parent


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = tempfile.TemporaryDirectory()
        Path(cls.runtime.name, 'bridge-key').write_text('test-key-only')
        with patch.dict(os.environ, {'ARTEX_BRIDGE_RUNTIME': cls.runtime.name,
                                    'ARTEX_CLAUDE_MODEL': 'claude-haiku-4-5'}):
            spec = importlib.util.spec_from_file_location('tested_bridge', CODE_DIR / 'bridge.py')
            cls.bridge = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.bridge)

    @classmethod
    def tearDownClass(cls):
        cls.runtime.cleanup()

    def test_instruction_roles_and_completed_tool_results(self):
        history = [
            {'role': 'system', 'content': 'Stay inside the laboratory scope.'},
            {'role': 'developer', 'content': [{'type': 'text', 'text': 'Require real evidence.'}]},
            {'role': 'user', 'content': 'Read the diagnostic file.'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call1', 'type': 'function', 'function': {'name': 'Read', 'arguments': '{"file_path":"/tmp/probe"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call1', 'content': 'line "one"\nline two'},
        ]
        body = {'model': 'claude-haiku-4-5', 'messages': history}
        text, tools = self.bridge.prepare(body)
        forwarded = json.loads(text[text.index('\n') + 1:])
        self.assertEqual(forwarded, history[2:])
        system = self.bridge.system_instructions(body, tools)
        self.assertIn(history[0]['content'], system)
        self.assertIn('Require real evidence.', system)
        self.assertNotIn(history[-1]['content'], system)

    def test_native_tool_inventory_matches_filtered_tools(self):
        body = {'messages': [], 'tool_choice': 'none'}
        system = self.bridge.system_instructions(body, {})
        self.assertIn('(no tools supplied for this request)', system)
        body['tool_choice'] = 'auto'
        system = self.bridge.system_instructions(body, {'Bash': {}})
        self.assertIn('Current ARTEX function inventory: Bash.', system)
        self.assertIn('mcp__artex__NAME', system)

    def test_usage_does_not_duplicate_streamed_blocks(self):
        events = [
            {'type': 'assistant', 'message': {'id': 'm1', 'usage': {
                'input_tokens': 10, 'output_tokens': 2, 'cache_read_input_tokens': 5}}},
            {'type': 'assistant', 'message': {'id': 'm1', 'usage': {
                'input_tokens': 10, 'output_tokens': 7, 'cache_read_input_tokens': 5}}},
        ]
        self.assertEqual(self.bridge.cli_usage(events)['total_tokens'], 22)
        events.append({'type': 'result', 'usage': {'input_tokens': 20, 'output_tokens': 9}})
        self.assertEqual(self.bridge.cli_usage(events)['total_tokens'], 29)

    def test_invalid_output_recovery_is_bounded(self):
        body = {'messages': [{'role': 'user', 'content': 'diagnostic'}]}
        with patch.object(self.bridge, 'complete_once', side_effect=[ValueError(), {'ok': True}]) as run:
            self.assertEqual(self.bridge.complete(body), {'ok': True})
            self.assertEqual(run.call_count, 2)
        self.assertEqual(len(body['messages']), 1)
        with patch.object(self.bridge, 'complete_once', side_effect=RuntimeError()) as run:
            with self.assertRaises(RuntimeError):
                self.bridge.complete(body)
            self.assertEqual(run.call_count, 1)

    def test_mcp_captures_a_call_without_executing_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / 'must-not-exist'
            (root / 'tools.json').write_text(json.dumps({'Bash': {
                'description': 'External executor', 'parameters': {'type': 'object'}}}))
            process = subprocess.Popen([sys.executable, str(CODE_DIR / 'mcp_relay.py'), str(root)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                def request(method, params):
                    process.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': 1,
                                                    'method': method, 'params': params}) + '\n')
                    process.stdin.flush()
                request('initialize', {})
                self.assertIn('tools', json.loads(process.stdout.readline())['result']['capabilities'])
                request('tools/list', {})
                self.assertEqual(json.loads(process.stdout.readline())['result']['tools'][0]['name'], 'Bash')
                request('tools/call', {'name': 'Bash', 'arguments': {'command': 'touch ' + str(marker)}})
                deadline = time.monotonic() + 3
                while not (root / 'call.json').exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                captured = json.loads((root / 'call.json').read_text())
                self.assertEqual(json.loads(captured['arguments'])['command'], 'touch ' + str(marker))
                self.assertFalse(marker.exists())
            finally:
                process.kill()
                process.wait()
                process.stdin.close()
                process.stdout.close()


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch
import importlib.util
import os
from pathlib import Path
import tempfile

_runtime = tempfile.TemporaryDirectory()
unittest.addModuleCleanup(_runtime.cleanup)
Path(_runtime.name, 'bridge-key').write_text('test-key-only')
with patch.dict(os.environ, {'ARTEX_BRIDGE_RUNTIME': _runtime.name}):
    spec = importlib.util.spec_from_file_location('tested_codex_bridge', Path(__file__).with_name('bridge.py'))
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.body = {'model': bridge.MODEL, 'messages': [{'role': 'user', 'content': 'test'}],
                     'tools': [{'type': 'function', 'function': {'name': 'echo', 'parameters': {
                         'type': 'object', 'properties': {'value': {'type': 'string'}},
                         'required': ['value'], 'additionalProperties': False}}}]}
        _, self.tools = bridge.prepare(self.body)

    def test_tool_call_is_returned_as_data(self):
        with patch.object(bridge.subprocess, 'Popen') as process:
            result = bridge.convert({'content': '', 'tool_calls': [
                {'name': 'echo', 'arguments': '{"value":"hello"}'}]}, self.body, self.tools)
            process.assert_not_called()
        self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(result['choices'][0]['message']['tool_calls'][0]['function']['name'], 'echo')

    def test_unauthorized_tool_rejected(self):
        with self.assertRaises(ValueError):
            bridge.convert({'content': '', 'tool_calls': [
                {'name': 'shell', 'arguments': '{}'}]}, self.body, self.tools)

    def test_invalid_arguments_rejected(self):
        with self.assertRaises(bridge.jsonschema.ValidationError):
            bridge.convert({'content': '', 'tool_calls': [
                {'name': 'echo', 'arguments': '{"value":1}'}]}, self.body, self.tools)

    def test_none_forbids_calls(self):
        self.body['tool_choice'] = 'none'
        with self.assertRaises(ValueError):
            bridge.convert({'content': '', 'tool_calls': [
                {'name': 'echo', 'arguments': '{"value":"hello"}'}]}, self.body, self.tools)

    def test_required_cannot_silently_return_text(self):
        self.body['tool_choice'] = 'required'
        with self.assertRaises(ValueError):
            bridge.convert({'content': 'done', 'tool_calls': []}, self.body, self.tools)

    def test_streaming_and_unknown_models_rejected(self):
        for change in [{'stream': True}, {'model': 'arbitrary'}]:
            with self.assertRaises(ValueError):
                bridge.prepare({**self.body, **change})

    def test_native_tools_are_disabled(self):
        args = bridge.command('/tmp/unused')
        self.assertIn('--ignore-user-config', args)
        self.assertIn('--ignore-rules', args)
        self.assertIn('read-only', args)
        for flag in ['shell_tool', 'unified_exec', 'multi_agent', 'apps', 'hooks']:
            self.assertEqual(args[args.index(flag)-1], '--disable')


if __name__ == '__main__':
    unittest.main()

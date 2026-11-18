"""Expose ARTEX tool schemas and capture one call for the external executor."""
import json
import os
from pathlib import Path
import sys
import time

root = Path(sys.argv[1])
tools = json.loads((root / 'tools.json').read_text())
for line in sys.stdin:
    message = json.loads(line)
    method = message.get('method')
    with (root / 'methods').open('a') as log:
        log.write(str(method) + '\n')
    if 'id' not in message:
        continue
    if method == 'initialize':
        result = {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}},
                  'serverInfo': {'name': 'artex-external-executor', 'version': '1'}}
    elif method == 'tools/list':
        result = {'tools': [{'name': name, 'description': fn.get('description', ''),
                             'inputSchema': fn.get('parameters', {'type': 'object'})}
                            for name, fn in tools.items()]}
    elif method == 'tools/call':
        params = message['params']
        if params['name'] not in tools:
            result = {'isError': True, 'content': [{'type': 'text', 'text': 'Unknown tool'}]}
        else:
            temporary = root / 'call.tmp'
            temporary.write_text(json.dumps({'name': params['name'], 'arguments': json.dumps(params.get('arguments', {}))}))
            os.replace(temporary, root / 'call.json')
            # The adapter returns this call to ARTEX. Execution and approvals remain there.
            while True:
                time.sleep(1)
    elif method == 'ping':
        result = {}
    else:
        print(json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'error': {'code': -32601, 'message': 'Unknown method'}}), flush=True)
        continue
    print(json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result}), flush=True)

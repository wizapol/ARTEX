# CLI model connections

These adapters connect ARTEX Chat Completions profiles to the operator's signed-in
Codex and Claude Code clients. Claude registers ARTEX functions through MCP and
returns captured calls to the application executor. Application instructions keep
their system/developer roles and conversation history retains tool-call IDs.

- `claude_bridge/` supports the Haiku 4.5 and Opus 4.6 profiles.
- `codex_bridge/` preserves the GPT-6.2-Sol connector.
- `systemd/` contains the installed host service configurations as deployment examples.

Runtime state belongs in an external private directory containing `bridge-key`
and `empty/`. Set `ARTEX_BRIDGE_RUNTIME` to that directory. The service examples
use `/srv/naztlan-apps/artex-runtime`; adapt their paths to the destination host.
When deploying these tracked sources, point each service ExecStart to the matching
`integrations/*_bridge/bridge.py` and set ARTEX_BRIDGE_RUNTIME to its external state.

Claude also requires ARTEX_CLAUDE_MODEL and ARTEX_CLAUDE_PORT. Its parallel request
limit is ARTEX_CLAUDE_MAX_PARALLEL (default 4), supporting simultaneous planner and
worker calls. Profiles use OpenAI format with streaming disabled. The bearer key
is the local adapter's bridge-key. Account login is managed by the host CLI.

Install Python dependencies and run the isolated regression checks:

```sh
python -m pip install -r integrations/requirements.txt
python -m unittest discover -s integrations -p test_bridge.py -v
```

The host verification completed diagnostic task 49 through goal decomposition,
planning, Bash/Write/Read, fact persistence and prove_goal. Separate checks covered
WebFetch, parallel requests and token reporting for MCP calls and final replies.

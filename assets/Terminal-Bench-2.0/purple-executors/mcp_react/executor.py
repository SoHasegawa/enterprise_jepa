"""Terminal-Bench-2.0 purple executor: WM-guided shell agent over native commands.

Terminal-Bench analogue of EnterpriseOps-Gym's ``mcp_react`` executor. The gym runs a
ReAct loop over MCP tool servers with an optional ``ejepa_wm`` world model; Terminal-Bench
has no MCP layer, so this reuses the sibling ``llm_shell`` executor's shell-command tool
loop and adds the same shared ``ejepa_wm`` world-model hook (see ``wm_react.WmShellExecutor``).

With ``WM_STRATEGY`` unset / ``none`` it behaves exactly like ``llm_shell``. Set
``WM_STRATEGY=selection WM_BACKEND=ewm_predict`` (+ ``WM_EWM_MCP_URL`` / ``WM_N``) to have
the EWM world model rank candidate commands, or ``WM_STRATEGY=prompt_injection
WM_BACKEND=ewm_imagined`` to inject imagined-trajectory guidance. See this directory's
``README.md`` for the full env-var list.
"""
import importlib.util
import sys
from pathlib import Path

from a2a.server.agent_execution import AgentExecutor

EXECUTOR_DIR = Path(__file__).resolve().parent
if str(EXECUTOR_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTOR_DIR))

# Load the sibling wm_react.py under a unique module name (CRMArenaPro's mcp_react ships a
# ``wm_react.py`` too; a bare ``import wm_react`` would collide in a shared process).
_spec = importlib.util.spec_from_file_location(
    "tb_mcp_react_wm", EXECUTOR_DIR / "wm_react.py"
)
wm_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = wm_module
_spec.loader.exec_module(wm_module)
WmShellExecutor = wm_module.WmShellExecutor


def build_executor() -> AgentExecutor:
    return WmShellExecutor()

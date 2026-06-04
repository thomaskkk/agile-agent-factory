import os
import shutil
import subprocess
from agile_agent_factory.config import AIDER_ENABLED, AIDER_MODEL, ANTHROPIC_API_KEY, OPENAI_API_KEY, LLM_TIMEOUT_SECONDS, PRODUCT_ROOT
from agile_agent_factory.tools.logger import log


def is_available() -> bool:
    if not AIDER_ENABLED:
        log("Aider disabled via AIDER_ENABLED=false.")
        return False
    if not shutil.which("aider"):
        log("Aider binary not found on PATH.")
        return False
    return True


def run_task(task_description: str, blueprint: str, review_feedback: str = "") -> dict[str, object]:
    feedback_section = (
        f"\n\nPrevious code review rejected this implementation. You MUST address all of the following:\n{review_feedback}"
        if review_feedback else ""
    )
    full_message = (
        "You are implementing a software product. "
        "Write code to app/ and tests to tests/. "
        f"Follow the blueprint exactly.\n\nBlueprint:\n{blueprint}\n\nTask:\n{task_description}"
        f"{feedback_section}"
    )
    cmd = [
        "aider",
        "--model", AIDER_MODEL,
        "--message", full_message,
        "--yes",
        "--no-git",
        "--no-auto-commits",
        "--chat-language", "english",
    ]
    if ANTHROPIC_API_KEY:
        cmd += ["--api-key", f"anthropic={ANTHROPIC_API_KEY}"]

    env = os.environ.copy()
    if ANTHROPIC_API_KEY:
        env["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
    if OPENAI_API_KEY:
        env["OPENAI_API_KEY"] = OPENAI_API_KEY

    timeout = LLM_TIMEOUT_SECONDS * 10
    log(f"Running aider from {PRODUCT_ROOT} with model {AIDER_MODEL}.")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PRODUCT_ROOT),
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log(f"Aider timed out after {timeout}s.")
        return {"success": False, "output": f"Aider timed out after {timeout}s."}
    except Exception as e:
        log(f"Aider subprocess error: {e}")
        return {"success": False, "output": str(e)}

    output = result.stdout + result.stderr
    log(f"Aider finished with exit code {result.returncode}.")
    return {"success": result.returncode == 0, "output": output}

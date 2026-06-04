import os
import subprocess

from agile_agent_factory.config import UV_BIN, PRODUCT_ROOT
from agile_agent_factory.tools.logger import log


def run_pytest(extra_packages: list[str] | None = None) -> tuple[int, str]:
    tests_dir = str(PRODUCT_ROOT / "tests")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PRODUCT_ROOT}:{existing}" if existing else str(PRODUCT_ROOT)
    )

    with_args = [f"--with={pkg}" for pkg in (extra_packages or [])]
    if extra_packages:
        log(f"Running pytest against {tests_dir} with extra packages: {', '.join(extra_packages)}.")
    else:
        log(f"Running pytest against {tests_dir}.")
    cmd = [UV_BIN, "run", *with_args, "python", "-m", "pytest", tests_dir, "-v"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PRODUCT_ROOT.parent),
    )
    output = result.stdout + result.stderr
    log(f"pytest finished with exit code {result.returncode}.")
    return result.returncode, output

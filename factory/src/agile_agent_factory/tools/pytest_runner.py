import os
import subprocess

from agile_agent_factory.config import UV_BIN, PRODUCT_ROOT, PYTEST_TIMEOUT_SECONDS
from agile_agent_factory.tools.logger import log


def run_pytest(
    extra_packages: list[str] | None = None,
    test_targets: list[str] | None = None,
) -> tuple[int, str]:
    tests_dir = str(PRODUCT_ROOT / "tests")
    targets = test_targets or [tests_dir]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PRODUCT_ROOT}:{existing}" if existing else str(PRODUCT_ROOT)
    )

    with_args = [f"--with={pkg}" for pkg in (extra_packages or [])]
    target_label = ", ".join(targets)
    if extra_packages:
        log(f"Running pytest against {target_label} with extra packages: {', '.join(extra_packages)}.")
    else:
        log(f"Running pytest against {target_label}.")
    cmd = [UV_BIN, "run", *with_args, "python", "-m", "pytest", *targets, "-v"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(PRODUCT_ROOT),
            timeout=PYTEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        output = (
            f"{stdout}{stderr}\n\n"
            f"pytest timed out after {PYTEST_TIMEOUT_SECONDS}s."
        ).strip()
        log(f"pytest timed out after {PYTEST_TIMEOUT_SECONDS}s.")
        return 124, output
    output = result.stdout + result.stderr
    log(f"pytest finished with exit code {result.returncode}.")
    return result.returncode, output

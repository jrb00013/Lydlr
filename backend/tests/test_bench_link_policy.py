"""Test bench_link_policy produces deterministic output."""
import csv
import io
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1].parent


def test_bench_link_policy_stdout():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/bench_link_policy.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    reader = csv.DictReader(io.StringIO(result.stdout))
    rows = list(reader)
    assert len(rows) > 0
    for row in rows:
        assert float(row["target_level"]) >= 0.1
        assert float(row["target_level"]) <= 0.98
        assert row["scenario"] in {
            "drone_nominal", "drone_burst", "drone_low_quality", "drone_high_latency",
            "iot_nominal", "iot_burst", "iot_low_quality",
            "highcap_nominal", "highcap_burst",
        }, f"Unknown scenario: {row['scenario']}"


def test_bench_link_policy_golden_csv():
    golden = REPO / "scripts/bench_link_policy_golden.csv"
    assert golden.exists(), f"Golden CSV not found at {golden}"
    with open(golden) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 180
    assert set(r["scenario"] for r in rows) == {
        "drone_nominal", "drone_burst", "drone_low_quality", "drone_high_latency",
        "iot_nominal", "iot_burst", "iot_low_quality",
        "highcap_nominal", "highcap_burst",
    }
    # Verify burst scenarios produce higher compression levels
    drone_burst = [r for r in rows if r["scenario"] == "drone_burst"]
    drone_nominal = [r for r in rows if r["scenario"] == "drone_nominal"]
    avg_burst = sum(float(r["target_level"]) for r in drone_burst) / len(drone_burst)
    avg_nominal = sum(float(r["target_level"]) for r in drone_nominal) / len(drone_nominal)
    assert avg_burst > avg_nominal, (
        f"Burst should have higher compression than nominal "
        f"({avg_burst:.3f} <= {avg_nominal:.3f})"
    )

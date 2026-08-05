import subprocess
import sys

import pandas as pd

DATA_PATH = "data/test_data.csv"


def test_cli_smoke(tmp_path):
    subset = pd.read_csv(DATA_PATH).head(40)
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    subset.to_csv(input_csv, index=False)

    result = subprocess.run(
        [
            sys.executable, "-m", "MOBSTERm.cli",
            str(input_csv), str(output_csv),
            "-i", "30",
            "-c", "2,3",
            "-S", "40",
            "-q",
        ],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_csv.is_file()

    output_df = pd.read_csv(output_csv)
    assert set(output_df.columns) == {"mutation_id", "cluster_id"}
    assert len(output_df) == len(subset)

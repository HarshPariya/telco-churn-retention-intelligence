"""Execute data validation script and output DATA_QUALITY_REPORT.md."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.data.ingest import load_raw_data
from src.telco_churn.data.validate import generate_markdown_report, validate_raw_data
from src.telco_churn.logging_config import configure_logger

logger = configure_logger("validate_data")


def main() -> None:
    load_config()
    df = load_raw_data()

    logger.info("Running programmatic data validation...")
    report = validate_raw_data(df)

    report_md = generate_markdown_report(report)
    out_path = PROJECT_ROOT / "docs" / "DATA_QUALITY_REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(f"Data quality report successfully written to {out_path}")
    if not report.validation_passed:
        logger.error("Data validation failed! Check report for critical errors.")
        sys.exit(1)
    logger.info("Data validation passed successfully.")


if __name__ == "__main__":
    main()

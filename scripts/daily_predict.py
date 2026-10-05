from __future__ import annotations

import argparse
from datetime import date

from dotenv import load_dotenv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a training prediction for a chosen date.")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="Date to predict (defaults to tomorrow); past dates and today are also supported.",
    )
    parser.add_argument("--local", action="store_true", help="Load environment variables from .env")
    args = parser.parse_args()
    if args.local:
        load_dotenv()
    from run_daily import main

    main(args.date)

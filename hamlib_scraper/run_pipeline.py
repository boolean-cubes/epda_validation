"""Download HamLib chemistry metadata and generate scaling reports."""

import argparse
import csv
import sys

from fit_scaling import add_fit_options, analyze_from_args
from render_report import render
from scrape_metadata import add_scrape_options, scrape_from_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scrape_options(parser)
    add_fit_options(parser)
    parser.add_argument(
        "--offline", action="store_true",
        help="Analyze existing metadata without making HTTP requests.",
    )
    args = parser.parse_args()

    try:
        if not args.offline and scrape_from_args(args):
            print(
                "The crawl was incomplete or found no metadata. See the manifest.",
                file=sys.stderr,
            )
            return 1
        analyze_from_args(args)
        render(args.report_dir)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

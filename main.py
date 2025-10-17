import argparse
from src.experiments import (
    run_univariate_experiments,
    run_multivariate_experiments,
    run_regression_analysis,
)
from src.reports_gen import ensure_report_templates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gd-univariate", action="store_true")
    parser.add_argument("--gd-multivariate", action="store_true")
    parser.add_argument("--reg-analysis", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--reports", action="store_true")
    args = parser.parse_args()

    if args.all or args.gd_univariate:
        run_univariate_experiments()
    if args.all or args.gd_multivariate:
        run_multivariate_experiments()
    if args.all or args.reg_analysis:
        run_regression_analysis()
    if args.all or args.reports:
        ensure_report_templates()


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Manual ASOS Scraper Runner
Run the scraper manually with custom options
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run ASOS scraper manually')
    parser.add_argument(
        '--categories',
        '-c',
        type=int,
        default=None,
        help='Maximum number of categories to scrape (default: all)'
    )
    parser.add_argument(
        '--sample-only',
        '-s',
        action='store_true',
        help='Run with sample data only (for testing)'
    )
    parser.add_argument(
        '--test',
        '-t',
        action='store_true',
        help='Run test mode with minimal categories'
    )
    parser.add_argument(
        '--browser',
        '-b',
        action='store_true',
        help='Use browser automation (bypasses API blocking)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    print("🚀 ASOS Scraper Manual Runner")
    print("=" * 50)
    print(f"Start Time: {datetime.now()}")
    print()

    # Determine which script to run
    if args.sample_only:
        script_name = "asos_scraper.py"
        script_args = []
        print("📝 Mode: Sample data only (using 3.txt)")
    elif args.browser:
        script_name = "asos_scraper_browser.py"
        if args.test:
            script_args = ["5"]  # Test with 5 categories
            print("🧪 Mode: Browser test mode (5 categories)")
        elif args.categories:
            script_args = [str(args.categories)]
            print(f"🎯 Mode: Browser custom categories ({args.categories})")
        else:
            script_args = ["10"]  # Default to 10 categories for browser mode (slower)
            print("🌍 Mode: Browser scrape (10 categories - browser mode is slower)")
    elif args.test:
        script_name = "asos_scraper_multi_category.py"
        script_args = ["5"]  # Test with 5 categories
        print("🧪 Mode: API test mode (5 categories - may fail due to blocking)")
    else:
        script_name = "asos_scraper_multi_category.py"
        if args.categories:
            script_args = [str(args.categories)]
            print(f"🎯 Mode: API custom categories ({args.categories} - may fail due to blocking)")
        else:
            script_args = []
            print("🌍 Mode: API full scrape (all categories - likely to fail due to blocking)")

    print(f"Script: {script_name}")
    print(f"Arguments: {script_args}")
    print()

    # Check if script exists
    if not os.path.exists(script_name):
        print(f"❌ Error: {script_name} not found!")
        sys.exit(1)

    # Check if virtual environment should be activated
    venv_path = os.path.join(os.getcwd(), 'venv', 'Scripts', 'activate')
    if os.path.exists(venv_path):
        print("🐍 Activating virtual environment...")
        activate_cmd = f'"{venv_path}" && python {script_name} {" ".join(script_args)}'
        shell_cmd = f'cmd /c {activate_cmd}'
    else:
        shell_cmd = f'python {script_name} {" ".join(script_args)}'

    print("▶️  Starting scraper...")
    print("-" * 30)

    try:
        # Run the scraper
        result = subprocess.run(
            shell_cmd,
            shell=True,
            cwd=os.getcwd(),
            capture_output=not args.verbose,
            text=True
        )

        print("-" * 30)
        print("🏁 Scraper completed!"        print(f"End Time: {datetime.now()}")

        if result.returncode == 0:
            print("✅ Success!")
        else:
            print(f"❌ Failed with exit code: {result.returncode}")

        if not args.verbose:
            print("\nOutput (last 20 lines):")
            lines = result.stdout.split('\n')[-20:]
            for line in lines:
                if line.strip():
                    print(f"  {line}")

            if result.stderr:
                print("\nErrors:")
                print(result.stderr)

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Error running scraper: {e}")
        sys.exit(1)

    print("\n📊 Check the following files for results:")
    print("  - successful_products*.json (scraped products)")
    print("  - failed_products*.json (failed products)")
    print("  - *.log (detailed logs)")
    print("  - checkpoint_*.json (progress checkpoints)")

if __name__ == "__main__":
    main()
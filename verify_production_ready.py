"""
Production Readiness Verification Script

This script verifies that the system is ready for production deployment with real data.
Run this before launching data collection and model training.

Usage:
    python verify_production_ready.py
"""

import os
import sys
from pathlib import Path
import importlib.util

def print_header(text):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_check(name, passed, details=""):
    """Print check result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} {name}")
    if details:
        print(f"     {details}")

def check_python_version():
    """Check Python version >= 3.10."""
    print_header("Python Version")
    version = sys.version_info
    passed = version >= (3, 10)
    print_check(
        "Python 3.10+",
        passed,
        f"Current: Python {version.major}.{version.minor}.{version.micro}"
    )
    return passed

def check_dependencies():
    """Check critical dependencies are installed."""
    print_header("Dependencies")

    required_packages = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("lightgbm", "ML models"),
        ("xgboost", "ML models"),
        ("sklearn", "scikit-learn ML framework"),
        ("mlflow", "Experiment tracking"),
        ("requests", "HTTP requests"),
        ("yaml", "pyyaml Configuration"),
    ]

    all_passed = True
    for package_name, description in required_packages:
        try:
            spec = importlib.util.find_spec(package_name)
            passed = spec is not None
            print_check(f"{package_name} ({description})", passed)
            if not passed:
                all_passed = False
        except (ImportError, ModuleNotFoundError):
            print_check(f"{package_name} ({description})", False)
            all_passed = False

    return all_passed

def check_env_file():
    """Check .env file exists and contains ENTSOE_API_KEY."""
    print_header("Environment Configuration")

    env_path = Path(".env")
    env_example_path = Path(".env.example")

    # Check .env.example exists
    example_exists = env_example_path.exists()
    print_check(".env.example exists", example_exists)

    # Check .env exists
    env_exists = env_path.exists()
    print_check(
        ".env file exists",
        env_exists,
        "Create from: cp .env.example .env" if not env_exists else ""
    )

    if not env_exists:
        return False

    # Check ENTSOE_API_KEY is configured
    with open(env_path) as f:
        content = f.read()

    has_key = "ENTSOE_API_KEY=" in content
    not_example = "your_entsoe_api_key_here" not in content

    api_key_configured = has_key and not_example
    print_check(
        "ENTSOE_API_KEY configured",
        api_key_configured,
        "Update ENTSOE_API_KEY in .env file" if not api_key_configured else ""
    )

    return env_exists and api_key_configured

def check_config_file():
    """Check config.yaml exists."""
    print_header("Configuration Files")

    config_path = Path("config.yaml")
    passed = config_path.exists()
    print_check("config.yaml exists", passed)

    return passed

def check_directories():
    """Check required directories exist."""
    print_header("Directory Structure")

    required_dirs = [
        "data/raw_data/market_prices",
        "data/raw_data/weather",
        "data/raw_data/energy",
        "data/raw_data/fundamentals",
        "data/modified_data",
        "data/cache/entsoe",
        "outputs/logs",
        "outputs/figures",
        "models",
    ]

    all_passed = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        passed = path.exists()
        if not passed:
            # Create directory
            try:
                path.mkdir(parents=True, exist_ok=True)
                passed = True
                print_check(f"{dir_path}", passed, "Created")
            except Exception as e:
                print_check(f"{dir_path}", False, f"Error: {e}")
                all_passed = False
        else:
            print_check(f"{dir_path}", passed)

    return all_passed

def check_critical_files():
    """Check critical Python files exist."""
    print_header("Critical Files")

    critical_files = [
        "data_collection/entsoe_connector.py",
        "data_collection/odre_collector.py",
        "data_collection/api_cache.py",
        "data_collection/data_validator.py",
        "model/price_forecasting/data_loader.py",
        "model/price_forecasting/models.py",
        "trading_system/backtesting/backtesting_engine.py",
        "test_api_connection.py",
    ]

    all_passed = True
    for file_path in critical_files:
        path = Path(file_path)
        passed = path.exists()
        print_check(file_path, passed)
        if not passed:
            all_passed = False

    return all_passed

def test_entsoe_connection():
    """Test ENTSO-E API connection."""
    print_header("API Connection Test")

    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("ENTSOE_API_KEY")

        if not api_key or api_key == "your_entsoe_api_key_here":
            print_check(
                "ENTSO-E API key",
                False,
                "API key not configured in .env"
            )
            return False

        print_check("ENTSO-E API key", True, "Key found in .env")

        # Try to import connector
        try:
            from data_collection.entsoe_connector import EntsoeConnector
            print_check("Import EntsoeConnector", True)
        except ImportError as e:
            print_check("Import EntsoeConnector", False, f"Import error: {e}")
            return False

        # Try to create connector
        try:
            connector = EntsoeConnector(api_key=api_key, use_cache=False)
            print_check("Create connector", True)
        except Exception as e:
            print_check("Create connector", False, f"Error: {e}")
            return False

        print("\nℹ️  Run 'python test_api_connection.py' for full API test")
        return True

    except Exception as e:
        print_check("API connection test", False, f"Error: {e}")
        return False

def check_data_leakage_prevention():
    """Verify data leakage fixes are in place."""
    print_header("Data Leakage Prevention")

    try:
        # Read data_loader.py
        loader_path = Path("model/price_forecasting/data_loader.py")
        if not loader_path.exists():
            print_check("data_loader.py exists", False)
            return False

        with open(loader_path) as f:
            content = f.read()

        # Check for explicit exclusion of contemporaneous variables
        has_load_exclusion = '"load_mw"' in content or "'load_mw'" in content
        has_fuel_exclusion = "exclude_cols" in content

        print_check(
            "load_mw exclusion",
            has_load_exclusion,
            "Contemporaneous load_mw excluded from features"
        )
        print_check(
            "Fuel price exclusion",
            has_fuel_exclusion,
            "Contemporaneous fuel prices excluded"
        )

        # Check for TimeSeriesSplit
        train_script = Path("model/price_forecasting/train_price_forecast.py")
        if train_script.exists():
            with open(train_script) as f:
                train_content = f.read()

            has_timeseries_split = "TimeSeriesSplit" in train_content
            print_check(
                "TimeSeriesSplit used",
                has_timeseries_split,
                "Correct cross-validation for time series"
            )

        return has_load_exclusion and has_fuel_exclusion

    except Exception as e:
        print_check("Data leakage check", False, f"Error: {e}")
        return False

def print_summary(checks):
    """Print summary of checks."""
    print_header("Production Readiness Summary")

    total = len(checks)
    passed = sum(checks.values())
    percentage = (passed / total) * 100 if total > 0 else 0

    print(f"\nTotal checks: {total}")
    print(f"Passed: {passed} ({percentage:.0f}%)")
    print(f"Failed: {total - passed}\n")

    if percentage == 100:
        print("🎉 SYSTEM IS PRODUCTION READY!")
        print("\nNext steps:")
        print("  1. Run: python test_api_connection.py")
        print("  2. Collect data: python data_recuperation/data_market_prices.py")
        print("  3. Train models: python model/price_forecasting/train_price_forecast.py")
        print("  4. Run backtest: python run_backtest_example.py")
    elif percentage >= 80:
        print("⚠️  SYSTEM IS MOSTLY READY (minor issues to fix)")
        print("\nReview failed checks above and fix them.")
    else:
        print("❌ SYSTEM NOT READY (critical issues)")
        print("\nPlease fix failed checks before proceeding.")

def main():
    """Run all production readiness checks."""
    print("\n" + "=" * 70)
    print("  PRODUCTION READINESS VERIFICATION")
    print("  Energy Demand Forecasting & Trading System")
    print("=" * 70)

    # Run checks
    checks = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Environment File": check_env_file(),
        "Config File": check_config_file(),
        "Directories": check_directories(),
        "Critical Files": check_critical_files(),
        "API Connection": test_entsoe_connection(),
        "Data Leakage Prevention": check_data_leakage_prevention(),
    }

    # Print summary
    print_summary(checks)

    # Exit code
    all_passed = all(checks.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()

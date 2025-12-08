"""
Complete Project Audit for GitHub Portfolio
Checks: Structure, Documentation, Code Quality, Tests, Results
"""

import os
from pathlib import Path

class ProjectAuditor:
    def __init__(self, project_root: Path):
        self.root = project_root
        self.issues = []
        self.warnings = []
        self.successes = []

    def audit(self):
        print("="*80)
        print("PROJECT AUDIT FOR GITHUB PORTFOLIO")
        print("="*80)
        print(f"Project: {self.root.name}")
        print()

        self.check_structure()
        self.check_documentation()
        self.check_code_quality()
        self.check_results()
        self.check_git()
        self.check_security()

        self.print_summary()

    def check_structure(self):
        print("\n[1/6] PROJECT STRUCTURE")
        print("-" * 80)

        required_dirs = [
            "data", "data_collection", "model", "production",
            "research", "tests"
        ]

        for dir_name in required_dirs:
            dir_path = self.root / dir_name
            if dir_path.exists():
                self.successes.append(f"Directory {dir_name}/ exists")
            else:
                self.issues.append(f"Missing {dir_name}/ directory")

        print(f"  Required directories: {len([s for s in self.successes if 'exists' in s])}/{len(required_dirs)}")

    def check_documentation(self):
        print("\n[2/6] DOCUMENTATION")
        print("-" * 80)

        # README
        readme = self.root / "README.md"
        if readme.exists():
            content = readme.read_text(encoding='utf-8')
            if len(content) > 5000:
                self.successes.append("README.md is comprehensive (>5000 chars)")
            else:
                self.warnings.append("README.md seems short (<5000 chars)")

            required_sections = [
                "Abstract", "Installation", "Usage", "Results",
                "Limitations", "References"
            ]
            missing = []
            for section in required_sections:
                if section.lower() in content.lower():
                    self.successes.append(f"README has '{section}' section")
                else:
                    missing.append(section)

            if missing:
                self.warnings.append(f"README missing sections: {', '.join(missing)}")
        else:
            self.issues.append("README.md missing!")

        # Requirements
        req = self.root / "requirements.txt"
        if req.exists():
            self.successes.append("requirements.txt exists")
        else:
            self.issues.append("requirements.txt missing")

        # .gitignore
        gitignore = self.root / ".gitignore"
        if gitignore.exists():
            self.successes.append(".gitignore exists")
        else:
            self.warnings.append(".gitignore missing")

    def check_code_quality(self):
        print("\n[3/6] CODE QUALITY")
        print("-" * 80)

        # Check for main scripts
        production_pipeline = self.root / "production/trading_pipeline.py"
        if production_pipeline.exists():
            content = production_pipeline.read_text(encoding='utf-8')

            # Check for docstrings
            if '"""' in content:
                self.successes.append("Production pipeline has docstrings")
            else:
                self.warnings.append("Production pipeline lacks docstrings")

            # Check for type hints
            if "def " in content and "->" in content:
                self.successes.append("Uses type hints")
            else:
                self.warnings.append("Missing type hints in some functions")

            # Check for error handling
            if "try:" in content or "except" in content:
                self.successes.append("Has error handling")
            else:
                self.warnings.append("Limited error handling")
        else:
            self.issues.append("Main pipeline script missing")

    def check_results(self):
        print("\n[4/6] RESULTS & OUTPUTS")
        print("-" * 80)

        reports_dir = self.root / "research/reports"
        if reports_dir.exists():
            # Check for key result files
            key_files = [
                "trading_pipeline_latest.csv",
                "ridge_predictions.csv",
                "xgboost_predictions.csv"
            ]

            found = 0
            for fname in key_files:
                if (reports_dir / fname).exists():
                    found += 1

            if found >= 2:
                self.successes.append(f"Found {found}/{len(key_files)} key result files")
            else:
                self.warnings.append(f"Only {found}/{len(key_files)} result files present")

            # Check for visualizations
            png_files = list(reports_dir.glob("*.png"))
            if len(png_files) >= 2:
                self.successes.append(f"Has {len(png_files)} visualization(s)")
            else:
                self.warnings.append("Few visualizations")
        else:
            self.warnings.append("reports/ directory missing")

    def check_git(self):
        print("\n[5/6] GIT REPOSITORY")
        print("-" * 80)

        git_dir = self.root / ".git"
        if git_dir.exists():
            self.successes.append("Git repository initialized")

            # Check for .env in gitignore
            gitignore = self.root / ".gitignore"
            if gitignore.exists():
                content = gitignore.read_text()
                if ".env" in content:
                    self.successes.append(".env is gitignored")
                else:
                    self.warnings.append(".env not in .gitignore")
        else:
            self.issues.append("Not a git repository")

    def check_security(self):
        print("\n[6/6] SECURITY & SECRETS")
        print("-" * 80)

        # Check for .env.example
        env_example = self.root / ".env.example"
        if env_example.exists():
            self.successes.append(".env.example provided")
        else:
            self.warnings.append("No .env.example for template")

        # Check if .env is tracked
        env_file = self.root / ".env"
        if env_file.exists():
            # Should be in gitignore
            gitignore = self.root / ".gitignore"
            if gitignore.exists() and ".env" in gitignore.read_text():
                self.successes.append(".env file exists and is gitignored")
            else:
                self.issues.append(".env file exists but NOT gitignored!")

    def print_summary(self):
        print("\n" + "="*80)
        print("AUDIT SUMMARY")
        print("="*80)

        total = len(self.successes) + len(self.warnings) + len(self.issues)
        score = (len(self.successes) / total * 100) if total > 0 else 0

        print(f"\nOverall Score: {score:.1f}%")
        print(f"Successes: {len(self.successes)}")
        print(f"Warnings:  {len(self.warnings)}")
        print(f"Issues:    {len(self.issues)}")

        if self.issues:
            print("\nCRITICAL ISSUES (Fix before publishing):")
            for issue in self.issues:
                print(f"  - {issue}")

        if self.warnings:
            print("\nWARNINGS (Recommended to fix):")
            for warning in self.warnings[:10]:
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings)-10} more")

        print("\nTOP SUCCESSES:")
        for success in self.successes[:10]:
            print(f"  + {success}")
        if len(self.successes) > 10:
            print(f"  ... and {len(self.successes)-10} more")

        print("\n" + "="*80)
        if score >= 90:
            print("EXCELLENT - Ready to publish!")
        elif score >= 75:
            print("GOOD - Minor fixes recommended")
        elif score >= 60:
            print("FAIR - Several improvements needed")
        else:
            print("NEEDS WORK - Major issues to address")
        print("="*80)

if __name__ == "__main__":
    project_root = Path(".")
    auditor = ProjectAuditor(project_root)
    auditor.audit()

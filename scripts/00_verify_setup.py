"""
Verify Phase 1 setup - Configuration and Utilities
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import validate_config, PDF_SOURCES, EMBEDDING_CONFIG, LLM_CONFIG
from src.utils.logger import setup_logger
from src.utils.helpers import Timer, format_time, format_number


def verify_phase1():
    """Verify all Phase 1 components"""

    print("=" * 60)
    print("Phase 1 Verification: Foundation")
    print("=" * 60)

    all_passed = True

    # 1. Check directory structure
    print("\n1. Directory Structure:")
    required_dirs = [
        "data/pdfs",
        "data/processed/chunks",
        "data/processed/embeddings",
        "data/processed/indices",
        "data/cache",
        "src/loaders",
        "src/processors",
        "src/embeddings",
        "src/vector_store",
        "src/retrieval",
        "src/reranking",
        "src/generation",
        "src/pipeline",
        "src/utils",
        "scripts",
        "tests"
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"  [OK] {dir_path}")
        else:
            print(f"  [FAIL] {dir_path} - MISSING")
            all_passed = False

    # 2. Check configuration files
    print("\n2. Configuration Files:")
    config_files = [
        "requirements.txt",
        ".env",
        ".env.template",
        ".gitignore",
        "src/config.py"
    ]

    for file_path in config_files:
        path = Path(file_path)
        if path.exists() and path.is_file():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [FAIL] {file_path} - MISSING")
            all_passed = False

    # 3. Check utility modules
    print("\n3. Utility Modules:")
    try:
        import src.utils.logger
        print("  [OK] src.utils.logger")
    except ImportError as e:
        print(f"  [FAIL] src.utils.logger - {e}")
        all_passed = False

    try:
        import src.utils.helpers
        print("  [OK] src.utils.helpers")
    except ImportError as e:
        print(f"  [FAIL] src.utils.helpers - {e}")
        all_passed = False

    # 4. Test logger
    print("\n4. Logger Test:")
    try:
        logger = setup_logger("verification", level="INFO")
        logger.info("Logger test message")
        print("  [OK] Logger working")
    except Exception as e:
        print(f"  [FAIL] Logger error: {e}")
        all_passed = False

    # 5. Test helper utilities
    print("\n5. Helper Utilities Test:")
    try:
        with Timer("Test timer"):
            result = format_time(3665)
            assert result == "1.02h"
            result = format_number(1000000)
            assert result == "1,000,000"
        print("  [OK] Helper utilities working")
    except Exception as e:
        print(f"  [FAIL] Helper utilities error: {e}")
        all_passed = False

    # 6. Configuration validation
    print("\n6. Configuration:")
    config_errors = validate_config()

    if config_errors:
        print("  [WARN] Configuration warnings:")
        for error in config_errors:
            print(f"     - {error}")
        print("  Note: These are expected if data files not yet downloaded")
    else:
        print("  [OK] Configuration valid")

    # 7. Display configuration summary
    print("\n7. Configuration Summary:")
    print(f"  Data sources: {len(PDF_SOURCES)}")
    for name, config in PDF_SOURCES.items():
        print(f"    - {name}: {config['type']} ({config.get('priority', 'N/A')})")

    print(f"\n  Embedding model: {EMBEDDING_CONFIG['model_name']}")
    print(f"  LLM model: {LLM_CONFIG['model']}")

    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] Phase 1 Complete: Foundation Ready")
        print("=" * 60)
        return 0
    else:
        print("[FAIL] Phase 1 Incomplete: Some components missing")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = verify_phase1()
    sys.exit(exit_code)

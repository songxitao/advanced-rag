import sys

# Ensure stdout is in UTF-8 to handle any printing issues safely, or stick to ASCII.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

required_packages = ['ragas', 'matplotlib', 'pandas', 'docx', 'requests', 'openai', 'jieba']

failed = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"SUCCESS: Package '{pkg}' is importable.")
    except ImportError as e:
        print(f"FAILURE: Package '{pkg}' could not be imported. Error: {e}")
        failed.append(pkg)

if failed:
    print(f"\nEnvironment check failed. Missing packages: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\nEnvironment check passed. All packages are importable.")
    sys.exit(0)

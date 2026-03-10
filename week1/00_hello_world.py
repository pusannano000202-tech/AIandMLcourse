import sys
import importlib.util

def _supports_symbol(symbol):
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    try:
        symbol.encode(encoding)
        return True
    except UnicodeEncodeError:
        return False

PASS_ICON = "✅" if _supports_symbol("✅") else "[OK]"
FAIL_ICON = "❌" if _supports_symbol("❌") else "[NO]"

def check_package(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"{FAIL_ICON} {name}: Not installed")
    else:
        try:
            module = __import__(name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"{PASS_ICON} {name}: Installed (Version {version})")
        except ImportError:
            print(f"{FAIL_ICON} {name}: Installed but cannot be imported")

print("="*40)
print("Hello World! Environment Check")
print("="*40)
print(f"Python Version: {sys.version.split()[0]}")
print("-" * 40)

packages = ['numpy', 'matplotlib', 'tensorflow', 'reportlab']

for package in packages:
    check_package(package)

print("="*40)
print("If you see all checkmarks, your environment is ready!")

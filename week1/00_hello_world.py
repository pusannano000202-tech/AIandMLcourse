import sys
import importlib.util

def check_package(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"❌ {name}: Not installed")
    else:
        try:
            module = __import__(name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {name}: Installed (Version {version})")
        except ImportError:
            print(f"❌ {name}: Installed but cannot be imported")

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

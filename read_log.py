
import os
import sys

# Reconfigure stdout to utf-8 just in case
sys.stdout.reconfigure(encoding='utf-8')

log_file = 'training_log.txt'
if not os.path.exists(log_file):
    print(f"{log_file} not found.")
else:
    print(f"Size: {os.path.getsize(log_file)} bytes")
    print("-" * 20)
    try:
        # Try UTF-16 (standard for PowerShell redirection)
        print(open(log_file, 'r', encoding='utf-16').read())
    except Exception as e1:
        try:
            # Fallback to UTF-8
            print(open(log_file, 'r', encoding='utf-8', errors='replace').read())
        except Exception as e2:
            print(f"Failed to read: {e1}, {e2}")

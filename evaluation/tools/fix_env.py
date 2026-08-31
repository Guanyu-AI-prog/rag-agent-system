#!/usr/bin/env python3
import os

# Read the key from environment or use the provided key
key = os.environ.get('TEMP_API_KEY', '')
if not key:
    print("No key provided")
    exit(1)

print(f"Key length: {len(key)}")

with open('.env', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith('SILICONFLOW_API_KEY=*** and not line.startswith('#'):
        new_lines.append(f'SILICONFLOW_API_KEY=***    else:
        new_lines.append(line)

with open('.env', 'w') as f:
    f.writelines(new_lines)
print('done')

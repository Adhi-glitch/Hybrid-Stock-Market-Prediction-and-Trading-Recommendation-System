import re

# Read the file
with open('test_installation.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode characters
replacements = {
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[WARNING]',
    '🎉': '[SUCCESS]',
    'ℹ️': '[INFO]',
    '💡': '[TIP]',
    '📚': '[DOCS]',
    '📋': '[REFERENCE]',
    '📖': '[EXAMPLES]',
    '🧪': '[TEST]'
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Write back
with open('test_installation.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed all Unicode in test_installation.py')

import re

# Read the file
with open('run_full_analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode characters
replacements = {
    '📊': '[DATA]',
    '🚀': '[LAUNCH]',
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[WARNING]',
    '📁': '[FILES]',
    '📄': '[FILE]',
    '📋': '[REPORT]',
    '💡': '[TIP]',
    '📌': '[NOTE]',
    '🎯': '[TARGET]',
    '🔮': '[PREDICTION]',
    '📰': '[NEWS]',
    '🤖': '[AI]',
    '🎉': '[SUCCESS]'
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Write back
with open('run_full_analysis.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed run_full_analysis.py')

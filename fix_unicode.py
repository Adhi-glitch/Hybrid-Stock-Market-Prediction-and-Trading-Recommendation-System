"""
Quick fix for Windows Unicode issues
Replaces emoji characters with Windows-compatible text
"""

import re

def fix_unicode_issues():
    """Replace emoji characters with Windows-compatible alternatives"""
    
    # Define emoji replacements
    replacements = {
        '🤖': '[AI]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '📰': '[NEWS]',
        '📊': '[DATA]',
        '🔍': '[ANALYZING]',
        '💾': '[SAVED]',
        '📄': '[FILE]',
        '📋': '[REPORT]',
        '🎯': '[TARGET]',
        '💡': '[INSIGHT]',
        '🟢': '[POSITIVE]',
        '🔴': '[NEGATIVE]',
        'ℹ️': '[INFO]',
        '🧹': '[CLEANUP]',
        '📁': '[FILES]',
        '🚀': '[STRONG BUY]',
        '📈': '[BUY]',
        '🔻': '[STRONG SELL]',
        '📉': '[SELL]',
        '➡️': '[HOLD]'
    }
    
    # Files to fix
    files_to_fix = ['simp.py', 'reason.py']
    
    for filename in files_to_fix:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace emojis
            for emoji, replacement in replacements.items():
                content = content.replace(emoji, replacement)
            
            # Write back
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Fixed Unicode issues in {filename}")
            
        except Exception as e:
            print(f"Error fixing {filename}: {e}")
    
    print("Unicode fix complete!")

if __name__ == "__main__":
    fix_unicode_issues()

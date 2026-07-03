import os
import re
import sys

def fix_item_spaces(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # \item 바로 뒤에 공백이나 백슬래시(\)가 아닌 다른 글자가 오면 중간에 공백 삽입
    fixed_content = re.sub(r'\\item([^\s\\])', r'\\item \1', content)
    
    if content != fixed_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"Fixed item spacing in {file_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_item_spaces.py <dir>")
        sys.exit(1)
    target_dir = sys.argv[1]
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.tex'):
                fix_item_spaces(os.path.join(root, file))

if __name__ == '__main__':
    main()

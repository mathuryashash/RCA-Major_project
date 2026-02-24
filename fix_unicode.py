import os

def fix_unicode(directory, old_char='→', new_char='->'):
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    if old_char in content:
                        print(f"Fixing {path}")
                        content = content.replace(old_char, new_char)
                        with open(path, 'w', encoding='utf-8') as file:
                            file.write(content)
                except Exception as e:
                    print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    fix_unicode(r'd:\vscode\majorprojectt\src')
    fix_unicode(r'd:\vscode\majorprojectt\rca-system\src')

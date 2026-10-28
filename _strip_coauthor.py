import sys

text = sys.stdin.read()
lines = [
    ln for ln in text.splitlines()
    if "Co-authored-by: Cursor" not in ln
    and "cursoragent@cursor.com" not in ln
]
sys.stdout.write("\n".join(lines).rstrip() + "\n")

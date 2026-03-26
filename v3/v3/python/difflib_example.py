import difflib

text1 = "The quick brown fox jumps over the lazy dog."
text2 = "The quick brown fox leaps over the lazy dog."

diff = difflib.unified_diff(
    text1.splitlines(), text2.splitlines(),
    fromfile='text1', tofile='text2', lineterm=''
)
print('\n'.join(diff))

# 相似度
similar = difflib.SequenceMatcher(None, text1, text2).ratio()
print(f"Similarity: {similar:.2%}")

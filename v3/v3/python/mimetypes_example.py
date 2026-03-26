import mimetypes

files = ['index.html', 'image.jpg', 'script.py', 'data.json', 'archive.tar.gz']
for f in files:
    mime, encoding = mimetypes.guess_type(f)
    print(f"{f}: {mime} (encoding: {encoding})")

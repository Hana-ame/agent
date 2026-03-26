from html.parser import HTMLParser
import html.entities

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start tag: {tag}, attrs: {attrs}")
    def handle_endtag(self, tag):
        print(f"End tag: {tag}")
    def handle_data(self, data):
        print(f"Data: {data.strip()}")
    def handle_entityref(self, name):
        print(f"Entity: &{name};")
    def handle_charref(self, name):
        print(f"Char ref: &#{name};")

parser = MyHTMLParser()
parser.feed('<html><head><title>Test &amp; <>&#65;</title></head><body><p>Hello World!</p></body></html>')

# 转义和反转义
import html
escaped = html.escape('<div>Hello & World</div>')
print("Escaped:", escaped)
unescaped = html.unescape(escaped)
print("Unescaped:", unescaped)

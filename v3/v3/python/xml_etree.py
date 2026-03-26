import xml.etree.ElementTree as ET

xml_data = """<?xml version="1.0"?>
<data>
    <person id="1">
        <name>Alice</name>
        <age>30</age>
    </person>
    <person id="2">
        <name>Bob</name>
        <age>25</age>
    </person>
</data>"""

root = ET.fromstring(xml_data)
print("Root tag:", root.tag)

for person in root.findall('person'):
    pid = person.get('id')
    name = person.find('name').text
    age = person.find('age').text
    print(f"Person {pid}: {name}, {age}")

# 创建 XML
new_root = ET.Element("catalog")
product = ET.SubElement(new_root, "product", id="p1")
name = ET.SubElement(product, "name")
name.text = "Laptop"
price = ET.SubElement(product, "price")
price.text = "999.99"

tree = ET.ElementTree(new_root)
tree.write("temp.xml")
import os
os.remove("temp.xml")
print("XML created and removed")

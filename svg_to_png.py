#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import re

def simple_svg_to_png(svg_file, png_file):
    # This is a very basic SVG to PNG converter
    # It only handles basic shapes and won't work for complex SVGs
    
    # Parse SVG
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    # Get SVG dimensions
    width = int(root.get('width', 400))
    height = int(root.get('height', 300))
    
    # Create PIL image
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Process basic shapes
    for elem in root.iter():
        if elem.tag.endswith('circle'):
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            r = float(elem.get('r', 0))
            fill = elem.get('fill', 'black')
            stroke = elem.get('stroke', 'none')
            
            if fill != 'none':
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill)
            if stroke != 'none':
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=stroke)
                
        elif elem.tag.endswith('line'):
            x1 = float(elem.get('x1', 0))
            y1 = float(elem.get('y1', 0))
            x2 = float(elem.get('x2', 0))
            y2 = float(elem.get('y2', 0))
            stroke = elem.get('stroke', 'black')
            width = int(elem.get('stroke-width', 1))
            
            draw.line([x1, y1, x2, y2], fill=stroke, width=width)
            
        elif elem.tag.endswith('ellipse'):
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            rx = float(elem.get('rx', 0))
            ry = float(elem.get('ry', 0))
            fill = elem.get('fill', 'black')
            stroke = elem.get('stroke', 'none')
            
            if fill != 'none':
                draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=fill)
            if stroke != 'none':
                draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], outline=stroke)
    
    # Save as PNG
    img.save(png_file)
    print(f"Basic conversion completed: {png_file}")

if __name__ == "__main__":
    simple_svg_to_png('pelican-bicycle.svg', 'pelican-bicycle.png')
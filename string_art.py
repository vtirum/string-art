import cv2
import numpy as np
from tqdm import tqdm
import requests 
from collections import Counter
from xml.etree.ElementTree import Element, SubElement, ElementTree

def generate_circle_pins(num_pins, radius):
    return [
        (
            int(radius + radius * np.cos(2 * np.pi * i / num_pins)),
            int(radius + radius * np.sin(2 * np.pi * i / num_pins))
        )
        for i in range(num_pins)
    ]

def get_line_pixels(pin1, pin2):
    # returns a list of pixels for the line between pin1 and pin2
    key = tuple(sorted([pin1, pin2]))
    if key not in line_cache:
        line_cache[key] = bresenham_line(pin1[0], pin1[1], pin2[0], pin2[1])
    return line_cache[key]

def bresenham_line(x0, y0, x1, y1):
    pixels = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        pixels.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx: 
            err += dx
            y0 += sy
    return pixels

def get_next_pin(current_pin):
    best_pin = None
    best_color = None
    best_pixels = None
    max_err = -1

    for i in range(1, num_pins):
        new_pin = (current_pin + i) % num_pins
        if new_pin in recent_connections:
            continue

        pixels = get_line_pixels(pins[current_pin], pins[new_pin])
        if not pixels:
            continue

        for color in (1.0, 0.0):
            total_err = 0.0
            for x, y in pixels:
                if 0 <= x < width and 0 <= y < height:
                    idx = y * width + x
                    #dark_weight = 1.0 + (y_flat[idx] * 2.0) 
                    if color == 1.0:
                        total_err += err_w[idx]
                    else:
                        total_err += err_b[idx]

            if total_err > max_err:
                max_err = total_err
                best_pin = new_pin
                best_color = color
                best_pixels = pixels
    
    if best_pin is None:
        return None, None

    for x, y in best_pixels:
        if 0 <= x < width and 0 <= y < height:
            idx = y * width + x
            #dark_weight = 1.0 + (y_flat[idx] * 1.5)
            if best_color == 1.0:
                err_w[idx] = max(0.0, err_w[idx] - line_weight)
            else:
                err_b[idx] = max(0.0, err_b[idx] - line_weight)

    return best_pin, best_color

def get_next_pin_alt(current_pin):
    best_pin = None
    best_color = None
    max_err = -1

    valid_pins = np.ones(num_pins, dtype=bool)
    valid_pins[list(recent_connections)] = False
    valid_pins[current_pin] = False

    possible_pins = np.where(valid_pins)[0]

    for new_pin in possible_pins:
        pixels = get_line_pixels(pins[current_pin], pins[new_pin])
        if not pixels: 
            continue

        xs, ys = zip(*pixels)
        indices = np.array(ys) * width + np.array(xs)

        mask = (0 <= np.array(xs)) & (np.array(xs) < width) & (0 <= np.array(ys)) & (np.array(ys) < height)
        valid_indices = indices[mask]

        if len(valid_indices) == 0:
            continue

        total_err_w = np.sum(err_w[valid_indices])
        total_err_b = np.sum(err_b[valid_indices])

        if total_err_w > max_err:
            max_err = total_err_w
            best_pin = new_pin
            best_color = 1.0
        if total_err_b > max_err:
            max_err = total_err_b
            best_pin = new_pin
            best_color = 0.0

    if best_pin is None:
        return None, None
    
    pixels = get_line_pixels(pins[current_pin], pins[best_pin])
    xs, ys = zip(*pixels)
    indices = np.array(ys) * width + np.array(xs)
    mask = (0 <= np.array(xs)) & (np.array(xs) < width) & (0 <= np.array(ys)) & (np.array(ys) < height)
    valid_indices = indices[mask]

    if best_color == 1.0:
        err_w[valid_indices] = np.maximum(0.0, err_w[valid_indices] - line_weight)
    else:
        err_b[valid_indices] = np.maximum(0.0, err_b[valid_indices] - line_weight)

    return best_pin, best_color

def export_svg(filename, pins, pin_sequence, width, height):
    xs = [p[0] for p in pins]
    ys = [p[1] for p in pins]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pin_w = max_x - min_x
    pin_h = max_y - min_y
    margin = max(pin_w, pin_h) * 0.05

    scale_x = (width - 2 * margin) / pin_w
    scale_y = (height - 2 * margin) / pin_h
    scale = min(scale_x, scale_y)

    def transform(x, y):
        tx = (x - min_x) * scale + margin
        ty = (y - min_y) * scale + margin
        return int(tx), int(ty) 
    
    svg_root = Element('svg', xmlns="http://www.w3.org/2000/svg",
                       width=str(width), height=str(height),
                       viewBox=f"0 0 {width} {height}")
    
    SubElement(svg_root, 'rect', {
        'x': '0', 
        'y': '0', 
        'width': str(width), 
        'height': str(height), 
        'fill': 'gray'
    })

    for i in range(1, len(pin_sequence)):
        pin1_idx, _ = pin_sequence[i - 1]
        pin2_idx, color = pin_sequence[i]
        x1, y1 = transform(*pins[pin1_idx])
        x2, y2 = transform(*pins[pin2_idx])

        stroke_color = 'white' if color >= 0.5 else 'black'
        SubElement(svg_root, 'line', {
            'x1': str(x1), 
            'y1': str(y1), 
            'x2': str(x2), 
            'y2': str(y2), 
            'stroke': stroke_color, 
            'stroke-width': '0.5'
        })

    ElementTree(svg_root).write(filename)


url = None
filename = 'cookie/cookie.png'
if url is not None and filename is not None:
    resp = requests.get(url)
    img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
else:
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


y = 1.0 - img.astype(np.float32) / 255.0
height, width = y.shape
y = (y - y.min()) / (y.max() - y.min()) 
y_flat = y.flatten()

err_w = np.array((1.0 - y_flat).copy())
err_b = np.array(y_flat.copy())

radius = img.shape[0] // 2

num_pins = 300 
num_lines = 5000
start_pin = 0
line_weight = 0.5 / ((num_lines * 200) / (height * width)) 

pins = generate_circle_pins(num_pins, radius)
pin_sequence = [(start_pin, 0.0)]
line_cache = {}
recent_connections = []

for t in tqdm(range(num_lines)):
    current_pin = pin_sequence[-1][0]
    next_pin, color = get_next_pin(current_pin)
    
    if next_pin is None:
        break

    recent_connections.append(current_pin)
    if len(recent_connections) > 10:
        recent_connections.pop(0)
    pin_sequence.append((next_pin, color))

export_svg('cookie/cookie_1.svg', pins, pin_sequence, width=1500, height=1500)

with open('cookie/cookie_pin_seq.txt', 'w') as f:
    for pin, color in pin_sequence:
        f.write(f"{pin} {color}\n")
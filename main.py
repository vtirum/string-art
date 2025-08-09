import cv2
import numpy as np
import itertools
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

def get_pin_connections(current_pin):
    # returns a list of tuples (current_nail, other_nail_index)
    return [(current_pin, i) for i in range(num_pins) if i != current_pin]

def get_line_pixels(pin1, pin2):
    # returns a list of pixels for the line between pin1 and pin2
    key = tuple(sorted([pin1, pin2]))
    if key not in line_cache:
        line_cache[key] = bresenham_line(pin1[0], pin1[1], pin2[0], pin2[1])
    return line_cache[key]

def render_line(shape, pin1, pin2, color):
    canvas = np.zeros(shape, dtype=np.float32)
    cv2.line(canvas, pin1, pin2, color=1.0, thickness=1, lineType=cv2.LINE_AA)
    if color == 0.0:
        canvas *= -1.0
    return canvas.flatten()

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

def get_line_diff(pixels, color, canvas, target, fade = 0.8):
    """
    pixels: list of (x, y) tuples for the line
    color: float (e.g., 1.0 for white, 0.0 for black)
    canvas: flattened current image (float32)
    target: flattened target image (float32)
    width: width of the image (used for flattening)
    fade: blending factor (between 0 and 1)
    """
    total_diff = 0
    
    for x, y in pixels:
        if 0 <= x < width and 0 <= y < height:
            ind = y * width + x
            current_val = canvas[ind]
            target_val = target[ind]
            new_val = color * fade + current_val * (1 - fade)

            current_diff = abs(target_val - current_val)
            new_diff = abs(target_val - new_val)

            pixeldiff = current_diff - new_diff
            total_diff += pixeldiff if pixeldiff < 0 else pixeldiff / 5  
    return (total_diff / len(pixels)) ** 3

'''
def get_line_diff_new(color, pixels, fade=0.15):
    total_diff = 0

    for i in range(len(pixels)):
        p = pixels[i]
        ind = p[1] * width + p[0] 
        pixel_diff = 0
        new_c = color * fade + canvas_flat[ind] * (1 - fade)
        diff = abs(y_flat[ind] - new_c) - abs(canvas_flat[ind] - y_flat[ind])
        pixel_diff += diff
        if pixel_diff < 0:
            total_diff += pixel_diff
        if pixel_diff > 0: 
            total_diff += pixel_diff / 5

    return (total_diff / len(pixels)) ** 3
'''

def get_next_pin(current_pin, previous_pin, canvas_flat): 
    connections = get_pin_connections(current_pin)
    best_diff = float('inf')
    best_pin = None
    best_color = None

    for _, pin in connections:
        if pin == previous_pin:
            continue
        if (current_pin, pin) in recent_connections or (pin, current_pin) in recent_connections:
            continue

        pixels = get_line_pixels(pins[current_pin], pins[pin])

        for color in (1.0, 0.0):
            diff = get_line_diff(pixels, color, canvas_flat, y_flat)
            if diff < best_diff:
                best_diff = diff
                best_pin = pin
                best_color = color
            
    return best_pin, best_color
    
def export_svg(filename, pins, pin_sequence, width, height, bg_gray=0.5):
    gray_hex = "{:02x}{:02x}{:02x}".format(
        int(bg_gray * 255), 
        int(bg_gray * 255), 
        int(bg_gray * 255)
    )

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
        return int(tx), int(ty) #try with no int
    
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

url = "https://github.com/usedhondacivic/string-art-gen/blob/main/example.png?raw=true"
resp = requests.get(url)
img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

# Uncomment the next line if you want to read from a local file instead
# img = cv2.imread('example.png', cv2.IMREAD_GRAYSCALE)


y = 1.0 - img.astype(np.float32) / 255.0
height, width = y.shape
y = (y - y.min()) / (y.max() - y.min()) 
y_flat = y.flatten()
m = len(y_flat)
radius = img.shape[0] // 2

num_pins = 200 
num_lines = 3000
start_pin = 0
pins = []
pin_sequence = [(start_pin, 0.0)]
line_cache = {}
recent_connections = set()

pins = generate_circle_pins(num_pins, radius)
canvas = np.full_like(img, 0.5, dtype=np.float32)
canvas_flat = canvas.flatten()

'''
edges = list(itertools.combinations(range(num_pins), 2))
used_edges = set()
selected_edges = []
'''

for t in tqdm(range(num_lines)):
    current_pin = pin_sequence[-1][0]
    previous_pin = pin_sequence[-2][0] if len(pin_sequence) > 1 else None
    next_pin, color = get_next_pin(current_pin, previous_pin, canvas_flat)
    recent_connections.add((current_pin, next_pin))
    if len(recent_connections) > 20:
        recent_connections.pop()

    pixels = get_line_pixels(pins[current_pin], pins[next_pin])
    for x, y in pixels:
        if 0 <= x < width and 0 <= y < height:
            canvas[y, x] = color * 0.2 + canvas[y, x] * 0.8

    canvas_flat = canvas.flatten()
    pin_sequence.append((next_pin, color))

cv2.imshow('canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

export_svg('final_image_vector.svg', pins, pin_sequence, 
           width=800, height=800, bg_gray=0.5)

'''
out_img = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
cv2.imwrite('final_image_iterative.png', out_img)

svg_root = Element('svg', xmlns="http://www.w3.org/2000/svg", 
                   width=str(width), height=str(height))

for i in range(1, len(pin_sequence)):
    pin1_idx, _ = pin_sequence[i - 1]
    pin2_idx, color = pin_sequence[i]
    x1, y1 = pins[pin1_idx]
    x2, y2 = pins[pin2_idx]

    stroke_color = 'white' if color >= 0.5 else 'black'
    SubElement(svg_root, 'line', {
        'x1': str(x1), 
        'y1': str(y1), 
        'x2': str(x2), 
        'y2': str(y2), 
        'stroke': stroke_color, 
        'stroke-width': '0.5'
    })

tree = ElementTree(svg_root)
tree.write('final_image_vector.svg')
'''
'''
final_canvas = np.full_like(canvas, 0.5, dtype=np.float32)
color_counts = Counter(round(color, 1) for _, color in pin_sequence)
print(color_counts)

for i in range(1, len(pin_sequence)):
    pin1, _ = pin_sequence[i - 1]
    pin2, color = pin_sequence[i]
    pin1 = pins[pin1]
    pin2 = pins[pin2]
    line = render_line(canvas.shape, pin1, pin2, color)
    final_canvas += line.reshape(canvas.shape)
    
final_canvas = np.clip(final_canvas * 255.0, 0, 255).astype(np.uint8)

cv2.imwrite('final_image.png', final_canvas)
with open('pin_sequence.txt', 'w') as f:
    for pin, color in pin_sequence:
        f.write(f"{pin} {color}\n")
        '''
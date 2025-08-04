import cv2
import numpy as np
import itertools
from tqdm import tqdm
import requests 


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

def render_line(args):
    shape, pin1, pin2, color = args
    canvas = np.zeros(shape, dtype=np.float32)
    cv2.line(canvas, pin1, pin2, color=color, thickness=1)
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
            new_val = color * fade + current_val * (1- fade)

            current_diff = abs(target_val - current_val)
            new_diff = abs(target_val - new_val)

            pixeldiff = current_diff - new_diff
            total_diff += pixeldiff if pixeldiff < 0 else pixeldiff / 5
    return (total_diff / len(pixels)) ** 3

def get_next_pin(current_pin, previous_pin, canvas_flat): 
    connections = get_pin_connections(current_pin)
    best_diff = float('inf')
    best_pin = None

    for _, pin in connections:
        if pin == previous_pin:
            continue
        if (current_pin, pin) in recent_connections or (pin, current_pin) in recent_connections:
            continue
        pixels = get_line_pixels(pins[current_pin], pins[pin])
        w_diff = get_line_diff(pixels, 1.0, canvas_flat, y_flat)
        b_diff = get_line_diff(pixels, 0.0, canvas_flat, y_flat)
        
        if w_diff < best_diff:
            best_diff = w_diff
            best_pin = pin
            color = 1.0
        if b_diff < best_diff:
            best_diff = b_diff
            best_pin = pin
            color = 0.0
    
    #best_pin_ind = best_pin[1]
    
    return best_pin, color
    

url = "https://github.com/usedhondacivic/string-art-gen/blob/main/example.png?raw=true"
resp = requests.get(url)
img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
# Uncomment the next line if you want to read from a local file instead
#img = cv2.imread('cropped_image.png', cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (300, 300))
'''
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

y = 1.0 - img.astype(np.float32) / 255.0
height, width = y.shape
y_flat = y.flatten()
m = len(y_flat)
radius = img.shape[0] // 2

num_pins = 200 
num_lines = 5000
start_pin = 0
pins = []
pin_sequence = [(start_pin, 0.0)]
line_cache = {}
recent_connections = set()

pins = generate_circle_pins(num_pins, radius)
canvas = np.full_like(img, 0.5, dtype=np.float32)
canvas_flat = canvas.flatten()

edges = list(itertools.combinations(range(num_pins), 2))
used_edges = set()
selected_edges = []

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
            canvas[y, x] = color * 0.8 + canvas[y, x] * 0.2

    canvas_flat = canvas.flatten()
    pin_sequence.append((next_pin, color))

final_canvas = np.full_like(canvas, 0.5, dtype=np.float32)

for i in range(1, len(pin_sequence)):
    pin1, _ = pin_sequence[i - 1]
    pin2, color = pin_sequence[i]
    pin1 = pins[pin1]
    pin2 = pins[pin2]
    line = render_line((canvas.shape, pin1, pin2, color))
    final_canvas += line.reshape(canvas.shape)
    
final_canvas = np.clip(final_canvas * 255.0, 0, 255).astype(np.uint8)

cv2.imwrite('final_image.png', final_canvas)
with open('pin_sequence.txt', 'w') as f:
    for pin, color in pin_sequence:
        f.write(f"{pin} {color}\n")
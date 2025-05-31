import os
import sys
import argparse
import tifffile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from normalization import clip  # Not used in the current code
from extract_roi import crop

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.bce_config import NORM_LOWER_BOUND, NORM_UPPER_BOUND, CROPPING_REGIONS

PADDING = 3
COLLAGE_PADDING = 8
FONT_SIZE = 48
TOP_RIGHT_COLOR = 'magenta'  
CENTER_COLOR = 'yellow'      
BORDER_WIDTH = 5
COLLAGE_BG_COLOR = 'white'
TEXT_COLOR = 'white'
TEXT_FONT = "arial.ttf"

def draw_color_roi(draw):
    """Draw rectangles for the defined regions of interest (ROI) with padding."""
    for region_name, region in CROPPING_REGIONS.items():
        top_x1, top_y1 = region['top_left']
        top_x2, top_y2 = region['bottom_right']

        # Calculate rectangle size with padding
        padded_x1 = top_x1 - PADDING
        padded_y1 = top_y1 - PADDING
        padded_x2 = top_x2 + PADDING
        padded_y2 = top_y2 + PADDING

        width = padded_x2 - padded_x1
        height = padded_y2 - padded_y1

        print(f"Drawing rectangle for region '{region_name}':")
        print(f"Original Top-left corner: ({top_x1}, {top_y1})")
        print(f"Original Bottom-right corner: ({top_x2}, {top_y2})")
        print(f"Padded Top-left corner: ({padded_x1}, {padded_y1})")
        print(f"Padded Bottom-right corner: ({padded_x2}, {padded_y2})")
        print(f"Width (including padding): {width}")
        print(f"Height (including padding): {height}")

        color = TOP_RIGHT_COLOR if region_name == 'top-right' else CENTER_COLOR

        # Draw the rectangle with padding
        draw.rectangle(
            (padded_x1, padded_y1, padded_x2, padded_y2),
            outline=color,
            width=BORDER_WIDTH
        )

def add_border(img, color):
    """Add a border around the image."""
    img_with_border = Image.new('RGB', (img.width + 2 * COLLAGE_PADDING, img.height + 2 * COLLAGE_PADDING), color)
    img_with_border.paste(img, (COLLAGE_PADDING, COLLAGE_PADDING))
    return img_with_border

def main():
    src_path = r'D:\datasets\луковица\experiments_data\300_slices_for_markup_norm\reco_000774.tif'

    img = tifffile.imread(src_path)

    norm_img = (img - img.min()) / (img.max() - img.min())
    print("Normalized image value range:", np.min(norm_img), np.max(norm_img))

    img_to_save = (norm_img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img_to_save).convert('RGB')
    draw = ImageDraw.Draw(pil_img)

    # Draw rectangles with padding for the ROIs
    draw_color_roi(draw)

    # Extract regions of interest (ROI)
    roi1 = Image.fromarray((crop(norm_img, 'top-right') * 255).astype(np.uint8)).convert('RGB')
    roi2 = Image.fromarray((crop(norm_img, 'center') * 255).astype(np.uint8)).convert('RGB')

    roi1.save('top-right.png')
    roi2.save('center.png')

    width = 2 * roi1.width
    ratio = pil_img.height / pil_img.width
    height = int(width * ratio)

    collage_width = width
    collage_height = height + roi1.size[1]

    pil_img = pil_img.resize((width + 2 * COLLAGE_PADDING, height))

    # Create the collage
    collage = Image.new('RGB', (collage_width, collage_height), COLLAGE_BG_COLOR)
    
    collage.paste(pil_img, (0, 0))
    collage.paste(roi1, (0, height))
    collage.paste(roi2, (roi1.size[0], height))

    draw = ImageDraw.Draw(collage)
    draw.rectangle([0, height, roi1.size[0], height + roi1.size[1]], outline=TOP_RIGHT_COLOR, width=COLLAGE_PADDING)
    draw.rectangle([roi1.size[0], height, collage_width, collage_height], outline=CENTER_COLOR, width=COLLAGE_PADDING)

    font = ImageFont.truetype(TEXT_FONT, FONT_SIZE)

    # draw.text((20, height - 65 + COLLAGE_PADDING), "a", fill=TEXT_COLOR, font=font)
    # draw.text((20, collage.height - 65), "b", fill=TEXT_COLOR, font=font)
    # draw.text((collage.width // 2 + 20, collage.height - 65), "c", fill=TEXT_COLOR, font=font)

    # collage.save('roi_collage.png')

if __name__ == "__main__":
    main()

# ! encoding:UTF-8
import time
import os
import cv2
import argparse
import point2svg_div as ps
from point2svg_div import Point2svg
from decode import Decode_np
import decode as d
import vectorize_utils as vu
import torch
from DRL.actor import ResNet
import numpy as np
import glob
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
parser.add_argument('--actor', default='./model/actor.pkl', type=str, help='Actor model')
parser.add_argument('--input_folder', default='./image/', type=str, help='input folder containing images')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=32, type=int, help='divide the target image to get better resolution')
parser.add_argument('--width', default=128, type=int, help='width of each patch')
parser.add_argument('--output_dir', default='./output/', type=str, help='output path')
parser.add_argument('--batch_size', default=5, type=int, help='number of images to process before cleanup')
args = parser.parse_args()

width = args.width
divide = args.divide
output_dir = args.output_dir
canvas_cnt = divide * divide
use_patch_fill = True
use_PM = False  # Whether to use the pruning module

def cleanup_intermediate_files():
    """Clean up all intermediate files and directories"""
    # Clean up output_np directory
    if os.path.exists('./output_np/'):
        shutil.rmtree('./output_np/')
    
    # Clean up any remaining temporary files in output directory
    for root, dirs, files in os.walk(output_dir):
        for dir_name in dirs:
            if dir_name == 'tmp':
                shutil.rmtree(os.path.join(root, dir_name))
        for file_name in files:
            if file_name.endswith('.png') and not file_name == 'target.png':
                os.remove(os.path.join(root, file_name))

def process_image(img_path, output_base_dir, actor):
    # Initialize tensors for this image
    T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
    coord = vu.get_coord(width=width, device=device)
    
    # Create image-specific output directory
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img_output_dir = os.path.join(output_base_dir, img_name)
    os.makedirs(img_output_dir, exist_ok=True)
    
    # Create subdirectories for different outputs
    svg_dir = os.path.join(img_output_dir, 'svg')
    tmp_dir = os.path.join(img_output_dir, 'tmp')
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Load and process image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img0 = img
    (h, w) = img.shape
    origin_shape = (img.shape[1], img.shape[0])

    if use_patch_fill is True:
        canvas, patch_done_list = ps.patch_fill(img=img, div_num=divide)
        canvas = cv2.resize(canvas, (width * args.divide, width * args.divide)).astype('float32')
        canvas = torch.from_numpy(canvas / 255)
        canvas = canvas.unsqueeze(0).unsqueeze(0).to(device)
    else:
        canvas = torch.ones([1, 1, width, width]).to(device)
        _, patch_done_list = ps.patch_fill(img=img, div_num=args.divide)

    patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
    patch_img = vu.binarize(patch_img)
    patch_img = vu.gray_div(patch_img)
    r = h / w
    p = cv2.resize(patch_img, (int(width * args.divide), int(width * args.divide * r)))
    cv2.imwrite(filename=os.path.join(img_output_dir, 'target.png'), img=p)
    patch_img = vu.large2small(patch_img, canvas_cnt, divide, width)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).to(device).float() / 255.

    img = cv2.resize(img, (width, width))
    img = img.reshape(1, width, width, 1)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.tensor(img).to(device).float() / 255.

    p2s = Point2svg(width=width, div_num=divide, save_path=tmp_dir, init_num=0, img_w=w, img_h=h, img=img0,
                    use_patch_fill=use_patch_fill, patch_done_list=patch_done_list)

    act_list = []

    with torch.no_grad():
        if args.divide != 0:
            canvas = canvas[0].detach().cpu().numpy()
            canvas = np.transpose(canvas, (1, 2, 0))
            canvas = cv2.resize(canvas, (width * args.divide, width * args.divide))
            canvas = vu.large2small(canvas, canvas_cnt, divide, width)
            canvas = np.transpose(canvas, (0, 3, 1, 2))
            canvas = torch.tensor(canvas).to(device).float()
            coord = coord.expand(canvas_cnt, 2, width, width)
            T = T.expand(canvas_cnt, 1, width, width)
            vu.save_img(canvas, args.imgid, divide_number=divide, width=width, origin_shape=origin_shape, divide=True)
            args.imgid += 1
            start = time.time()
            for i in range(args.max_step):
                stepnum = T * i / args.max_step
                actions = actor(torch.cat([canvas, patch_img, stepnum, coord], 1))
                p2s.reset_gt_patch(gt=patch_img)
                canvas, res = vu.decode_list(actions, canvas)
                print('divided canvas step {}, Loss = {}'.format(i, ((canvas - patch_img) ** 2).mean()))
                p2s.add_action_div(actions)
                vu.save_img(canvas, args.imgid, divide_number=divide, width=width, origin_shape=origin_shape,
                            divide=True)
                args.imgid += 1

            end1 = time.time()
            unless_time = p2s.draw_action_list_for_all_patch(path_or_circle='path')
            unless_time2_s = time.time()
            d = Decode_np(div_num=divide, use_PM=use_PM)
            unless_time2_e = time.time()
            d.draw_decode()
            end2 = time.time()

            time_actor = end1 - start
            time_paint = end2 - end1 - unless_time - (unless_time2_e - unless_time2_s)
            print("actor time is : {}".format(time_actor))
            print("paint time is : {}".format(time_paint))
            
            # Return the paths for the final SVG and its destination
            final_svg = os.path.join(tmp_dir, f"{args.imgid-1}.svg")
            dest_svg = os.path.join(svg_dir, f"{img_name}.svg")
            return final_svg, dest_svg, tmp_dir

if __name__ == '__main__':
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load actor model
    actor = ResNet(5, 18, 9)
    actor.load_state_dict(torch.load(args.actor))
    actor = actor.to(device).eval()
    
    # Process all images in input folder
    image_files = glob.glob(os.path.join(args.input_folder, "*.png")) + \
                 glob.glob(os.path.join(args.input_folder, "*.jpg")) + \
                 glob.glob(os.path.join(args.input_folder, "*.jpeg"))
    
    # Process images in batches
    for i in range(0, len(image_files), args.batch_size):
        batch = image_files[i:i + args.batch_size]
        print(f"\nProcessing batch {i//args.batch_size + 1} of {(len(image_files) + args.batch_size - 1)//args.batch_size}")
        
        # Store paths of SVGs to move
        svgs_to_move = []
        tmp_dirs = []
        
        for img_path in batch:
            try:
                result = process_image(img_path, args.output_dir, actor)
                if result:
                    final_svg, dest_svg, tmp_dir = result
                    svgs_to_move.append((final_svg, dest_svg))
                    tmp_dirs.append(tmp_dir)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        # Move all SVGs to their final destinations
        print("\nMoving final SVGs to their destinations...")
        for final_svg, dest_svg in svgs_to_move:
            if os.path.exists(final_svg):
                shutil.move(final_svg, dest_svg)
                print(f"Moved {final_svg} -> {dest_svg}")
        
        # Clean up intermediate files after each batch
        print(f"\nCleaning up intermediate files after batch {i//args.batch_size + 1}")
        cleanup_intermediate_files()
    
    print("\nAll images processed successfully!")

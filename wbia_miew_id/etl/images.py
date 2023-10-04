import os
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from datasets import get_chip_from_img

def process_image(row, crop_bbox, preprocess_dir, chip_idx, target_size):
    image_path = row['file_path_orig']
    bbox = row['bbox']
    theta = row['theta'] if row['theta'] is not None else 0

    target_h, target_w = target_size

    image = cv2.imread(image_path)

    if crop_bbox:
        image = get_chip_from_img(image, bbox, theta)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(image)
    pil_image_resized = pil_image.resize((target_w, target_h))

    image_name = f"chip_{chip_idx:012}.jpg"
    out_path = os.path.join(preprocess_dir, image_name)
    pil_image_resized.save(out_path)
    return row.name, out_path


def preprocess_images(df, crop_bbox, preprocess_dir, target_size, num_workers=None):
    df['file_path_orig'] = df['file_path']
    # common_prefix = os.path.commonprefix(df['file_path'].tolist())
    # base_path = os.path.dirname(common_prefix)
    # df['reduced_path'] = df['file_path'].apply(lambda x: x.replace(base_path, ''))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, row, crop_bbox, preprocess_dir, chip_idx, target_size) for chip_idx, row in tqdm(df.iterrows(), total=len(df))]

    for future in as_completed(futures):
        index, out_path = future.result()
        df.at[index, 'file_path'] = out_path

    df['file_path'].apply(os.path.abspath)

    return df

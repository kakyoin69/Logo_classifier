import argparse
import sys
from pathlib import Path
from loguru import logger
import yaml
from src import DataLoader, ProcessManager


def load_yaml_config(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        # If the file is empty, safe_load returns None, so return an empty dict.
        return config if config is not None else {}
    except Exception as e:
        logger.error(f"Failed to load config file '{file_path}': {e}")
        return {}
    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load data paths for text, and image from command-line arguments if config is empty."
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Text question(optional): "Is it Mazda?"',
        default=None
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path or URL to image file: "./1.jpg" or "url"',
        default=None
    )
    parser.add_argument(
        '--images',
        type=str,
        help='Path to folder this to image files like list type(optional): "./1.jpg,url,...etc"',
        default=None
    )
    parser.add_argument(
        '--classes',
        type=str,
        help='Path to classes.txt file',
        default=None
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='Path to efficientnet_logo2k.pth file ',
        default=None
    )
    args = parser.parse_args()

    if args.images:
        args.images = [path.strip() for path in args.images.split(",")]  # Разделяем по запятым
    else:
        args.images = []
    return args


def process_answer(answer):
    if 'sim' in answer:
        sim_data = answer['sim']
        for brand, imgs in sim_data.items():
            logger.info(f"Logo: {brand} || Similar images:")

            for img in imgs:
               logger.info(f" image: {img}")

    else:
        logger.info("No logo matches found.")

    # Проверяем наличие ключа 'brand'
    if 'brand' in answer:
        brand_info = answer['brand']
        logger.info(f"{brand_info}")


def main():
    args = parse_args()
    default_config_path = Path("config") / "config.yaml"
    config = load_yaml_config(str(default_config_path))
    
    try:
        data_loader = DataLoader(config, args)
        data_paths = data_loader.load_data()
        process_manager = ProcessManager(data_paths)
        answer = process_manager.worker()
        process_answer(answer=answer)
        
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    

if __name__ == "__main__":
    main()
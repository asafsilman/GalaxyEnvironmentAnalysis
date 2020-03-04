import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def move_files_in_list(file_list, destination_dir: Path):
    destination_dir.mkdir(exist_ok=True, parents=True)
    for image_file in file_list:
        label = image_file.parent.stem
        label_dir = destination_dir / str(label)
        label_dir.mkdir(exist_ok=True)

        destination_file = label_dir/image_file.name
        logger.debug(f"Moving {image_file} to {destination_file}")
        image_file.replace(destination_file)

    logger.info(f"Moved {len(file_list)} images to {destination_dir}")

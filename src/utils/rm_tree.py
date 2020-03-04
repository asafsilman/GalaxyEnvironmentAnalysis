import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def rm_tree(pth: Path, depth: int=0):
    for child in pth.iterdir():
        if child.is_file():
            if child.name == ".gitkeep":
                continue
            logger.debug(f"Deleting file: {child}")
            child.unlink()
        else:
            rm_tree(child, depth+1)
    if depth > 0:
        logger.debug(f"Deleting directory: {pth}")
        pth.rmdir()
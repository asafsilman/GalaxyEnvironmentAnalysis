import logging
import tarfile
from hashlib import sha1
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.rm_tree import rm_tree

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 50

def make_data_set(config):
    data_raw_path = Path(config.get('data_raw_path', 'data/raw'))
    data_interim_path = Path(config.get('data_interim_path', 'data/interim'))

    assert data_raw_path.is_dir(), 'Raw data path is not a valid directory'
    assert data_interim_path.is_dir(), 'Interim data path is not a valid directory'

    rm_tree(data_interim_path)

    data_files = config.get('data_files', [])
    data_labels = config.get('data_labels', None)

    logger.debug(f"Processing raw data files in {data_raw_path}")
    for data_file_info in data_files:
        data_set_type = data_file_info['data_set_type']

        if data_set_type == 'categorical':
            file_name = data_file_info['file']
            data_file_path = data_raw_path.joinpath(file_name)
            assert data_file_path.is_file(), f"Data file {file_name} does not exist"
            logger.info(f"Processing data file '{file_name}' as '{data_set_type}'")

            make_data_set_categorical(data_file_info, data_file_path, data_interim_path, data_labels)
        elif data_set_type == 'categorical_2_channel':
            file_name_1 = data_file_info['file_1']
            file_name_2 = data_file_info['file_2']

            data_file_path_1 = data_raw_path.joinpath(file_name_1)
            data_file_path_2 = data_raw_path.joinpath(file_name_2)

            assert data_file_path_1.is_file(), f"Data file {file_name_1} does not exist"
            assert data_file_path_2.is_file(), f"Data file {file_name_2} does not exist"

            make_data_set_categorical_2_channel(
                data_file_info,
                data_file_path_1,
                data_file_path_2,
                data_interim_path,
                data_labels
            )
        
        elif data_set_type == 'regression':
            raise NotImplementedError('Regression dataset type not implemented yet')
        else:
            raise ValueError(f'Did not recognise data_set_type of {data_set_type}')

def make_data_set_categorical(data_file_info, data_file_path, data_interim_path, data_labels):
    image_size = data_file_info.get("image_size", DEFAULT_IMAGE_SIZE)

    image_file_name = data_file_info.get("image_file_name", "m1.dir/2dft.dat")
    label_file_names = data_file_info.get("label_file_names", "m1.dir/2dftn1.dat")
    if isinstance(label_file_names, str):
        label_file_names = [label_file_names]

    label_start = data_file_info.get("label_start", 0)
    
    logger.debug(f"Openning tar archive {data_file_path}")
    with tarfile.open(data_file_path) as archive:
        # Extract data files (image and labels)
        label_files = []

        for label_file_name in label_file_names:
            logger.debug(f"Reading label data in archive with path {label_file_name}")
            label_files.append(archive.extractfile(label_file_name))
        logger.debug(f"Reading image data in archive with path {image_file_name}")
        image_file = archive.extractfile(image_file_name)

        # Load data using np.loadtxt
        label_data_list = []
        for label_file in label_files:
            logger.debug(f"Loading label data in archive with path {label_file_name}")
            label_data = np.loadtxt(label_file, dtype=np.float32)
            label_data_list.append(label_data)
        logger.debug(f"Loading image data in archive with path {image_file_name}")
        image_data = np.loadtxt(image_file, dtype=np.float32)

        # Count number of rows
        n_rows = label_data_list[0].shape[0]

        # Reshape data using numpy reshape
        for i, label_data in enumerate(label_data_list):
            label_data_list[i] = label_data.reshape(n_rows, 1)
        label_data = np.concatenate(label_data_list, axis=1)
        logger.debug(f"Processing label data. Got shape {label_data.shape}")

        image_data = image_data.reshape(n_rows, image_size, image_size)
        logger.debug(f"Processing image data. Got shape {image_data.shape}")

        for i in range(n_rows):
            image = image_data[i]
            label = np.argmax(label_data[i]) + label_start
            
            if data_labels:
                label = data_labels.get(label, 'unclassified')

            label_dir = data_interim_path / str(label)
            label_dir.mkdir(exist_ok=True)

            image_hash = sha1(image).hexdigest()
            image_save_path = label_dir / f"{image_hash}.png"
            _image = image =  Image.fromarray(image * 255).convert("L")

            _image.save(image_save_path)

            logging.debug(f"Extracted image hash {image_hash} with label {label} to {image_save_path}")

def make_data_set_categorical_2_channel(data_file_info, data_file_path_1, data_file_path_2, data_interim_path, data_labels):
    image_size = data_file_info.get("image_size", DEFAULT_IMAGE_SIZE)

    image_file_name = data_file_info.get("image_file_name", "m1.dir/2dft.dat")
    label_file_name = data_file_info.get("label_file_name", "m1.dir/2dftn1.dat")

    label_start = data_file_info.get("label_start", 0)
    
    logger.debug(f"Openning tar archive {data_file_path_1}")
    with tarfile.open(data_file_path_1) as archive:
        logger.debug(f"Openning tar archive {data_file_path_2}")
        with tarfile.open(data_file_path_2) as archive_2:
            # Extract data files (image and labels)
            logger.debug(f"Reading image data in archive with path {image_file_name}")
            image_file_1 = archive.extractfile(image_file_name)
            image_data_1 = np.loadtxt(image_file_1, dtype=np.float32)

            logger.debug(f"Reading image data in archive_2 with path {image_file_name}")
            image_file_2 = archive_2.extractfile(image_file_name)
            image_data_2 = np.loadtxt(image_file_2, dtype=np.float32)

            logger.debug(f"Loading label data in archive with path {label_file_name}")
            label_file = archive.extractfile(label_file_name)
            label_data = np.loadtxt(label_file, dtype=np.float32)

            # Count number of rows
            n_rows = label_data.shape[0]

            logger.debug(f"Processing label data. Got shape {label_data.shape}")

            image_data_1 = image_data_1.reshape(n_rows, image_size, image_size)
            logger.debug(f"Processing image data 1. Got shape {image_data_1.shape}")

            image_data_2 = image_data_2.reshape(n_rows, image_size, image_size)
            logger.debug(f"Processing image data 2. Got shape {image_data_2.shape}")

            for i in range(n_rows):
                image_1 = image_data_1[i]
                image_2 = image_data_2[i]
                label = np.argmax(label_data[i]) + label_start
                
                # if label data specified, adjust label name
                if data_labels:
                    label = data_labels.get(label, 'unclassified')

                label_dir = data_interim_path / str(label)
                label_dir.mkdir(exist_ok=True)

                image_final = np.stack([image_1, image_2, np.zeros(image_1.shape)], axis=-1)
                image_final *= 255
                image_final = image_final.astype(np.uint8)

                image_hash = sha1(image_final).hexdigest()
                image_save_path = label_dir / f"{image_hash}.png"
                _image = image =  Image.fromarray(image_final).convert("RGB")
                _image.save(image_save_path)

                # image_hash = sha1(image_1).hexdigest()

                # # Save image 1
                # image_save_path = label_dir / f"{image_hash}_1.png"
                # _image = image =  Image.fromarray(image_1 * 255).convert("L")
                # _image.save(image_save_path)

                #  # Save image 1
                # image_save_path = label_dir / f"{image_hash}_2.png"
                # _image = image =  Image.fromarray(image_2 * 255).convert("L")
                # _image.save(image_save_path)

                logging.debug(f"Extracted image hash {image_hash} with label {label} to {image_save_path}")

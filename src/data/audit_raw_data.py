import hashlib
import csv
import tarfile
from pathlib import Path
import numpy as np

import logging

logger = logging.getLogger(__name__)

class AuditReportSingleFile:
    def __init__(self, data_id, file_path, expected_file_hash, expected_file_availability):
        self.data_id = data_id
        self.file_path = file_path
        self.expected_file_hash = expected_file_hash
        self.expected_file_availability = expected_file_availability

        self.file_availability = False
        self.file_hash = ""

    def audit_file(self):
        self.file_availability = self.file_path.exists()

        if self.file_availability == True:
            self.file_hash = sha256sum(self.file_path)

    def audit_pass(self):
        check = self.expected_file_hash==self.file_hash and \
            self.expected_file_availability == self.file_availability

        return check

    def __repr__(self):
        return f"AuditReportSingleFile({self.data_id}, \
            '{self.file_path}', \
            '{self.expected_file_hash}', \
            {self.expected_file_availability})"

    def __str__(self):
        return self.__repr__()

def sha256sum(filename):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()



def audit_single_file(data_files_dict, config, data_id):
    data_file = data_files_dict.get(data_id)
    if data_file is None:
        raise Exception(f"Could not file dataId: {data_id}")

    raw_data_path = Path(config.get("data_raw_path", "data/raw"))
    image_pixels = config.get("image_height", 50) * config.get("image_width", 50)

    file_path = raw_data_path.joinpath(data_file["dataFile"])
    data_id = data_file["dataId"]
    file_hash = data_file["fileHash"]
    file_availability = data_file["fileAvailable"]

    audit = AuditReportSingleFile(data_id, file_path, file_hash, file_availability)
    audit.audit_file()

    return audit

def audit_all_files(data_files_dict, config, out_file):
    audits = []

    for idx in data_files_dict:
        row = data_files_dict[idx]
        
        audits.append(
            audit_single_file(data_files_dict, config, idx)
        )

    with open(out_file, "w") as f:
        writer = csv.DictWriter(f, [
            "data_id",
            "file_path",
            "audit_pass",
            "expected_file_hash",
            "expected_file_availability",
            "file_hash",
            "file_availability"
        ])
        writer.writeheader()

        for audit in audits:
            data = audit.__dict__
            data["audit_pass"] = audit.audit_pass()

            writer.writerow(data)
            f.flush()

def generate_file_catalog(data_files_dict, config, data_id=None, extract_files=True, out_file="out.csv"):
    raw_data_path = Path(config.get("data_raw_path", "data/raw"))
    image_pixels = config.get("image_height", 50) * config.get("image_width", 50)

    with open(out_file, "w") as f:
        writer = csv.DictWriter(f, ["dataId", "fileAvailable", "fileHash", "imageFileNumRows", "labelFileNumRows"])
        writer.writeheader()
        for idx in data_files_dict:
            row = data_files_dict[idx]
            report_line = {}
            
            report_line["dataId"] = row["dataId"]
            if data_id:
                if int(row["dataId"]) != int(data_id):
                    logger.debug(f"Skipping: {row['dataId']}")
                    continue

            logger.info(f"Processing dataId {row['dataId']}")
            
            file_path = raw_data_path.joinpath(row["dataFile"])
            logger.debug(f"File Path: {file_path}")
            
            file_available = file_path.exists()
            report_line["fileAvailable"] = file_available
            logger.debug(f"File availability: {file_available}")

            if file_available:
                report_line["fileHash"] = sha256sum(file_path)
                
                if extract_files:
                    logger.debug("Extracting tarfile archive")
                    with tarfile.open(file_path) as archive:
                        # try get image num rows
                        image_file_path = row["imageFileName"]
                        try:
                            image_file = archive.extractfile(image_file_path)
                            image_data = np.loadtxt(image_file, dtype=np.float32)

                            report_line["imageFileNumRows"] = (image_data.shape[0] // image_pixels)
                        except Exception as e:
                            report_line["imageFileNumRows"] = 0

                        # try get label num rows
                        label_file_path = row["labelFileName"]
                        try:
                            label_file = archive.extractfile(label_file_path)
                            label_data = np.loadtxt(label_file, dtype=np.float32)

                            report_line["labelFileNumRows"] = label_data.shape[0]
                        except Exception as e:
                            report_line["labelFileNumRows"] = 0
            
            logger.debug(f"Processed dataId {row['dataId']}, writing to file")
            writer.writerow(report_line)
            f.flush()

if __name__=="__main__":
    from src.config.load_workbook import load_workbook
    config = {"model_config_path": "model_config.xlsx"}
    _, data_files = load_workbook(config)
    generate_file_catalog(data_files, config)

    # audit_single_file(data_files, config, 23)
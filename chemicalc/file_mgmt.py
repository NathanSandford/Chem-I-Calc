from typing import Dict, List, Union, Optional
import os
from pathlib import Path
import requests
from tqdm import tqdm


data_dir: Path = Path(os.path.dirname(__file__)).joinpath("data")
data_dir.mkdir(exist_ok=True)
ref_label_file = data_dir.joinpath("reference_labels.h5")

precomputed_res: List = [300000]
precomputed_ref_id: Dict[float, str] = {300000: "1I9GzorHm0KfqJ-wvZMVGbQDeyMwEu3n2"}
precomputed_cont_id: Dict[float, str] = {300000: "1Fhx1KM8b6prtCGOZ3NazVeDQY-x9gOOU"}
precomputed_label_id: str = "1-qCCjDXp2eNzRGCfIqI_2JZrzi22rFor"

def download_package_files(id, destination):
    """

    :param id:
    :param destination:
    :return:
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    def save_response_content(response, destination):
        chunk_size = 32768
        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size)):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={"id": id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {"id": id, "confirm": token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, destination)


def download_all_files(overwrite=True):
    """

    :param overwrite:
    :return:
    """
    if ref_label_file.exists() and not overwrite:
        print(f"{ref_label_file} exists")
    else:
        print(f"Downloading {ref_label_file}.h5")
        download_package_files(id=precomputed_label_id, destination=ref_label_file)

    for res in precomputed_res:
        resolution = precomputed_res[res]
        reference_file = data_dir.joinpath(f"reference_spectra_{resolution:06}.h5")
        reference_id = precomputed_ref_id[res]
        if reference_file.exists() and not overwrite:
            print(f"{reference_file} exists")
        else:
            print(f"Downloading {reference_file}")
            download_package_files(id=reference_id, destination=reference_file)
        continuum_file = data_dir.joinpath(f"reference_continuum_{resolution:06}.h5")
        continuum_id = precomputed_cont_id[res]
        if continuum_file.exists() and not overwrite:
            print(f"{continuum_file} exists")
        else:
            print(f"Downloading {continuum_file}")
            download_package_files(id=continuum_id, destination=continuum_file)


def check_label_format():
    raise NotImplementedError("Coming soon!")


def check_spec_format():
    raise NotImplementedError("Coming soon!")
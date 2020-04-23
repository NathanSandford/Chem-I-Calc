from typing import Dict, List, Union, Optional
import os
from pathlib import Path
import requests
from tqdm import tqdm


data_dir: Path = Path(os.path.dirname(__file__)).joinpath("data")
data_dir.mkdir(exist_ok=True)
inst_file = data_dir.joinpath('instruments.json')
ref_label_file = data_dir.joinpath("reference_labels.h5")

precomputed_res: List = [300000]
precomputed_ref_id: Dict[float, str] = {300000: "1I9GzorHm0KfqJ-wvZMVGbQDeyMwEu3n2"}
precomputed_label_id: str = "1-qCCjDXp2eNzRGCfIqI_2JZrzi22rFor"

precomputed_alpha_included: List[str] = [
    "MSTO_m1.5",
    "RGB_m1.0",
    "RGB_m1.5",
    "RGB_m2.0",
    "RGB_m2.5",
    "TRGB_m1.5",
    "Ph_k0i_m0.0",
    "Ph_k0i_m1.0",
    "Ph_k5iii_m0.0",
    "Ph_k5iii_m1.0",
]


def check_label_format(labelfile: str) -> None:
    """
    Warning: Not Implemented Yet

    :return:
    """
    raise NotImplementedError("Coming soon!")


def check_spec_format(specfile: str) -> None:
    """
    Warning: Not Implemented Yet

    :return:
    """
    raise NotImplementedError("Coming soon!")


def download_package_files(id: str, destination: Union[str, Path]) -> None:
    """
    Generic function to download large file from Google Drive

    :param str id: Google Drive file ID
    :param Union[str,Path] destination: Path to download location
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


def download_all_files(overwrite: bool = True) -> None:
    """
    Downloads all external files: Label File, Normalized Spectra File(s)

    :param bool overwrite: Overwrite existing files
    :return:
    """
    if ref_label_file.exists() and not overwrite:
        print(f"{ref_label_file} exists")
    else:
        print(f"Downloading {ref_label_file}")
        download_package_files(id=precomputed_label_id, destination=ref_label_file)

    for res in precomputed_res:
        reference_file = data_dir.joinpath(f"reference_spectra_{res:06}.h5")
        reference_id = precomputed_ref_id[res]
        if reference_file.exists() and not overwrite:
            print(f"{reference_file} exists")
        else:
            print(f"Downloading {reference_file}")
            download_package_files(id=reference_id, destination=reference_file)

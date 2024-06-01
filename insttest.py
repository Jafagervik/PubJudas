import os
from datetime import datetime

import h5py
import requests


def streaaams():
    urls = ["http://example.com/file1.pdf", "http://example.com/file2.pdf"]
    with requests.Session() as session:
        for url in urls:
            with session.get(url) as response:
                filename = url.split("/")[-1]
                with open(filename, "wb") as f:
                    f.write(response.content)


def format_url(url: str):
    name = url.split("/")[-1]
    name = name.split("_")
    return name[-2] + "_" + name[-1]
    # return os.path.splitext(name)[0]


def get_datetime_file(file_name: str):
    tiss = file_name.split("_")[-2] + "_" + file_name.split("_")[-1]

    return datetime.strptime(tiss, "%Y%m%d_%H%M%S")


def downloadFile(url: str):
    file_name = format_url(url)
    print(file_name)
    path = os.path.join(os.getcwd(), "data", file_name)
    print(path)
    with open(path, "wb") as file:
        response = requests.get(url)
        file.write(response.content)

    print(f"Downloaded {url} to {file_name}")


url = "https://g-a51e00.a1bfb5.bd7c.data.globus.org/FORESEE/Data/202003/FORESEE_UTC_20200301_000015.hdf5"

print(get_datetime_file(format_url(url)))

# downloadFile(url)

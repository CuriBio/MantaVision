import os
import glob
import zipfile
from typing import Tuple, List
from tkinter import Tk as tk
from tkinter.filedialog import askopenfilename, askdirectory


def zipDir(input_dir_path: str, zip_file_path: str, sdk_files_only: bool = False):
    zip_file = zipfile.ZipFile(zip_file_path, 'w')
    for dir_name, _, file_names in os.walk(input_dir_path):
        for file_name in file_names:
            if sdk_files_only and 'sdk' not in file_name:
                continue  # only include files intended for the sdk
            file_path = os.path.join(dir_name, file_name)
            zip_file.write(file_path, os.path.basename(file_path))
    zip_file.close()


def contentsOfDir(
    dir_path: str,
    search_terms: List[str],
    search_extension_only: bool = True
) -> Tuple[str, List[Tuple[str, str]]]:
    """ return the base directory path and list of [file_name, file_extension] tuples """
    all_files_found = []
    if os.path.isdir(dir_path):
        base_dir = dir_path
        for search_term in search_terms:
            glob_search_term = '*' + search_term
            if not search_extension_only:
                glob_search_term += '*'
            files_found = glob.glob(os.path.join(dir_path, glob_search_term))
            if len(files_found) > 0:
                all_files_found.extend(files_found)
    else:
        # presume it's actually a single file path
        base_dir = os.path.dirname(dir_path)
        all_files_found = [dir_path]
    if len(all_files_found) < 1:
        return None, None

    files = []
    for file_path in all_files_found:
        file_name, file_extension = os.path.splitext(os.path.basename(file_path))
        files.append((file_name, file_extension))
    return base_dir, files


def fileNameParametersForSDK(file_name: str) -> Tuple[str, str]:
    """ return well name and data stamp from expected positions if the file is long enough to contain them """
    well_name = 'A001'  # default in case the file doesn't/can't contain one
    num_chars_in_well_name = len(well_name)
    date_stamp = '2020-02-02'  # default in case the file doesn't/can't contain one
    num_chars_in_datestamp = len(date_stamp)
    min_num_chars_in_file_name = num_chars_in_datestamp + num_chars_in_well_name
    if len(file_name) >= min_num_chars_in_file_name:
        file_name_length = len(file_name)
        well_name = file_name[file_name_length - num_chars_in_well_name:]
        date_stamp = file_name[:num_chars_in_datestamp]
    return well_name, date_stamp


def getDirPathViaGUI(window_title: str = '') -> str:
    """ Display an "Open" dialog box and return the path to a selected directory """
    window = tk()
    window.withdraw()
    window.lift()
    window.overrideredirect(True)
    window.call('wm', 'attributes', '.', '-topmost', True)
    return askdirectory(
        initialdir='./',
        title=window_title
    )


def getFilePathViaGUI(window_title: str = '') -> str:
    """ Display an "Open" dialog box and return the path to a selected file """
    window = tk()
    window.withdraw()
    window.lift()
    window.overrideredirect(True)
    window.call('wm', 'attributes', '.', '-topmost', True)
    return askopenfilename(
        initialdir='./',
        title=window_title
    )

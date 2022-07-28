import os
import glob
from typing import Tuple, List
from tkinter import Tk as tk
from tkinter.filedialog import askopenfilename, askdirectory


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


def contentsOfDir(
        dir_path: str,
        search_terms: List[str],
        search_extension_only: bool = True
) -> Tuple[List[str], List[Tuple[str]]]:
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

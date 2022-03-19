""" there to indicate the paths."""
import os

from corai_util.tools.src.function_writer import factory_fct_linked_path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

linker_path_to_result_file = factory_fct_linked_path(ROOT_DIR, 'data_result')
linker_path_to_data_file = factory_fct_linked_path(ROOT_DIR, 'data_input')
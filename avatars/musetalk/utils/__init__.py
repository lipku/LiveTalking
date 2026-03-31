import sys
from os.path import abspath, dirname
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
sys.path.append(parent_dir+'/utils')

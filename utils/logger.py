import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fhandler = logging.FileHandler('zl-talking.log')
fhandler.setFormatter(formatter)
fhandler.setLevel(logging.INFO)
logger.addHandler(fhandler)

shandler = logging.StreamHandler(sys.stdout)
shandler.setFormatter(formatter)
shandler.setLevel(logging.INFO)
logger.addHandler(shandler)

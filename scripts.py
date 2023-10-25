from datetime import datetime
from src.utils import get_gmst_from_epoch
import numpy as np

t = 1716123280.0
t2 = 1716122620.0
t3 = 1741668470.0
t4 = 1712606620.0
t5 = 1741668470.0

print(datetime.fromtimestamp(t4))
print(datetime.fromtimestamp(t5))
# print(get_gmst_from_epoch(t2))

# RAW Studio — лаунчер для Windows (без консольного вікна)
# Подвійний клік на цьому файлі відкриває застосунок, якщо Python вже встановлений.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import raw_studio

raw_studio.run_gui()

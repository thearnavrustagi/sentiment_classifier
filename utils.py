import re
from constants import PAD_TOKEN, START_TOKEN, END_TOKEN
from constants import SENTENCE_LEN

def text_preprocess(arg):
    arg = arg.lower()
    arg = re.compile('[^a-z1-9 ]').sub("",arg)
    arr = arg.split(" ")
    diff = max(SENTENCE_LEN - len(arr) - 2,0)
    return [START_TOKEN] + arr + [END_TOKEN] + [PAD_TOKEN]*diff

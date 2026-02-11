
import codecs
import sys
try:
    with codecs.open("prediction_vnd.txt", "r", "utf-16le") as f:
        for line in f:
            if "Price" in line and "VND" in line:
                print(line.strip())
except Exception as e:
    # Try utf-8 if utf-16 fails
    try:
        with open("full_log.txt", "r", encoding='utf-8') as f:
            print(f.read())
    except:
        print(e)

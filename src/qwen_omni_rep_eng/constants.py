EMOTIONS = ["anger","disgust","sadness","joy","neutral","surprise","fear"]
IDX = {e: i for i, e in enumerate(EMOTIONS)}

# For MELD we'll mostly care about this binary margin:
JOY = IDX["joy"]
SAD = IDX["sadness"]

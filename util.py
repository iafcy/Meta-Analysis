from datetime import datetime
import time

def time_str_to_seconds(time_str: str) -> int:
    time_obj = datetime.strptime(time_str, "%H:%M:%S")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def seconds_to_time_str(seconds: float) -> str:
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

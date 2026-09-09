import datetime

def now():
    """
        Returns Date-Time-TimeZone in ISO format upto seconds
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')





def now_frac():
    """
        Returns Date-Time-TimeZone in ISO format
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
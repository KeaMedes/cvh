from datetime import datetime


def run_with_time(msg, func):
    begin = datetime.now()
    ret = func()
    end = datetime.now()
    dur = end - begin
    print("%s finish, time usage: %f" % (msg, dur.total_seconds()))
    return ret

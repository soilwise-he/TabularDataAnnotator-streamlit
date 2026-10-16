import inspect, sys, os
orig_getframeinfo = inspect.getframeinfo
orig_getsourcefile = inspect.getsourcefile
orig_getfile = inspect.getfile

def debug_getframeinfo(frame, *args, **kwargs):
    try:
        info = orig_getframeinfo(frame, *args, **kwargs)
        print('DEBUG getframeinfo: frame_file=', getattr(frame, 'f_code', None).co_filename if getattr(frame, 'f_code', None) else None)
        print('DEBUG getframeinfo: info filename=', getattr(info, 'filename', None), 'type=', type(getattr(info, 'filename', None)))
        return info
    except Exception as e:
        print('DEBUG getframeinfo EXC', type(e).__name__, e)
        raise

def debug_getsourcefile(object, *args, **kwargs):
    try:
        result = orig_getsourcefile(object, *args, **kwargs)
        print('DEBUG getsourcefile object=', object, 'type=', type(object), '->', result)
        return result
    except Exception as e:
        print('DEBUG getsourcefile EXC', type(e).__name__, e, 'object=', object, 'type=', type(object))
        raise

inspect.getframeinfo = debug_getframeinfo
inspect.getsourcefile = debug_getsourcefile
print('patched inspect')
import torch
print('torch imported OK', torch.__version__)

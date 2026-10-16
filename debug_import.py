import sys
import types

print('hello from debug script', file=sys.stderr)

try:
    import zenodo_autopilot
    print('import_ok')
except Exception as e:
    import traceback
    traceback.print_exc()
    with open('debug_import_error.txt', 'w', encoding='utf-8') as f:
        f.write(traceback.format_exc())
    raise

with open('debug_import_ok.txt', 'w', encoding='utf-8') as f:
    f.write('import_ok')

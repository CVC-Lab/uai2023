"""
    This files contains a set of utilities helpful when using sacred.

    TODO: documentation
"""

import glob, sys, json, re

'''
    This function loads all the available runs inside the specified directory.
'''
def load_runs(base_directory):
    if base_directory[-1] != '/':
        base_directory += '/'
    runs = {}
    runs_filenames = glob.glob(base_directory + '*/config.json')
    run_extractor = re.compile(base_directory + '([0-9]+)/config.json')
    for r in runs_filenames:
        try:
            run_number = int(run_extractor.match(r).group(1))
            runs[run_number] = {}
            runs[run_number]['config'] = json.load(open(base_directory + str(run_number) + '/config.json'))
            runs[run_number]['run'] = json.load(open(base_directory + str(run_number) + '/run.json'))
            runs[run_number]['metrics'] = json.load(open(base_directory + str(run_number) + '/metrics.json'))
        except:
            del runs[run_number]
    return runs

'''
    TODO: docs
'''
def recursive_json_selector(obj, selector):
    try:
        if selector is not None:
            for i, qk in enumerate(selector.split('.')):
                if isinstance(obj, dict):
                    obj = obj[qk]
                elif isinstance(obj, list):
                    obj = [o[qk] for o in obj]
        return obj
    except:
        return None

'''
    This function returns a filtered dictionary containing all the runs that
    meet the requirements specified with the query parameter. Parameters at
    different levels of the run specification are given using the dot notation.
'''
def filter_runs(query, runs, avoid_missing=True):
    keys = list(runs.keys())
    for key, value in query.items():
        # Check if the still unfiltered runs have the specified parameter. If not, remove them
        _keys = []
        for run_key in keys:
            obj = runs[run_key]
            # Dot notation at any level
            obj = recursive_json_selector(obj, key)
            # Check if it matches value
            if obj is None and value is None:
                _keys.append(run_key)
            elif obj is None and not avoid_missing:
                _keys.append(run_key)
            elif obj is not None and obj == value:
                _keys.append(run_key)
            elif obj is not None and isinstance(obj, list) and value in obj:
                _keys.append(run_key)
        keys = _keys
    # Now create a filtered object only with the selected runs
    _runs = {key: runs[key] for key in keys}
    return _runs

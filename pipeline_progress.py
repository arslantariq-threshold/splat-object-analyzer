"""Shared progress messages for the local viewer and standalone API."""
import re
import json


def progress_from_line(line):
    if line.startswith('[environment] '):
        return json.loads(line[len('[environment] '):])
    if '[navigation]' in line:
        stage = 'Estimating camera space'
    elif '[render]' in line or '[pipeline] Step 1' in line or '[spz]' in line:
        stage = 'Selecting and rendering views'
    elif '[discover]' in line:
        stage = 'Discovering object names'
    elif '[detect] Processed' in line or 'Loading OWLv2' in line or '[pipeline] Step 2' in line:
        stage = 'Detecting objects'
    elif '[pipeline] Step 3' in line or '[cluster]' in line:
        stage = 'Associating detections'
    elif '[identity]' in line:
        stage = 'Segmenting objects and fitting boxes'
    else:
        return None
    match = re.search(r'(\d+)\s*/\s*(\d+)', line)
    return dict(stage=stage, completed=int(match[1]) if match else None, total=int(match[2]) if match else None)

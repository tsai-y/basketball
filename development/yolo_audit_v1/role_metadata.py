"""Optional visible-hand-held-ball attributes; never a fourth detection class."""
from copy import deepcopy

ROLE_PROTOCOL = 'visible-hand-held-ball-v1'
ROLE_VALUES = (None, 'yes', 'no', 'uncertain')


def validate_role_metadata(review):
    if 'role_metadata_protocol' in review and review['role_metadata_protocol'] != ROLE_PROTOCOL:
        raise ValueError('Unsupported role metadata protocol')
    counts = {'unreviewed': 0, 'yes': 0, 'no': 0, 'uncertain': 0}
    for row in review['images']:
        for annotation in row['annotations']:
            if annotation['class_id'] != 0:
                if 'holding_ball' in annotation:
                    raise ValueError('holding_ball is only allowed on active_player')
                continue
            value = annotation.get('holding_ball')
            if value not in ROLE_VALUES:
                raise ValueError('holding_ball must be null, yes, no or uncertain')
            counts['unreviewed' if value is None else value] += 1
    return counts


def add_role_metadata(review):
    """Clone legacy or role-enabled progress without inferring any role labels."""
    validate_role_metadata(review)
    result = deepcopy(review)
    result['role_metadata_protocol'] = ROLE_PROTOCOL
    for row in result['images']:
        for annotation in row['annotations']:
            if annotation['class_id'] == 0:
                annotation.setdefault('holding_ball', None)
    validate_role_metadata(result)
    return result

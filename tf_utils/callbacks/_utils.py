from typing import Iterable, Any


def only_one_is_not_none(*values) -> bool:

    found: bool = False

    for value in values:
        if value is not None:
            if found:
                return False
            else:
                found: bool = True
    else:
        return True
    

def isnumber(obj: Any) -> bool:
    return isinstance(obj, int) or isinstance(obj, float)
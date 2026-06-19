def select_target(boxes):
    if not boxes:
        return None
    return min(boxes, key=lambda b: b.h)

# def IOU(x1, x2, y1, y2, u1, u2, v1, v2):
#     inters = max(min(x2, u2) - max(x1, u1), 0) * max(min(y2, v2) - max(y1, v1), 0)
#     unions = (x2 - x1) * (y2 - y1) + (u2 - u1) * (v2 - v1) - inters
#     return inters / unions


def IOU(x1, x2, y1, y2, u1, u2, v1, v2):
    inters = max(min(x2, u2) - min(x1, u1), 0) * max(min(y2, v2) - min(y1, v1), 0)
    outers = (x2 - x1) * (y2 - y1) + (u2 - u1) * (v2 - v1) - inters
    return inters / outers
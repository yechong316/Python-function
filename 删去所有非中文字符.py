import re
def cleantxt(raw):
    fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+', re.UNICODE)
    return fil.sub(' ', raw)
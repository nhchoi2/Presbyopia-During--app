import pyshorteners

def generate_share_link(image_url):
    s = pyshorteners.Shortener()
    return s.tinyurl.short(image_url)

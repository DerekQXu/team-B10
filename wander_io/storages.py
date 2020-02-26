
"""
Modify django-storages for GCloud to set static, media folder in a bucket
"""
from storages.backends.gcloud import GoogleCloudStorage


class GoogleCloudMediaStorage(GoogleCloudStorage):
    """
    GoogleCloudStorage suitable for Django's Media files.
    """
    location = 'media'


class GoogleCloudStaticStorage(GoogleCloudStorage):
    """
    GoogleCloudStorage suitable for Django's Static files
    """
    location = 'static'
import datetime



class MediaFile:
    def __init__(self, filename: str, size: float, created_date: datetime, created_by: str):
        self.filename = filename
        self.size = size
        self.created_date = created_date
        self.created_by = created_by

    def __str__(self):
        return (f'Filename: {self.filename}\n'
                f'Size: {self.size}\n'
                f'Created Date: {self.created_date}\n'
                f'Created By: {self.created_by}\n\n')

class VideoFile:
    def __init__(self, filename: str, size: float, created_date: datetime, created_by: str):

media_file = MediaFile('123', 3.25, datetime.date(2000,1,1), 'NAME')
print(media_file)
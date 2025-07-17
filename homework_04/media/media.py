import datetime
from storage.storage import Storage
from debugpy.common.timestamp import current


class MediaFile:
    def __init__(self,
                 filename: str,
                 size: float,
                 created_date: datetime,
                 created_by: str,
                 file: bytes):
        self.filename = filename
        self.size = size
        self.created_date = created_date
        self.created_by = created_by
        self.file = file
        self.storage_type = Storage

    def __str__(self):
        return (f'Filename: {self.filename}\n'
                f'Size: {self.size}\n'
                f'Created Date: {self.created_date}\n'
                f'Created By: {self.created_by}\n\n')

    def save(self,path: str): # Сохраняет файл по указанному пути
        self.path = path

    def update(self, **kwargs): # Обновление параметров медиа
        pass

    def delete(self): # Удаляет данные о видео

    def collect_features(self): pass # Извлекает фичи из объекта

class VideoFile(MediaFile):
    def __init__(self,
                 filename: str,
                 size: float,
                 created_date: datetime,
                 created_by: str,
                 file: bytes,
                 video_format: str,
                 duration: int):
        super().__init__(filename,size,created_date,created_by,file)
        video_format = video_format.lower()
        duration = int(duration)
        last_stop_time = 0 # Время, на котором пользователь остановил просмотр

    def play(self): # Воспроизводит видео
        """
        current_play_time: int - параметр описывающий время, которое видео воспроизводилось в ms.
        while current_play_time < duration: - воспроизводим видео, пока оно не закончилось. Если last_stop_time > 0, то воспроизводим с этого времени
        else stop() - когда дошли до конца вызываем метод стоп
        ...
        """
        pass

    def stop(self): # Останавливает/Ставит на паузу воспроизведение видео
        """
        current_play_time: int - параметр описывающий время, которое видео воспроизводилось в ms.
        last_stop_time =  current_play_time if current_play_time < duration else 0 - сохраняем текущее время проигрывания если недосмотрели до конца

        """
    def maximize(self): pass # Открывает видео в полноэкранном разрешении

    def minimize(self): pass # Сворачивает видео

class ImageFile(MediaFile):
    def __init__(self,filename, size: float, created_date: datetime, created_by: str, file: bytes):
        super().__init__(filename,size,created_date,created_by)

    def open(self): pass # Открывает фото

    def close(self): pass # Закрывает фото



class AudioFile(MediaFile):
    def __init__(self,
                 filename: str,
                 size: float,
                 created_date: datetime,
                 created_by: str,
                 file: bytes,
                 audio_format: str,
                 duration: int):
        super().__init__(filename,size,created_date,created_by,file)
        audio_format = audio_format.lower()
        duration = int(duration)

    def play(self, last_stop_time):  # Воспроизводит видео
        """
        current_play_time: int - параметр описывающий время, которое аудио воспроизводилось в ms.
        while current_play_time < duration: - воспроизводим аудио, пока оно не закончилось
        else stop() - когда дошли до конца вызываем метод стоп
        ...
        """
        pass

    def stop(self):  # Останавливает/Ставит на паузу воспроизведение видео
        """
        current_play_time: int - параметр описывающий время, которое аудио воспроизводилось в ms.
        last_stop_time =  current_play_time if current_play_time < duration else 0 - сохраняем текущее время проигрывания если недослушали до конца

        """


from decorator import append

from media.media import *
from storage.storage import Storage

running = True
media_files: [MediaFile] = [] # храним набор медиафайлов в памяти
user_input_data = '' # описание действия пользователя

def load_media_file(user_input): # добавляем новый файл
    media_data = user_input.split(',')
    new_media_file = MediaFile(media_data)
    media_files.append(new_media_file)

def download_media_file(user_input):
    for media_file in media_files:
        if media_file.file_name == user_input:
            return media_file.file
        else:
            return 'Not Found'

def play_media(media_file: VideoFile | AudioFile):
    media_file.play()

def stop_media(media_file: VideoFile | AudioFile):
    media_file.stop()



while running:
    user_action = input() # считываем действие пользователя
    if user_action == 'load media':
        load_media_file(user_input_data)
    elif user_action == 'download media':
        download_media_file(user_input_data)
    elif user_action == 'play media':
        play_media(download_media_file(user_input_data))

'''
И так по всем функциям :)
'''

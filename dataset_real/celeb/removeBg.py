import requests


URL = 'https://api.remove.bg/v1.0/removebg'
IMG_PATH = '/Users/bernattorres/Downloads/img_align_celeba/'
API_KEY = '86HwkASrdD8ieHhfaDYt5TBa'

def get_photo_id(i: int):
    photo_id = ''
    to_fill = 6-len(str(i))
    for _ in range(1,to_fill+1):
        photo_id += '0'

    return photo_id + str(i)


for i in range(950,1000):
    response = requests.post(
        URL,
        files={'image_file': open(IMG_PATH+get_photo_id(i)+'.jpg', 'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': API_KEY},
    )
    if response.status_code == requests.codes.ok:
        with open('transparent_bg/'+get_photo_id(i)+'-bg.png', 'wb') as out:
            out.write(response.content)
    else:
        print("Error:", response.status_code, response.text)

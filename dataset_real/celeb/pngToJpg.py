from PIL import Image


def get_photo_id(i: int):
    photo_id = ''
    to_fill = 6-len(str(i))
    for _ in range(1,to_fill+1):
        photo_id += '0'

    return photo_id + str(i)


for i in range(1,1000):
    try:
        im = Image.open("transparent_bg/"+get_photo_id(i) + "-bg.png")
        non_transparent=Image.new('RGB',im.size,(255,255,255))
        non_transparent.paste(im,(0,0),im)
        non_transparent.save('white_bg/'+get_photo_id(i)+'.jpg')
    except:
        pass

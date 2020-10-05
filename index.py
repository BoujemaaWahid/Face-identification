from PIL import ImageFont
from recognation import LiveRecognation, Person, RecogOneFromImages, RecogMultiFromImages


t_live = LiveRecognation([
    Person("Name 1", "name1.jpg"),
    Person("Name 2", "name2.jpg")
])
t_live.process()

'''
t_one = RecogOneFromImages(Person("Name 1", "name1.jpg"), "faces", rect_color="#0275d8", rectangle_width=5)
t_one.process()


t_multi = RecogMultiFromImages([
    Person("Name 1", "name1.jpg"),
    Person("Name 2", "name2.jpg")
    ], "faces", rectangle_width=4, font=ImageFont.truetype("arial.ttf",20))

t_multi.process()
'''

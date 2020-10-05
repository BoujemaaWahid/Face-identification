import face_recognition as fr
import os
from PIL import Image, ImageDraw, ImageFont
import random
import abc
import cv2
import numpy as np
import sys
class Person():
    __name = None
    __base = None
    __code = None
    color = None
    def __init__(self, name, face_file_path):
        self.__name = name
        self.__base = fr.load_image_file(face_file_path)
        self.__code = fr.face_encodings(self.__base)[0]
        print(name, "is imported.")
    
    def getName(self):
        return self.__name
    def getImage(self):
        return self.__base
    def getCode(self):
        return self.__code


class Recognation(metaclass=abc.ABCMeta):
    __images_folder = None
    __font = None
    __rectangle_width = None
    __rect_color = None
    __text_color = None
    __precision = None
    def __init__(self, images_folder=None, font=None, rectangle_width=2, rect_color=(216,117,2), text_color=(255,255,255), precision=0.6):
        self.__images_folder=images_folder
        self.__font = font
        self.__rectangle_width = rectangle_width
        self.__rect_color = rect_color
        self.__text_color = text_color
        self.__precision = precision

    @abc.abstractmethod
    def process(self):
        pass
    
    def getImagesFolder(self):
        return self.__images_folder
    def getTextFont(self):
        return self.__font
    def getRectangleWidth(self):
        return self.__rectangle_width
    def getRectangleColor(self):
        return self.__rect_color
    def getTextColor(self):
        return self.__text_color
    def getPrecision(self):
        return self.__precision



class LiveRecognation(Recognation):
    __video_capture = None
    __persons = None
    __known_face_encodings = []
    __known_face_names = []
    def __init__(self, persons, font=cv2.FONT_HERSHEY_DUPLEX, rectangle_width=2, rect_color=(216, 117, 2), text_color=(255,255,255), precision=0.6):
        super().__init__(font=font, rectangle_width=rectangle_width, rect_color=rect_color, text_color=text_color, precision=precision)
        self.__video_capture = cv2.VideoCapture(0)
        self.__persons = persons
        for person in self.__persons:
            self.__known_face_encodings.append(person.getCode())
            self.__known_face_names.append(person.getName())
            
        
    
    def process(self):
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        while True:
            ret, frame = self.__video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            if process_this_frame:
                face_locations = fr.face_locations(rgb_small_frame)
                face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    matches = fr.compare_faces(self.__known_face_encodings, face_encoding, tolerance=self.getPrecision())
                    name = "Unknown"
                    face_distances = fr.face_distance(self.__known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.__known_face_names[best_match_index]
                        face_names.append(name)
            process_this_frame = not process_this_frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), self.getRectangleColor(), self.getRectangleWidth())
                cv2.rectangle(frame, (left, bottom - 20), (right, bottom), self.getRectangleColor(), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), self.getTextFont(), 0.5, self.getTextColor(), 1)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.__video_capture.release()
        cv2.destroyAllWindows()

class RecogOneFromImages(Recognation):
    __person = None
    def __init__(self, person, images_folder, font=None, rectangle_width=2, rect_color=(216, 117, 2), text_color=(255,255,255), precision=0.6):
        super().__init__(images_folder=images_folder, font=font, rectangle_width=rectangle_width, rect_color=rect_color, text_color=text_color, precision=precision)
        self.__person = person
      

    def __getData(self, path):
        img = fr.load_image_file( path )
        code = fr.face_encodings(img)
        result = fr.compare_faces(self.__person.getCode(), code, tolerance=self.getPrecision())
        if(True in result):
            dest = fr.face_distance(self.__person.getCode(), code)
            index = np.argmin(dest)
            return (result, index, img)
        return (result, -1, None)

    def process(self):
        for fileName in os.listdir(self.getImagesFolder()):
            path = f'{self.getImagesFolder()}/{fileName}'
            result, index, img = self.__getData(path)
            if(True in result):
                try:
                    location = fr.face_locations( img )[index]
                    pil = Image.fromarray( img )
                    draw = ImageDraw.Draw( pil )
                    top, right, bottom, left = location
                    rect = ( (left, top), (right, bottom) )
                    txt = (left, bottom+10)
                    draw.rectangle( rect, outline=self.getRectangleColor(), width=self.getRectangleWidth())
                    draw.text(txt, self.__person.getName(), fill=self.getTextColor(), font=self.getTextFont())
                    pil.show()
                except:
                    print('ERROR ', fileName, sys.exc_info()[0])
            else:
                print(fileName, 'No identical faces')


class RecogMultiFromImages(Recognation):
    __persons = None
    def __init__(self, persons, images_folder, font=None, rectangle_width=2, text_color=(255,255,255), precision=0.6):
        super().__init__(images_folder=images_folder, font=font, rectangle_width=rectangle_width, text_color=text_color, precision=precision)
        self.__persons = persons
        for person in self.__persons:
            person.color = self.__getColor()

    def __getColor(self):
        done = False
        lt = (255,255,255)
        while( done == False):
            _ok_ = False
            while(lt[0] == 255 and lt[1] == 255 and lt[2] == 255) and not _ok_:
                lt = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                for p in self.__persons:
                    if( p.color == None ):
                        _ok_ = True
                    elif( lt[0]-15 > p.color[0] or lt[1]-15 > p.color[1] or lt[2]-15 > p.color[2]):
                        _ok_ = True
                    else:
                        _ok_ = False
                done = True
        return lt

    def __getData(self, code, person):
        result = fr.compare_faces(person.getCode(), code, tolerance=self.getPrecision())
        dest = fr.face_distance(person.getCode(), code)
        index = np.argmin( dest )
        return (result, index)

    def process(self):
        for fileName in os.listdir(self.getImagesFolder()):
            path = f'{self.getImagesFolder()}/{fileName}'
            img = fr.load_image_file( path )
            code = fr.face_encodings(img)
            coords = []
            for person in self.__persons:
                result, index = self.__getData(code, person)
                if(True in result):
                    try:
                        location = fr.face_locations( img )[index]
                        coords.append( (location, person.getName(), person.color))
                    except:
                        print('ERROR LOCATION FACE', fileName)
            if( coords.__len__() > 0 ):
                pil = Image.fromarray( img )
                draw = ImageDraw.Draw( pil )
                for face in coords:
                    location = face[0]
                    name = face[1]
                    top, right, bottom, left = location
                    rect = ( (left, top), (right, bottom) )
                    rect2 = ( (left, bottom+20), (right, bottom) )
                    txt = (left, bottom)
                    draw.rectangle( rect, outline=face[2], width=self.getRectangleWidth())
                    draw.rectangle( rect2, outline=face[2], fill=face[2], width=self.getRectangleWidth())
                    draw.text(txt, name, fill=self.getTextColor(), font=self.getTextFont())
                pil.show()
            else:
                print(fileName, 'No identical faces')

import os
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, make_response, Response
import sys
import yaml
import os.path
import base64
from predict_digit import *
from matplotlib import image as mplimg
import cv2
import numpy as np
#from flask_json import FlaskJSON, JsonError, json_response, as_json
from flask.ext.responses import json_response


app = Flask(__name__)
#FlaskJSON(app)

number=list()
data=list()

def make_number():
	global number
	stng=''.join([str(i) for i in number])
	num=int(stng)
	number=list()
	return num

def apply_padding(img, border, val):
    h,w=img.shape
    cols=np.ones((h,border))*val
    tmp=np.concatenate([cols,img,cols],axis=1)
    rows=np.ones((border, w+2*border))*val
    res=np.concatenate([rows,tmp,rows])
    return res

def argsort(lst):
    return sorted(range(len(lst)), key=lst.__getitem__)

def extract_img(fname="digit_image.jpg"):
    im = cv2.imread(fname)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray=255-gray
    cv2.imwrite("grayscale.png",gray)
    image,contours,hierarchy= cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    idx=0
    print("No of digits: ", len(contours))
    gray2=cv2.imread("grayscale.png")
    gray2=cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)
    c=[ (0,0,255), #red
	(0,255,0), #green
	(255,0,0),#blue
	(255,255,255), #white
	(128,128,128), #gray
	(0,0,0)#black
      ]
    total=len(contours)
    pnt_idxs=argsort([(x,y) for cnt in contours for x,y,w,h in [cv2.boundingRect(cnt)]])
    lst=list()
    for index,ix in enumerate(pnt_idxs):
        x,y,w,h = cv2.boundingRect(contours[ix])
        lst.append((x,y,w,h))
        idx += 1
        #x,y,w,h = cv2.boundingRect(cnt)
        roi=gray2[y:y+h,x:x+w]
        #cv2.imwrite("tmp.jpg", roi)
        #cv2.copyMakeBorder(roi, new_img, borderz, borderz, borderz, borderz, cv2.BORDER_CONSTANT, 255)
        new_img=apply_padding(roi, 20, 0)
        new_img=255-new_img
        cv2.imwrite(str(idx)+".jpg", new_img)
        #cv2.rectangle(im,(x,y),(x+20,y+20),c[index],2)
    #cv2.imwrite("annotated.jpg", im)
    #print("Lst :",lst)
    return lst,total

exp=None
@app.route("/", methods=["GET","POST"])
def root():
	return render_template("root.html")
		
def xtract_number():
    indices,count=extract_img()
    #print("Count: ",count)
    ans=list()
    for i in range(1,count+1):
        ans.append(predict_drawn_img(str(i)+".jpg")[0])
    number=int(''.join([str(i) for i in ans]))
    #print("Ans: ", ans)
    #print(indices)
    return number
   
def is_number(dat):
	try:
		int(dat)
	except:
		return False
	return True

@app.route("/operator", methods=["GET","POST"])
def operators():
	global data
	ans=0.0
	op=request.json["operator"]
	if op=="reset":
		data=list()
		return json_response({"num":"Draw the number above", "res":0.0}, status_code=200)
	elif op=="backspace":
		if len(data):
			data=data[:-1]
		exp=' '.join([str(i) for i in data])
		return json_response({"num":exp, "res":ans}, status_code=200)
	if data and is_number(data[-1]):
		if op=='=':
			exp=' '.join([str(i) for i in data+['=']])
			ans=solve()
			return json_response({"num":exp, "res":ans}, status_code=200)
		elif op in ['+', '-', '*','/']:
			data.append(op)
			exp=' '.join([str(i) for i in data])
			return json_response({"num":exp, "res":ans}, status_code=200)
	with open("digit_image.jpg",'wb')as f:
		f.write(base64.b64decode(request.json["image"].split(',')[1]))
	number=xtract_number()
	data.append(number)
	data.append(op)
	exp=' '.join([str(i) for i in data])
	if op=='=':
		data=data[:-1]
		ans=solve()
	return json_response({"num":exp, "res":ans}, status_code=200)
		
def solve():
	global data
	print(data)
	total=data[0]
	for index in range(1,len(data),2):
		op=data[index]
		if op=='+':
			total+=data[index+1]
		elif op=='-':
			total-=data[index+1]
		elif op=='*':
			total*=data[index+1]
		elif op=='/':
			total/=data[index+1]
	data=list()
	print("Total= ", total)
	return total
	
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=False, host='0.0.0.0', port=31456)

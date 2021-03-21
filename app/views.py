import secrets
import os
from PIL import Image
from flask import render_template, request, redirect, jsonify, make_response, flash, url_for, Blueprint
from app import app, db, bcrypt, mail
from app.forms import (LoginForm, RegistrationForm, UpdateAccountForm, ChildForm, 
						RequestResetForm, ResetPasswordForm, UploadForm, TestDateForm, ProgressForm)
from app.models import User
from datetime import datetime
from flask_login import login_user, current_user, logout_user, login_required
from is_safe_url import is_safe_url
from flask_mail import Message
from werkzeug.utils import secure_filename
from pyzbar.pyzbar import decode
import cv2 as cv
import numpy as np
import pymongo
#from keras.models import model_from_json
#from keras.preprocessing import image
import json
#from keras.models import load_model
import requests
from user_agents import parse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

parent_dir = "F:/Google Drive/RISU/System development/Matrix"
date = "2020-01-01"
#Restore trained network weight
#model = model_from_json(open('F:/Google Drive/User_Backup/model/cnn_model4.json').read())
#model.load_weights('F:/Google Drive/User_Backup/model/cnn_model_weights4.hdf5')
#model = load_model("F:/Google Drive/User_Backup/model/trained_20201016.h5")
#model_t = load_model("cnn/models/tens_digit_20201013.h5")
#model_oo = load_model("cnn/models/trained_20201014.h5")
#model_o = load_model("cnn/models/ones_digit_20201013.h5")
#model = load_model("cnn/models/25x44_2digit_20201018.h5")
#model_single = load_model("cnn/models/25x44_1digit_20201018.h5")
#model_s = load_model("cnn/models/single_digit_25x25_32_64_20210114.h5")
path = ""; secure_files = []; form_id =""; browser=""; mobile = False; pid = 0; cid=0; children =""

@app.route("/")
def index():
	headers = requests.utils.default_headers()
	ua = parse(headers['User-Agent'])
	print (ua.os.family)
	if current_user.is_authenticated:
		children = find_child(current_user.id)
		#print("Children", children)
		child_num = len(children)
		if child_num == 0:
			return render_template("public/add_child.html", form = form)
	if ua.os.family == 'iOS' or 'Android':
		mobile = True
	else: 
		mobile = False
	return render_template("public/index.html", mobile = ua.os.family)

@app.route("/home")
def home():
	headers = requests.utils.default_headers()
	ua = parse(headers['User-Agent'])
	if current_user.is_authenticated:
		children = find_child(current_user.id)
		#print("Children", children)
		child_num = len(children)
		if child_num == 0:
			return render_template("public/add_child.html", form = form)
	if ua.os.family == 'iOS' or 'Android':
		mobile = True
	else:
		mobile = False
	return render_template("public/index.html", mobile = mobile)

@app.route("/about")
def about():
	return render_template("public/about.html")

@app.route("/login", methods =["GET", "POST"])
def login():
	global pid
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = LoginForm()
	if form.validate_on_submit():
		user = User.query.filter_by(email = form.email.data).first()
		pid = user.id
		if user and bcrypt.check_password_hash(user.password, form.password.data):
			login_user(user, remember=form.remember.data)
			#next_page = request.args.get('next')
			return redirect(url_for('home'))
			#if not is_safe_url(next_page):
			#		return flask.abort(400)
			#return redirect(next_page) if next_page else redirect(url_for('home'))
		else:
			flash('Login Unsuccessful. Please check email and password', 'danger')
	# return render_template(url_for('login'), title="Login", form = form)
	return render_template("public/login.html", title="Login", form = form)

def ins_parent_id (id, zipcode):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["parent_db"]
	mydict = {"id":id, "zipcode":zipcode, "child":[]}
	x = mycol.insert_one(mydict)
	return x.inserted_id

def read_parent_id ():
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["parent_id"]
	mydoc = mycol.find_one()
	return mydoc["parent_id"]

@app.route("/register", methods =["GET", "POST"])
def register():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = RegistrationForm()
	if form.validate_on_submit():
		hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
		user = User(username = form.username.data, email = form.email.data, password = hashed_password, zipcode = form.zipcode.data)
		db.session.add(user)
		db.session.commit()
		# Insert row to Mongo DB parent_id 
		#inserted_id = ins_parent_id (user.id, form.zipcode.data)
		#path = os.path.join(parent_dir, 'static/img/upload/', form.username.data)
		#try:
		#	os.makedirs(path, exist_ok = True)
		#	print("Directry '%s' created successfully" %form.username.data)
		#except OSError as error:
		#	print("Directory 's%' can not be created")
		flash('Your account has been created. You are able to login.', 'success')
		return redirect(url_for("login"))
	return render_template('public/registration.html', title="Register", form = form)

@app.route("/logout")
def logout():
	logout_user()
	return redirect(url_for('home'))

def save_picture(form_picture):
	random_hex = secrets.token_hex(8)
	_, f_ext = os.path.splitext(form_picture.filename)
	picture_fn = random_hex + f_ext
	picture_path = os.path.join(app.root_path, 'static/img/', picture_fn)
	output_size = (125, 125)
	i = Image.open(form_picture)
	i.thumbnail(output_size)
	i.save(picture_path)
	return picture_fn

def delete_picture(current_picture):
	picture_path = os.path.join(app.root_path, 'static/img/', current_picture)
	r = os.remove(picture_path)
	return

def find_child(pid):
	children = []
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["child_db"]
	myquery = {"pid":pid}
	mydoc = mycol.find(myquery)
	for child in mydoc:
		children.append(child)
	return children

@app.route("/account", methods =["GET", "POST"])
@login_required
def account():
	global pid
	form = UpdateAccountForm()
	if form.validate_on_submit():
		if form.picture.data:
			picture_file = save_picture(form.picture.data)
			delete_picture(current_user.image_file)
			current_user.image_file = picture_file
		current_user.username = form.username.data
		current_user.email = form.email.data
		current_user.zipcode = form.zipcode.data
		db.session.commit()
		flash('Your account has been updated.', 'success')
		return redirect(url_for('account'))
	elif request.method == 'GET':
		form.username.data = current_user.username
		form.email.data = current_user.email
		form.zipcode.data = current_user.zipcode
		pid = current_user.id
		print("Current User ID: ", current_user.id)
	children = find_child(current_user.id)
	image_file = url_for('static', filename='img/'+current_user.image_file)
	return render_template("public/account.html", title="Account", image_file=image_file, children = children, form = form,)

@app.route("/add_child", methods =["GET", "POST"])
@login_required
def add_child():
	form = ChildForm()
	if form.validate_on_submit():
		print ("Form submitted", current_user.id)
		myclient = pymongo.MongoClient("mongodb://localhost:27017/")
		mydb = myclient["mydatabase"]
		mycol = mydb["last_child_id"]
		mydoc = mycol.find_one()
		print(mydoc)
		cid = int(mydoc["last_child_id"])+1
		myquery = {"last_child_id":mydoc["last_child_id"]}
		newvalue = {"$set":{"last_child_id":cid, "datetime":datetime.now()}}
		#print ("Cid: ", cid, "Newvalue: ", newvalue)
		mycol.update_one(myquery, newvalue)
		mycol = mydb["child_db"]
		mydict = {"cid":cid, "pid":current_user.id, "cfirst":form.cfirst.data, "clast":form.clast.data, "cgender":form.cgender.data,
					"cgrade":form.cgrade.data}
					#, "email":form.email.data, "temail1":form.temail1.data}
					#"temail2":form.temail2.data, "tuemail1":form.tuemail1.data, "tuemail2":form.tuemail2.data, "oemail1":form.oemail1.data,
					#"oemail2":form.oemail2.data, "oemail3":form.oemail3.data, "oemail4":form.oemail4.data}
		print (mydict)
		mycol.insert_one(mydict)
		flash('Child data has been added.', 'success')
		return redirect(url_for('home'))
	elif request.method == 'GET':
		return render_template("public/add_child.html", form = form)

@app.route("/profile/<username>")
def profile(username):
	users = {
	"mitsuhiko": {
		"name": "Armin Ronacher",
		"bio": "Creatof of the Flask framework",
		"twitter_handle": "@mitsuhiko"
	},
	"gvanrossum": {
		"name": "Guido Van Rossum",
		"bio": "Creator of the Python programming language",
		"twitter_handle": "@gvanrossum"
	},
	"elonmusk": {
		"name": "Elon Musk",
		"bio": "technology entrepreneur, investor, and engineer",
		"twitter_handle": "@elonmusk"
	}
	}	
	user = None
	if username in users:
		user = users[username]
	return render_template("public/profile.html", username = username, user = user)

def send_reset_email(user):
	token = user.get_reset_token()
	msg = Message('Password Reset Request',
				  sender='support@us.myrisu.com',
				  recipients=[user.email])
	msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}
If you did not make this request then simply ignore this email and no changes will be made.
'''
	mail.send(msg)

@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = RequestResetForm()
	if form.validate_on_submit():
		user = User.query.filter_by(email=form.email.data).first()
		send_reset_email(user)
		flash('An email has been sent with instructions to reset your password.', 'info')
		return redirect(url_for('login'))
	return render_template('public/reset_request.html', title='Reset Password', form=form)


@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	user = User.verify_reset_token(token)
	if user is None:
		flash('That is an invalid or expired token', 'warning')
		return redirect(url_for('reset_request'))
	form = ResetPasswordForm()
	if form.validate_on_submit():
		hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
		user.password = hashed_password
		db.session.commit()
		flash('Your password has been updated! You are now able to log in', 'success')
		return redirect(url_for('login'))
	return render_template('public/reset_token.html', title='Reset Password', form=form)
	
def allowed_file(filename):
	ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/testdate", methods=['GET', 'POST'])
@login_required
def testdate():
	global date, path, cid
	if current_user.is_authenticated == False:
		flash('Login required. Please login before enter the test date', 'danger')
		return redirect(url_for('login'))
	form = TestDateForm()
	if form.validate_on_submit():
		#print ("Files: ", form.testdate.data)
		#print("Currnet_user name", current_user.username)
		date = str(form.testdate.data)
		#cid = str(form.child.data)
		msg = "Test Date is " + date
		flash(msg)
		cid = request.form.get('child')
		if cid == "":
			children = find_child(current_user.id)
			#print("Children", children)
			child_num = len(children)
			msg = "Please select a child who worked on the worksheet."
			return render_template("public/testdate.html", title="Test Date", form = form, children = children, child_num = child_num, msg = msg)
		#path = os.path.join(parent_dir, "uploads", current_user.username, date)
		path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date)
		#print("Path:", path)
		#print(os.path.exists(path))
		if os.path.exists(path) == False:
			try:
				os.makedirs(path, exist_ok = True)
				print("Directry '%s' created successfully" %path)
			except OSError as error:
				print("Directory 's%' can not be created")    
		return redirect(url_for('upload_files'))
	elif request.method == 'GET':
		children = find_child(current_user.id)
		#print("Children", children)
		child_num = len(children)
		print("children", children, " child_number", child_num)
		msg = ""
		if child_num == 0:
			return render_template("public/add_child.html", form = form)
		else:
			return render_template("public/testdate.html", title="Test Date", form = form, children = children, child_num = child_num, msg = msg)
	else:
		flash('Else')
		return render_template("public/testdate.html", title="Test Date", form = form)

def read_codes(frame):
	# Load the predefined dictionary#Load the dictionary that was used to generate the markers.

	dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

	# Initialize the detector parameters using default values

	parameters =  cv.aruco.DetectorParameters_create()

	# Detect the markers in the image

	#frame = cv.imread("Scan.jpg")
	markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
	#print(markerIds)
	#print(markerCorners)
	#form_id=decode(Image.open('horn4.png'))[0][0].decode("utf-8")
	form_id=decode(frame)[0][0].decode("utf-8")
	#print(form_id)
	return (form_id, markerCorners, markerIds)

def center_aruco(markerCorners, markerIds, pts_src):
	#print(pts_src)
	# Calculate the center of each corner markers
	i = 0
	for each_corner in np.array(markerCorners):
		# print (each_corner)
		# each_corner contains array of four corners coordinates of each aruco marker
		c = np.array(each_corner[0])
		# take out a braket
		x,y = 0,0
		# calculate center point of four corners
		for cor in np.array(c):
			x += cor[0]
			y += cor[1]
			#print("X: ", x, " Y: ",y)
		# xy[markerIds[i][0]] = [x,y]
		# pts_src[markerIds[i][0] - 1] = [x/4,y/4]
		pts_src[i] = [x/4,y/4]
		i += 1
	#print(pts_src)
	return (pts_src)

def read_formdb(form_id):
	global form_name, form_time, q_row, q_col, op
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["forms"]
	myquery = { "form_id": form_id }
	doc = mycol.find(myquery)
	aruco_location = doc[0]["aruco"]
	areas = doc[0]["areas"]
	layout = doc[0]["layout"]
	form_name = doc[0]["form_name"]
	form_time = [doc[0]["time_m"], doc[0]["time_s"]]
	q_row = doc[0]["areas"][0]["question_row"]
	q_col = doc[0]["areas"][0]["question_col"]
	op = doc[0]["areas"][0]["op"]
	#print("q_row: ",q_row, " q_col: ", q_col)
	return(aruco_location, areas, layout, form_name, form_time)

def set_marker_location(markerIds, aruco_location, pts_dst):
	i = 0; # pts_dst=[None]*len(np.array(markerIds))
	for cor in np.array(markerIds):
		#print(cor)
		#print(aruco_location[str(cor[0])])
		pts_dst[i] = aruco_location[str(cor[0])]
		i += 1
	return (pts_dst)

def sortf(e):
	return e[1]

def eval_logic(img, answer):
	x = np.expand_dims(img, axis=0)
	if answer > 9:
		o_proba = model.predict(x)
		r = 100
	else:
		o_proba = model_single.predict(x)
		r = 10
	o_result = o_proba.tolist()
	pred = int(np.argmax(o_result, axis=-1))
	ave_predict = ans_ave_Pred = o_result[0][1]
	conf_level = 0
	if answer != pred:
		s = o_result
		ns = []
		#print("answer: ",answer,"Number: ",k)
		for i in range(0,r):
			ns.append([i,round(s[0][i]*100,1)])
		ns.sort(reverse=True, key=sortf)
		flag = True
		for l in range(0,5):
			if answer == ns[l][0]:
				flag = False
				print("answer: ",answer," Prediction: ",ns[0], " hit at ",l, ns[l])
				pred = ns[l][0]
				ave_predict = ans_ave_Pred = ns[l][1]
				conf_level = l
				break
		if flag == True:
			print("answer: ",answer, ns[0], ns[1], ns[2], ns[3])
			scan_result=[]
			t1 = False
			sum_cells = 0
			if 9 < answer:
				j1 = int(answer//10)
				j2 = int(answer%10)
				#print ("j1: ", j1, " j2: ", j2)
			for i in range(0,20,3):
				#print(x.shape)
				eex = img[:,i:i+25]
				#print(i, eex.shape)
				ex = np.expand_dims(eex, axis=0)
				s_proba = model_s.predict(ex)
				s_result = s_proba.tolist()
				ind = int(np.argmax(s_result,axis=-1))
				sum_cells += ind
				#print(ind, s_result)
				if 9 < answer:
					if t1 == True and j2 == ind:
						pred = answer
						flag = False
					if t1 == False and j1 == ind:
						t1 = True
				else:
					if ind == answer:
						flag = False
				scan_result.append(ind)
				# scan_result.append([ind, s_result[0][ind]])
			if flag:
				if sum_cells > 60:
					print("answer: ",answer, " Prediction: ",ns[0], "Scan: ", scan_result, "Blank")
					conf_level = -2
				else:
					print("answer: ",answer, " Prediction: ",ns[0], "Scan: ", scan_result)
					conf_level = -1
				ave_predict = ans_ave_Pred = 0
			else:
				print("answer: ",answer, " Prediction: ",ns[0], "Scan: ", scan_result, "hit")
				conf_level = 5
				ave_predict = ans_ave_Pred = 0
	else:
		ave_predict = ans_ave_Pred = o_result[0][0]
		conf_level = 0
	#print("answer: ", answer, "Missed: ", missed, " Within4th: ", within4th, " Scanned: ", scanned)
	return (pred, ave_predict, ans_ave_Pred, conf_level)

def evaluate_img(path, filename, answer):

	#newsize = (45, 25)
	#img = img.resize(newsize)
	#filename = 'F:/Google Drive/User_Backup/test_data/20_1.png'
	img = image.load_img(path + "/" + filename, color_mode ='grayscale', target_size=(25, 44))

	x = image.img_to_array(img)
	x /= 255
	#x = np.expand_dims(x, axis=0)
	#y_proba = model.predict(x)
	#result = y_proba.tolist()
	test_new_logic = True
	if test_new_logic == True:
		num, ave_predict, ans_ave_pred, conf_level = eval_logic(x, answer)
	else:
		if answer//10 == 0:
			oo_proba = model_oo.predict(x)
			oo_result = oo_proba.tolist()
			ans_ave_pred = oo_result[0][answer]
			ave_predict =oo_result[0][model_o.predict_classes(x)[0]]
			num = [model_oo.predict_classes(x)[0]][0]
		else:
			o_proba = model_o.predict(x)
			t_proba = model_t.predict(x)
			o_result = o_proba.tolist()
			t_result = t_proba.tolist()
			t = [model_t.predict_classes(x)[0]][0]
			o = [model_o.predict_classes(x)[0]][0]
			num = int(t)*10 + int(o)
	#num = int(t)*10 + int(o)
		#ave_predict = (o_result[0][model_o.predict_classes(x)[0]] + t_result[0][model_t.predict_classes(x)[0]])/2
		#ans_ave_pred = (o_result[0][int(answer%10)]+t_result[0][int(answer//10)])/2
			ave_predict = o_result[0][model_o.predict_classes(x)[0]]
			ans_ave_pred = t_result[0][int(answer//10)]
	# Evaluation result
	#print(model.predict_classes(x))
	#for i in range(100):
		#print(i, result[0][i])
	#return (int(model.predict_classes(x)[0]), result[0][model.predict_classes(x)[0]], result[0][answer])
	return (num, ave_predict, ans_ave_pred, conf_level)
	#return (num, .5, .5)

def write_evaluation (user, date, image, form_id, evaluation):
	global cid
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["evaluation"]
	result_id = user + str(cid) + date + image[:-4]
	#print(result_id, "Evaluation: ", evaluation)
	key = {"result_id": result_id}
	mydict = {
		"result_id": result_id,
		"pid": current_user.id,
		"user": user,
		"child": cid,
		"date": date,
		"image": image,
		"form_id": form_id,
		"reviewed": False,
		"mistakes": 0,
		"eval": evaluation,
		"updated":datetime.now()}
	mycol.update(key, mydict, upsert = True)
	return

def evaluate_answer(areas, im_out, img, form_id):
	global q_row, q_col, op
	grp = 0
	eval_result=[]; area_result = []; group_result = {} 
	for area in areas:
		row_h = (area["lower_xy"][1] - area["upper_xy"][1])/area["row"]
		col_w = (area["lower_xy"][0] - area["upper_xy"][0])/area["col"]
		group_result["group"] = area["group"]; group_result["row_h"] = row_h; group_result["col_w"] = col_w
		group_result["row"] = area["row"]; group_result["col"] = area["col"]; 
		cell_eval = {}
		#print(upper_xy, lower_xy, row, col)
		for row in range(area["row"]):
			for col in range(area["col"]):
				upperleft_y = int(area["upper_xy"][1]+row_h*row)
				upperleft_x = int(area["upper_xy"][0]+col_w*col)
				lowerright_y = int(area["upper_xy"][1]+row_h*(row+1))
				lowerright_x = int(area["upper_xy"][0]+col_w*(col+1))
				#print(upperleft_y, lowerright_y, upperleft_x,lowerright_x )
				cut = im_out[upperleft_y:lowerright_y, upperleft_x:lowerright_x]
				#cut = cv.bitwise_not(cut)
				filename = str(grp)+"-"+str(row)+"-"+str(col)+".png"
				#path = os.path.join(parent_dir, "uploads", current_user.username, date, img[:-4])
				path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date, img[:-4])
				if os.path.exists(path) == False:
					try:
						os.makedirs(path, exist_ok = True)
						print("Directry '%s' created successfully" %path)
					except OSError as error:
						print("Directory 's%' can not be created")    
				#path = os.path.join(path, filename)    
				cv.imwrite(path + "/" + filename, cut)
				#print(row, col, area["answer"][row][col])
				# Evaluate each cell of image with Keras
				#predict, prob_pred, prob_ans = evaluate_img(path, filename, area["answer"][row][col])
				#pred = predict.item()
				#pred, prob_pred, prob_ans, conf_level = evaluate_img(path, filename, area["answer"][row][col])
				#print(pred, prob_pred, prob_ans)
				#pred = predict.item()
				# Prepare for constructing JSON data
				cell_eval["row"] = row; cell_eval["col"] = col
				pred = ''
				#cell_eval["upper_x"] = area["upper_xy"][0]; cell_eval["upper_y"] = area["upper_xy"][1]
				cell_eval["upper_x"] = upperleft_x; cell_eval["upper_y"] = upperleft_y; cell_eval["color"]="white"
				cell_eval["predict"] = pred; cell_eval["prob_pred"] = 0; cell_eval["correct"]=False
				cell_eval["prob_ans"] = ''; cell_eval["conf_level"] = 0; 
				cell_eval["answer"] = area["answer"][row][col]; cell_eval["miss_recog"]= False
				cell_eval["match"] = True
				cell_eval["op"]= str(q_col[row]) + " " + op + " " + str(q_row[col])
				#print("q_row: ",q_row, " q_col: ", q_col)
				#if pred == area["answer"][row][col]:
				#    cell_eval["match"] = True
				#else:
				#    cell_eval["match"] = False
				#    #print(pred, area["answer"][row][col])
				#    mpath = img[:-4]+"/missed"
				#    if os.path.exists(mpath) == False:
				#		try:
				#		    os.makedirs(mpath, exist_ok = True)
				#		    print("Directry '%s' created successfully" %mpath)
				#		except OSError as error:
				#		    print("Directory 's%' can not be created")   
				#    if area["answer"][row][col] >= 10:
				#		filename = str(row*area["row"] + col) + "-" + str(area["answer"][row][col]) +".jpg"
				#    else:
				#		filename = str(row*area["row"] + col) + "-" + "0" + str(area["answer"][row][col]) +".jpg"
				#    cv.imwrite(mpath + "/" + filename, cut)
				#print(grp, row, col, pred, prob_pred, prob_ans, area["answer"][row][col])
				area_result.append(cell_eval); cell_eval = {}; 
		grp += 1
		#eval_result.append(area_result)
		group_result["eval_cells"] = area_result; area_result = [] 
		eval_result.append(group_result); group_result = {}
	#print (eval_result)
	return (eval_result)

def eval_image(img):
	# read image as gray scale
	#print("img in eval_image ", img)
	frame = cv.imread(img,0)
	# Convert the image to binary iwth cv2.Thresh_OTSU.
	#ret2, frame = cv.threshold(frame, 0, 255, cv.THRESH_OTSU)
	path = path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date)
	#x=cv.imwrite(os.path.join(path, 'threshold.png'),frame)
	form_id, markerCorners, markerIds = read_codes(frame)
	#print(form_id, markerCorners, markerIds)
	aruco_location, areas, layout, form_name, form_time = read_formdb(form_id)
	#print(aruco_location, areas, layout)
	#evaluation = {"user": current_user.username, "image": img, "date": date}
	#print(aruco_location)
	if len(markerIds) > 3:
		pts_src, pts_dst = np.zeros((len(markerIds),2)), np.zeros((len(markerIds),2)) 
		#pts_src = np.array([[0,0],[0,0],[0,0],[0,0]])
		# calculate center of each aruco marker
		#print(pts_src)
		pts_src = center_aruco(markerCorners, markerIds, pts_src)
		#print(pts_src)
		# prepare arrays for homography
		pts_dst = set_marker_location(markerIds, aruco_location, pts_dst)
		# Calculate Homography
		#print("pts_dst: ", pts_dst, "pts_src: ", pts_src)
		h, status = cv.findHomography(pts_src, pts_dst)
		#print(status)
		# Warp source image to destination based on homography
		# final image size (744, 531) needs to be set form by form 
		if layout == "L":
			size = (960, 720)
		else:
			size = (720, 960)
		im_out = cv.warpPerspective(frame, h, size)
		#path = os.path.join(parent_dir, "uploads", current_user.username, date, img[:-4])
		path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date, img[:-4])
		if os.path.exists(path) == False:
			try:
				os.makedirs(path, exist_ok = True)
				print("Directry '%s' created successfully" %path)
			except OSError as error:
				print("Directory 's%' can not be created") 
		path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date, img[:-4], "adjust.jpg")
		#print(path)
		x = cv.imwrite(path, im_out)
		#print(x)
		eval_result = evaluate_answer(areas, im_out, img, form_id)
		#print(eval_result)
		#evaluation["evaluation"] = eval_result
	else:
		print("Error: program is not able to detect four or more corner marks")
		return render_template("public/retake.html")
	#return (evaluation, path) 
	return (eval_result, path, form_id) 

@app.route("/upload_files", methods=['GET', 'POST'])
@login_required
def upload_files():
	if current_user.is_authenticated == False:
		flash('Login required. Please login before uploading files', 'danger')
		return redirect(url_for('login'))
	form = UploadForm()
	global files, path, secure_files, date, form_id
	secure_files = []
	if form.validate_on_submit():
		files = form.file.data
		if date == "2020-01-01":
			return redirect(url_for('testdate'))
		for file in files:
			if allowed_file(file.filename) == False:
				msg = "Wrong file format: " + file.filename
				flash(msg) 
			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)
				#path = os.path.join(parent_dir, "uploads", current_user.username, date)
				path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date)
				file.save(os.path.join(path, filename))
				secure_files.append(filename)
				#msg = "Uploaded: " + filename
				#flash(msg)
		#print("Secure_Files: ", secure_files)
		for file in secure_files:
			print("Upload_files' file", file)
			#path = os.path.join(parent_dir, "uploads", current_user.username, date)
			path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date)
			eval_result, path, form_id = eval_image(os.path.join(path, file))
			write_evaluation (current_user.username, date, file, form_id, eval_result)
			children = find_child(current_user.id)
			child_num = len(children)
		#    eval_result = eval_image(img)
		print("scoring")
		return render_template("public/scoring.html", title="Scoring", form = form, files = secure_files, children = children, child_num = child_num)
	elif request.method == 'GET':
		return render_template("public/upload_files.html", title="Upload", form = form)
	else:
		flash('Else')
		return render_template("public/upload_files.html", title="Upload", form = form)

@app.route("/scan", methods=['GET','POST'])
@login_required
def scan():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			#path = os.path.join(parent_dir, "uploads", current_user.username, date)
			path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date)
			file.save(os.path.join(path, filename))
			eval_result, path, form_id = eval_image(os.path.join(path, filename))
			write_evaluation (current_user.username, date, file, form_id, eval_result)
			children = find_child(current_user.id)
			return redirect(url_for('scoring', filename=filename, child = children))
	return render_template('public/scan.html')

def read_result (user, date, image):
	global cid
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["evaluation"]
	result_id = user + str(cid) + date + image[:-4]
	myquery = {"result_id": result_id}
	mydoc = mycol.find(myquery,{"_id":0, "result_id": 1, "user": 1, "date": 1, "image": 1, "form_id": 1, "eval": 1,"reviewed": 1, "mistakes": 1})
	evaluation = {}
	for x in mydoc:
		evaluation = x
		continue
	return evaluation

def update_result (user, date, image, form_id, data):
	global cid
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["evaluation"]
	result_id = user + str(cid) + date + image[:-4]
	myquery = {"result_id": result_id}
	data["pid"] = current_user.id
	data["timestamp"] = datetime.now()
	newvalues = { "$set" : data}
	mycol.update_one(myquery, newvalues)
	return 

def write_review_log (user, date, image, form_id, data):
	global cid
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["review_log"]
	result_id = user + str(cid) + date + image[:-4]
	key = {"result_id": result_id}
	data["cid"] = cid
	data["pid"] = current_user.id
	data["timestamp"] = datetime.now()
	mydict = data
	mycol.update(key, mydict, upsert = True)
	#print("Write review log", data)
	return

def return_page (secure_files, i):
	evaluation = read_result(current_user.username, date, secure_files[i] )
	#print(evaluation)
	#path = os.path.join(parent_dir, "uploads", current_user.username, date)
	path = os.path.join('static\\img\\upload\\', current_user.username, date, secure_files[i][:-4])
	return_json = {
			'path': path,
			'total_files': len(secure_files),
			'number_of_file':i,
			'file': secure_files[i],
			'eval': evaluation,
			'type': "eval"
		}
	#with open ("C:/Users/Hiroji/Documents/json.txt","w") as json_file:
	#    json.dump(return_json, json_file)
	res = make_response(jsonify(return_json), 200) 
	return res

def return_endpage (secure_files):
	res = []
	for f in secure_files:  
		evaluation = read_result(current_user.username, date, f )
		res.append({"image": evaluation["image"],"mistakes": evaluation["mistakes"] })
	return_json = {
			'path': "",
			'total_files': len(secure_files),
			'number_of_file':i,
			'file': secure_files[i],
			'eval': res,
			'type': "end"
		}
	#with open ("C:/Users/Hiroji/Documents/json.txt","w") as json_file:
	#    json.dump(return_json, json_file)
	res = make_response(jsonify(return_json), 200) 
	return res


@app.route("/api", methods=['POST'])
def api():
	global secure_files, path, i, cid, child_name, form_id

	req = request.get_json()  
	#print("Request: ", req["action"], "secure_files: ", secure_files)
	if req["action"] == "JSON":
		i = 0
		#print(secure_files)
		res = return_page (secure_files, i)
	else:
		#i = 0
		#print(req)
		#cid = req["data"]["child"]
		#child_name = req["data"]["child_name"]
		time_min = req["data"]["minutes"]
		time_sec = req["data"]["seconds"]
		#print("min: ", time_min, " sec: ", time_sec)
		form_id = req["data"]["form_id"]
		write_review_log(current_user.username, date, secure_files[i], req["data"]["form_id"], req["data"])
		update_result(current_user.username, date, secure_files[i], req["data"]["form_id"], req["data"])
		if req["action"] == "End":
			#print("API action End")
			i = 0
			res = return_endpage (secure_files)
		elif req["action"] == "+" and i <len(secure_files):
			i += 1
			res = return_page (secure_files, i)
		elif req["action"] == "-" and i > 0:
			i -= 1
			res = return_page (secure_files, i)
	return res		

## @app.route("/test", methods=['GET', 'POST'])
## def test():
##    if request.method == "POST":
##		firstname = request.form['firstname']
##		lastname = request.form['lastname']

def find_child_name(pid, cid):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["child_db"]
	pid = int(pid)
	cid = int(cid)
	myquery = {"pid": pid, "cid": cid}
	#print ("pid: ", pid, "cid: ", cid)
	mydoc = mycol.find_one(myquery)
	#print(mydoc)
	child_name = mydoc["cfirst"] + " " + mydoc["clast"]
	return child_name

def sortFunc(e):
	return e[0]

def graph_data_gen(pid, cid, form_id):
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	mydb = myclient["mydatabase"]
	mycol = mydb["evaluation"]
	#result_id = user + str(cid) + date + image[:-4]
	#myquery = {"result_id": result_id}
	myquery = {"pid": int(pid), "cid": cid, "form_id": form_id}
	#mydoc = mycol.find(myquery,{"_id":0, "result_id": 0, "user": 0, "date": 1, "image": 0, "form_id": 0, "eval": 0,"reviewed": 1, "mistakes": 1, "minutes": 1, "seconds": 1})
	mydoc = mycol.find(myquery)
	plot_data = []
	for x in mydoc:
		plot_data.append([x["date"], x["mistakes"], x["minutes"],x["seconds"]])
	plot_data.sort(key = sortFunc)
	return plot_data

@app.route("/progress", methods=['GET','POST'])
@login_required
def progress():
	global cid, child_name, form_id, date, form_name, form_time
	form = ProgressForm()
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			#path = os.path.join(parent_dir, "uploads", current_user.username, date)
			path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date)
			file.save(os.path.join(path, filename))
			eval_result, path, form_id = eval_image(os.path.join(path, filename))
			write_evaluation (current_user.username, date, file, form_id, eval_result)
			children = find_child(current_user.id)
			return redirect(url_for('scoring', filename=filename, child = children))
		return render_template('public/scan.html')
	elif request.method == 'GET':
		child_name = find_child_name(current_user.id, cid)
		#print("Child Name: ", child_name, " worksheet: ",form_name)
		img = "progress_prot.png"
		path = os.path.join(app.root_path, 'static/img/upload/', current_user.username, date)
		plot_data = graph_data_gen(current_user.id, cid, form_id)
		#print(graph_data)
		#s = pd.Series([1,2,3])
		#fig, ax = plt.subplots()
		#s.plot.line()
		#fig.savefig(path + "/" + img)
		path = os.path.join('static//img//upload//', current_user.username, date, img)
		return render_template("public/progress.html", img = path, child = child_name, worksheet = form_name, plot_data = plot_data, tmin = form_time[0], tsec=form_time[1])
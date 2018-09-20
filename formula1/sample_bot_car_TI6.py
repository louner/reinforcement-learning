#!env python
#
# Auto-driving Bot
#
# Revision:	  v1.2
# Released Date: Aug 20, 2018
#
#coding=UTF-8

from time import time
from PIL  import Image
from io   import BytesIO
import os
import cv2
import math
import numpy as np
import base64
import logging
import tempfile


IS_DEBUG=False
TESTING=False

class Log(object):
	def __init__(self,is_debug=True):
		self.is_debug=is_debug
		self.msg=None
		self.logger = logging.getLogger('hearts_logs')
		hdlr = logging.FileHandler('logs/hearts_logs.log')
		formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
		hdlr.setFormatter(formatter)
		self.logger.addHandler(hdlr)
		self.logger.setLevel(logging.INFO)
	def show_message(self,msg):
		if self.is_debug:
			print(msg)
	def save_logs(self,msg):
		self.logger.info(msg)

system_log=Log(IS_DEBUG)

class FileModifierError(Exception):
	pass

class FileModifier(object):

	def __init__(self, fname):
		self.__write_dict = {}
		self.__filename = fname
		self.__tempfile = tempfile.TemporaryFile()
		with open(fname, 'rb') as fp:
			for line in fp:
				self.__tempfile.write(line)
		self.__tempfile.seek(0)

	def write(self, s, line_number = 'END'):
		if line_number != 'END' and not isinstance(line_number, (int, float)):
			raise FileModifierError("Line number %s is not a valid number" % line_number)
		try:
			self.__write_dict[line_number].append(s)
		except KeyError:
			self.__write_dict[line_number] = [s]

	def writeline(self, s, line_number = 'END'):
		self.write('%s\n' % s, line_number)

	def writelines(self, s, line_number = 'END'):
		for ln in s:
			self.writeline(s, line_number)

	def __popline(self, index, fp):
		try:
			ilines = self.__write_dict.pop(index)
			for line in ilines:
				fp.write(line)
		except KeyError:
			pass

	def close(self):
		self.__exit__(None, None, None)

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		with open(self.__filename,'w') as fp:
			for index, line in enumerate(self.__tempfile.readlines()):
				self.__popline(index, fp)
				fp.write(str(line))
			for index in sorted(self.__write_dict):
				for line in self.__write_dict[index]:
					fp.write(line)
		self.__tempfile.close()

class TrainingDataCollector(object):

	def __init__(self,path):
		self.states=[]
		self.hand_cards_record = []
		self.played_cards=[]
		self.action=[]
		self.order=[]
		self.text_file_path=path
		text_file = open(self.text_file_path, "w")
		text_file.close()
		#with FileModifier(self.text_file_path) as fp:
			#message="round,position,hcard1,hcard2,hcard3,hcard4,hcard5,hcard6,hcard7,hcard8,hcard9,hcard10,hcard11,hcard12,hcard13,rcard1,rcard2,rcard3,rcard4,1_0,1_1,1_2,1_3,1_4,1_5,1_6,1_7,1_8,1_9,1_10,1_11,1_12,2_0,2_1,2_2,2_3,2_4,2_5,2_6,2_7,2_8,2_9,2_10,2_11,2_12,2_13,2_14,2_15,2_16,2_17,2_18,2_19,2_20,2_21,2_22,2_23,3_0,3_1,3_2,3_3,3_4,3_5,3_6,3_7,3_8,3_9,4_0,4_1,4_2,4_3,4_4,4_5,4_6,4_7,4_8,4_9,4_10,4_11,4_12,4_13,5_0,5_1,5_2,5_3,5_4,5_5,5_6,5_7,5_8,5_9,5_10,5_11,strategy,score"
			#fp.writeline(message)  # To write the title
	def set_record(self,order,played_cards, state,action,hand_cards):
		self.order.append(order)
		self.played_cards.append(played_cards)
		self.states.append(state)
		self.action.append(action)
		self.hand_cards_record.append(hand_cards)

	def save_data_direct(self,message):
		with FileModifier(self.text_file_path) as fp:
			fp.writeline(message)  # To write at the end of the file
		self.reflush()

	def save_records_and_flush(self,rewards,rank):
		with FileModifier(self.text_file_path) as fp:
			for i in range(len(self.order)):
				round=i+1
				message= str(round)+","+str(self.order[i])+","
				hand_cards=self.hand_cards_record[i]
				for card in hand_cards:
					message += card.to_string() + ","
				remain=13-len(hand_cards)
				if remain>0:
					for j in range(remain):
						message += "null,"
				round_cards=self.played_cards[i]
				for card in round_cards:
					message+=card.to_string()+","
				current_state=self.states[i]
				for state_value in current_state:
					message += str(state_value) + ","
				message+=str(self.action[i])+","+str(rewards[i])+","
				message +=str(rank)
				fp.writeline(message)  # To write at the end of the file
		self.reflush()

	def reflush(self):
		self.states=[]
		self.hand_cards_record = []
		self.played_cards=[]
		self.action=[]
		self.order=[]


def logit(msg):
	print("%s" % msg)

class PID:

	def __init__(self,name, Kp, Ki, Kd, max_integral, min_interval = 0.001, set_point = 0.0, last_time = None):
		self._Kp		   = Kp #percentage
		self._Ki		   = Ki #i
		self._Kd		   = Kd #different
		self._oriKp = Kp  # percentage
		self._oriKi = Ki  # i
		self._oriKd = Kd  # different
		self._min_interval = min_interval
		self._max_integral = max_integral #10 or 0.5
		self.name		  = name
		self._set_point	= set_point
		self._last_time	= last_time if last_time is not None else time()
		self._p_value	  = 0.0
		self._i_value	  = 0.0
		self._d_value	  = 0.0
		self._d_time	   = 0.0
		self._d_error	  = 0.0
		self._last_error   = 0.0
		self._output	   = 0.0
		self.brakes		= 0.001
		self.velocy		= 0.01
		self.increase=0.01
		self.decrease=0.001
		self.current_increase=-1
		self.stability	 = True
		self.variant_begin_save=False
		self.current_angle = -100
		self.last_steering_angle=-100
		self.steering_variant = []
		self.step=0
		self.keep_steps=0
		self.current_action=0
		self.need_to_keep=False
		self.keep_time=0
		self.complete_trans=False
		self.complete_trans_step=0
		self.is_find_the_parameters=False
		self.kp_decrease=0.001
		self.ki_decrease = 0.0001
		self.kd_decrease=0.001
		self.record=[]
		self.is_turn=False
		self.turn_angle=0
		self.turn_count=0

	def pid_reset(self):
		self._Kp=self._oriKp
		self._Ki=self._oriKi
		self._Kd=self._oriKd
		self._last_time = time()
		self._p_value = 0.0
		self._i_value = 0.0
		self._d_value = 0.0
		self._d_time = 0.0
		self._d_error = 0.0
		self._last_error = 0.0
		self._output = 0.0
		self.current_increase = -1
		self.stability = True
		self.variant_begin_save = False
		self.current_angle = -100
		self.last_steering_angle = -100
		self.steering_variant = []
		self.step = 0
		self.keep_steps = 0
		self.current_action = 0
		self.need_to_keep = False
		self.keep_time = 0
		self.complete_trans = False
		self.complete_trans_step = 0
		self.is_find_the_parameters = False
		self.record = []
		self.is_turn = False
		self.turn_angle = 0
		self.turn_count = 0

	def stability_of_wheel(self,wheel_variants):
		variant_diff=1
		variant_count=0
		current_site=-1
		# for i in range (len(wheel_variants)):
		#	 if math.fabs(wheel_variants[i]) >= 0.01:
		#		 if current_site==-1:
		#			 current_site=i
		#			 variant_diff = wheel_variants[i]
		#		 else:
		#			 if math.fabs(current_site-i)==1:
		#				 if wheel_variants[i] > 0:
		#					 if variant_diff < 0:
		#						 variant_count += 1
		#					 variant_diff = wheel_variants[i]
		#				 else:
		#					 if variant_diff > 0:
		#						 variant_count += 1
		#					 variant_diff = wheel_variants[i]
		#				 current_site=i
		#			 else:
		#				 current_site=i
		#				 variant_diff = wheel_variants[i]
		# stability = variant_count / (len(wheel_variants) * 1.0)
		# return stability
		for variant in wheel_variants:
			if math.fabs(variant)>=0.01:
				if variant_diff==1:
					variant_diff=variant
				if variant>0:
					if variant_diff<0:
						variant_count+=1
					variant_diff = variant
				else:
					if variant_diff>0:
						variant_count+=1
					variant_diff = variant
		stability=variant_count / (len(wheel_variants) * 1.0)
		return stability

	def stability_of_wheel_2(self,wheel_variants):
		keep_stable=0
		count=0
		stability=0
		for i in range (len(wheel_variants)):
			if math.fabs(wheel_variants[i]) >= 0.05:
				keep_stable+=math.fabs(wheel_variants[i])
				count+=1
		if count!=0:
			stability=keep_stable/(count*1.0)
		return stability

	def save_steering_angle(self,angle_value):
		self.steering_variant.append(angle_value)

	def update_speed(self,angle_value,speed_value,d_time,d_error,cur_time,error):
		self.save_steering_angle(angle_value)
		self._p_value = error
		self._i_value = min(max(error * d_time, -self._max_integral), self._max_integral)
		self._d_value = d_error / d_time if d_time > 0 else 0.0
		if len(self.steering_variant) >= 5:
			probability = self.stability_of_wheel(
				self.steering_variant[(len(self.steering_variant) - 5):len(self.steering_variant)])
			probability2 = self.stability_of_wheel_2(
				self.steering_variant[(len(self.steering_variant) - 5):len(self.steering_variant)])
			#print "probability:{}".format(probability)
			if probability >= 0.2 or probability2>0.7:
				increase_value = 0
				if self.current_increase != -1:
					increase_value = math.fabs(self.current_increase - 0.05)
					self.current_increase = -1
				#print "break"

				self._output = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd
				self._d_time = d_time
				self._d_error = d_error
				self._last_time = cur_time
				self._last_error = error

				self._output = self._output - self.brakes - increase_value
				self.stability = False
			else:
				self._output = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd
				self._d_time = d_time
				self._d_error = d_error
				self._last_time = cur_time
				self._last_error = error

				self.current_increase = -1
				self.stability = True
				self.steering_variant = []
		else:
			if self.stability == True:
				self._output = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd
				self._d_time = d_time
				self._d_error = d_error
				self._last_time = cur_time
				self._last_error = error

				#print "speed up"
				if self.current_increase == -1:
					self.current_increase = self.velocy
				else:
					self.current_increase += self.increase
				self._output += self.current_increase
			else:
				self._output = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd
				self._d_time = d_time
				self._d_error = d_error
				self._last_time = cur_time
				self._last_error = error

	def update_wheel(self,angle_value,d_time,d_error,cur_time,error):
		self.save_steering_angle(angle_value)
		self._p_value = error
		self._i_value = min(max(error * d_time, -self._max_integral), self._max_integral)
		self._d_value = d_error / d_time if d_time > 0 else 0.0
		if len(self.steering_variant) >= 3:
			probability = self.stability_of_wheel_2(self.steering_variant[(len(self.steering_variant) - 3):len(self.steering_variant)])
			if probability >= 0.1 and probability < 0.7:
				if self._Kp > 0.06:
					self._Kp -= self.kp_decrease
				if self._Ki > 0.003:
					self._Ki -= self.ki_decrease
				if self._Kd > 0.03:
					self._Kd -= self.kd_decrease
				self.stability = False
			elif probability >= 0.7:
				if self._Kp > 0.18:
					self._Kp -= self.kp_decrease
				if self._Ki > 0.008:
					self._Ki -= self.ki_decrease
				if self._Kd > 0.08:
					self._Kd -= self.kd_decrease
				self.stability = False
				self._Kp = 0.18
				self._Ki = 0.008
				self._Kd = 0.08
			else:
				self.current_increase = -1
				self.stability = True
				self.steering_variant = []

			self._output = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd
			self._d_time = d_time
			self._d_error = d_error
			self._last_time = cur_time
			self._last_error = error
		else:
			if self.stability == True:
				if self._Kp < 0.18:
					self._Kp += self.kp_decrease
				if self._Ki < 0.008:
					self._Ki += self.ki_decrease
				if self._Kd < 0.08:
					self._Kd += self.kd_decrease
				self._Kp = 0.18
				self._Ki = 0.008
				self._Kd = 0.08
			self._output = self._p_value * self._Kp + self._i_value * self._Ki + self._d_value * self._Kd
			self._d_time = d_time
			self._d_error = d_error
			self._last_time = cur_time
			self._last_error = error
		self._output *= 1.3

	def update(self,cur_value1, cur_value, cur_time = None):

		if cur_time is None:
			cur_time = time()

		error   = self._set_point - cur_value #Different with zero
		d_time  = cur_time - self._last_time #Time interval
		d_error = error - self._last_error # differencial error

		if d_time >= self._min_interval: #0.001
			if self.name!="wheel":
				self.update_speed(cur_value1,cur_value,d_time,d_error,cur_time,error)
			else:
				self.update_wheel(cur_value1, d_time, d_error, cur_time, error)

		return self._output,self._Kp,self._Ki,self._Kd

	def reset(self, last_time = None, set_point = 0.0):
		self._set_point	= set_point
		self._last_time	= last_time if last_time is not None else time()
		self._p_value	  = 0.0
		self._i_value	  = 0.0
		self._d_value	  = 0.0
		self._d_time	   = 0.0
		self._d_error	  = 0.0
		self._last_error   = 0.0
		self._output	   = 0.0

	def assign_set_point(self, set_point):
		self._set_point = set_point

	def get_set_point(self):
		return self._set_point

	def get_p_value(self):
		return self._p_value

	def get_i_value(self):
		return self._i_value

	def get_d_value(self):
		return self._d_value

	def get_delta_time(self):
		return self._d_time

	def get_delta_error(self):
		return self._d_error

	def get_last_error(self):
		return self._last_error

	def get_last_time(self):
		return self._last_time

	def get_output(self):
		return self._output

class ImageProcessor(object):

	track_index_current = 0
	left_or_right=2
	BLUE_LINE = 0
	RED_LINE = 2
	GREEN_LINE = 1
	AUTO_DETECT=3
	is_translating=False
	TRANSLATION_STEPS=50
	current_step=0
	ALREADY_TRANSED=False
	current_angle=-1
	WALL_NOISE=3
	left_wall_count=0
	right_wall_count=0

	@staticmethod
	def switch_color(switch_color):
		if switch_color==ImageProcessor.AUTO_DETECT:
			ImageProcessor.track_index_current=-1
		else:
			ImageProcessor.track_index_current=switch_color

	@staticmethod
	def show_image(img, name = "image", scale = 1.0):
		if scale and scale != 1.0:
			newsize=3
			img = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC)
		cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
		cv2.imshow(name, img)
		cv2.waitKey(1)

	@staticmethod
	def save_image(folder, img, prefix = "img", suffix = ""):
		from datetime import datetime
		filename = "%s-%s%s.jpg" % (prefix, datetime.now().strftime('%Y%m%d-%H%M%S-%f'), suffix)
		cv2.imwrite(os.path.join(folder, filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


	@staticmethod
	def rad2deg(radius): # Radius to degree
		return radius / np.pi * 180.0


	@staticmethod
	def deg2rad(degree): # Degree to Radius
		return degree / 180.0 * np.pi


	@staticmethod
	def bgr2rgb(img): #Covert to color space
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


	@staticmethod
	def _normalize_brightness(img):
		maximum = img.max()
		if maximum == 0:
			return img
		adjustment = min(255.0/img.max(), 3.0)
		normalized = np.clip(img * adjustment, 0, 255)
		normalized = np.array(normalized, dtype=np.uint8)
		return normalized


	@staticmethod
	def _flatten_rgb(img):
		r, g, b = cv2.split(img)
		r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
		g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
		b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
		y_filter = ((r >= 128) & (g >= 128) & (b < 100))
		#w_filter = ((r <= 160) & (r >= 50) & (g <= 160) & (g >= 50) & (b <= 160) & (b >= 50))

		r[y_filter], g[y_filter] = 255, 255
		b[np.invert(y_filter)] = 0
		#r[w_filter], g[w_filter] = 255, 255
		#b[np.invert(w_filter)] = 0

		b[b_filter], b[np.invert(b_filter)] = 255, 0
		r[r_filter], r[np.invert(r_filter)] = 255, 0
		g[g_filter], g[np.invert(g_filter)] = 255, 0

		flattened = cv2.merge((r, g, b))
		return flattened


	@staticmethod
	def _crop_image(img):
		bottom_half_ratios = (0.55, 1.0)
		bottom_half_slice  = slice(*(int(x * img.shape[0]) for x in bottom_half_ratios))
		bottom_half		= img[bottom_half_slice, :, :]
		return bottom_half


	@staticmethod
	def preprocess(img):
		img = ImageProcessor._crop_image(img)
		#img = ImageProcessor._normalize_brightness(img)
		img = ImageProcessor._flatten_rgb(img)
		return img


	@staticmethod
	def find_lines(img):
		grayed	  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blurred	 = cv2.GaussianBlur(grayed, (3, 3), 0)
		#edged	  = cv2.Canny(blurred, 0, 150)

		sobel_x	 = cv2.Sobel(blurred, cv2.CV_16S, 1, 0)
		sobel_y	 = cv2.Sobel(blurred, cv2.CV_16S, 0, 1)
		sobel_abs_x = cv2.convertScaleAbs(sobel_x)
		sobel_abs_y = cv2.convertScaleAbs(sobel_y)
		edged	   = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

		lines	   = cv2.HoughLinesP(edged, 1, np.pi / 180, 10, 5, 5)
		return lines


	@staticmethod
	def _find_best_matched_line(thetaA0, thetaB0, tolerance, vectors, matched = None, start_index = 0):
		if matched is not None:
			matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
			matched_angle = abs(np.pi/2 - matched_thetaB)

		for i in xrange(start_index, len(vectors)):
			distance, length, thetaA, thetaB, coord = vectors[i]

			if (thetaA0 is None or abs(thetaA - thetaA0) <= tolerance) and \
			   (thetaB0 is None or abs(thetaB - thetaB0) <= tolerance):
				
				if matched is None:
					matched = vectors[i]
					matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
					matched_angle = abs(np.pi/2 - matched_thetaB)
					continue

				heading_angle = abs(np.pi/2 - thetaB)

				if heading_angle > matched_angle:
					continue
				if heading_angle < matched_angle:
					matched = vectors[i]
					matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
					matched_angle = abs(np.pi/2 - matched_thetaB)
					continue
				if distance < matched_distance:
					continue
				if distance > matched_distance:
					matched = vectors[i]
					matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
					matched_angle = abs(np.pi/2 - matched_thetaB)
					continue
				if length < matched_length:
					continue
				if length > matched_length:
					matched = vectors[i]
					matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
					matched_angle = abs(np.pi/2 - matched_thetaB)
					continue

		return matched


	@staticmethod
	def find_steering_angle_by_line(img, last_steering_angle, debug = True):
		steering_angle = 0.0
		lines		  = ImageProcessor.find_lines(img)

		if lines is None:
			return steering_angle

		image_height = img.shape[0]
		image_width  = img.shape[1]
		camera_x	 = image_width / 2
		camera_y	 = image_height
		vectors	  = []

		for line in lines:
			for x1, y1, x2, y2 in line:
				thetaA   = math.atan2(abs(y2 - y1), (x2 - x1))
				thetaB1  = math.atan2(abs(y1 - camera_y), (x1 - camera_x))
				thetaB2  = math.atan2(abs(y2 - camera_y), (x2 - camera_x))
				thetaB   = thetaB1 if abs(np.pi/2 - thetaB1) < abs(np.pi/2 - thetaB2) else thetaB2

				length   = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
				distance = min(math.sqrt((x1 - camera_x) ** 2 + (y1 - camera_y) ** 2),
							   math.sqrt((x2 - camera_x) ** 2 + (y2 - camera_y) ** 2))

				vectors.append((distance, length, thetaA, thetaB, (x1, y1, x2, y2)))

				if debug:
					# draw the edges
					cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

		#the line of the shortest distance and longer length will be the first choice
		vectors.sort(lambda a, b: cmp(a[0], b[0]) if a[0] != b[0] else -cmp(a[1], b[1]))

		best = vectors[0]
		best_distance, best_length, best_thetaA, best_thetaB, best_coord = best
		tolerance = np.pi / 180.0 * 10.0

		best = ImageProcessor._find_best_matched_line(best_thetaA, None, tolerance, vectors, matched = best, start_index = 1)
		best_distance, best_length, best_thetaA, best_thetaB, best_coord = best

		if debug:
			#draw the best line
			cv2.line(img, best_coord[:2], best_coord[2:], (0, 255, 255), 2)

		if abs(best_thetaB - np.pi/2) <= tolerance and abs(best_thetaA - best_thetaB) >= np.pi/4:
			print("*** sharp turning")
			best_x1, best_y1, best_x2, best_y2 = best_coord
			f = lambda x: int(((float(best_y2) - float(best_y1)) / (float(best_x2) - float(best_x1)) * (x - float(best_x1))) + float(best_y1))
			left_x , left_y  = 0, f(0)
			right_x, right_y = image_width - 1, f(image_width - 1)

			if left_y < right_y:
				best_thetaC = math.atan2(abs(left_y - camera_y), (left_x - camera_x))

				if debug:
					#draw the last possible line
					cv2.line(img, (left_x, left_y), (camera_x, camera_y), (255, 128, 128), 2)
					cv2.line(img, (left_x, left_y), (best_x1, best_y1), (255, 128, 128), 2)
			else:
				best_thetaC = math.atan2(abs(right_y - camera_y), (right_x - camera_x))

				if debug:
					#draw the last possible line
					cv2.line(img, (right_x, right_y), (camera_x, camera_y), (255, 128, 128), 2)
					cv2.line(img, (right_x, right_y), (best_x1, best_y1), (255, 128, 128), 2)

			steering_angle = best_thetaC
			#steering_angle = best_thetaB
		else:
			steering_angle = best_thetaB

		if (steering_angle - np.pi/2) * (last_steering_angle - np.pi/2) < 0:
			last = ImageProcessor._find_best_matched_line(None, last_steering_angle, tolerance, vectors)

			if last:
				last_distance, last_length, last_thetaA, last_thetaB, last_coord = last
				steering_angle = last_thetaB

				if debug:
					#draw the last possible line
					cv2.line(img, last_coord[:2], last_coord[2:], (255, 128, 128), 2)

		if debug:
			#draw the steering direction
			r = 60
			x = image_width / 2 + int(r * math.cos(steering_angle))
			y = image_height	- int(r * math.sin(steering_angle))
			cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 2)
			logit("line angle: %0.2f, steering angle: %0.2f, last steering angle: %0.2f" % (ImageProcessor.rad2deg(best_thetaA), ImageProcessor.rad2deg(np.pi/2-steering_angle), ImageProcessor.rad2deg(np.pi/2-last_steering_angle)))

		return (np.pi/2 - steering_angle)

	@staticmethod
	def wall_detection (sr, sg, sb):
		black_count = 0
		yellow_count = 0
		for i in range(int(len(sr) / 10)):
			for j in range(int(len(sr[i]) / 10)):
				inverse_i = int(len(sr)/8) + i
				inverse_j = len(sr[i]) - 1 - j
				#print "r:{},g:{},b:{}".format(sr[i][j],sg[i][j],sb[i][j])
				if sr[i][j] == 0 and sg[i][j] == 0 and sb[i][j] == 0:
					black_count += 1
				if sr[inverse_i][inverse_j] == 0 and sg[inverse_i][inverse_j] == 0 and sb[inverse_i][inverse_j] == 0:
					yellow_count += 1
		#print("black_count:{}, yellow_count:{}".format(black_count,yellow_count))
		is_left_wall=False
		is_right_wall=False
		if black_count>=320:
			#ImageProcessor.right_wall_count = 0
			#if ImageProcessor.left_wall_count>=ImageProcessor.WALL_NOISE:
				is_left_wall = True
				#ImageProcessor.left_wall_count = 0
			#else:
			 #   ImageProcessor.left_wall_count+=1
		elif yellow_count>=40:
			#ImageProcessor.left_wall_count = 0
			if yellow_count>=40:
				#if ImageProcessor.right_wall_count >= ImageProcessor.WALL_NOISE:
					is_right_wall=True
				#	ImageProcessor.right_wall_count = 0
				#else:
				#   ImageProcessor.right_wall_count+=1
			#else:
			#	ImageProcessor.right_wall_count = 0

		return is_left_wall, is_right_wall

	@staticmethod
	def turn_to_second_large_color_region(is_left_wall, is_right_wall,tracks):
		turn_index=-1
		if ImageProcessor.ALREADY_TRANSED == False:
			if is_left_wall or is_right_wall:
				if ImageProcessor.is_translating == False:
					ImageProcessor.is_translating = True
					target_value = sorted(tracks)[1]
					for i in range(len(tracks)):
						if tracks[i] == target_value:
							turn_index = i
							break
		if ImageProcessor.is_translating and ImageProcessor.current_step < ImageProcessor.TRANSLATION_STEPS:
			ImageProcessor.current_step += 1
		elif ImageProcessor.is_translating and ImageProcessor.current_step >= ImageProcessor.TRANSLATION_STEPS:
			ImageProcessor.current_step = 0
			ImageProcessor.is_translating = False
			ImageProcessor.ALREADY_TRANSED = True
		return turn_index

	@staticmethod
	def keep_in_same_color_region(tracks):
		maximum_color_idx = np.argmax(tracks, axis=None)
		if ImageProcessor.track_index_current == -1 or ImageProcessor.track_index_current == ImageProcessor.AUTO_DETECT:
			ImageProcessor.track_index_current = maximum_color_idx
		if ImageProcessor.track_index_current != maximum_color_idx:
			maximum_color_idx = ImageProcessor.track_index_current
		return maximum_color_idx

	@staticmethod
	def get_color_region_angle(current_angle,tracks,track_list,px,image_height,camera_x):
		if np.isnan(px):
			maximum_color_idx = np.argmax(tracks, axis=None)
			if maximum_color_idx==ImageProcessor.track_index_current:
				sorted(tracks)[1]
				target_value = sorted(tracks)[1]
				for i in range(len(tracks)):
					if tracks[i] == target_value:
						maximum_color_idx = i
						break
			ImageProcessor.track_index_current = maximum_color_idx
			_target = track_list[maximum_color_idx]
			_y, _x = np.where(_target == 255)
			px = np.mean(_x)
			ImageProcessor.ALREADY_TRANSED=False
		if np.isnan(px):
			steering_angle=current_angle
		else:
			steering_angle = math.atan2(image_height, (px - camera_x))
		return steering_angle

	@staticmethod
	def get_object_count(sr, sg, sb):
		object_count = 0
		for i in range(int(len(sr) / 10)):
			for j in range(int(len(sr[i]) / 10)):
				inverse_i = i
				inverse_j = len(sr[i])/2 + j
				if sr[inverse_i][inverse_j] == 0 and sg[inverse_i][inverse_j] == 0 and sb[inverse_i][inverse_j] == 0:
					object_count += 1
		return object_count

	@staticmethod
	def is_crash(img):
		r, g, b	  = cv2.split(img)
		image_height = img.shape[0]
		image_width  = img.shape[1]
		camera_x	 = image_width / ImageProcessor.left_or_right
		image_sample = slice(int(image_height*0), int(image_height))
		sr, sg, sb   = r[image_sample, :], g[image_sample, :], b[image_sample, :]	 #sampling
		is_left_wall, is_right_wall = ImageProcessor.wall_detection(sr, sg, sb)
		return is_left_wall or is_right_wall

	@staticmethod
	def find_steering_angle_by_color(img, last_steering_angle, debug = True):
		r, g, b	  = cv2.split(img)
		image_height = img.shape[0]
		image_width  = img.shape[1]
		camera_x	 = image_width / ImageProcessor.left_or_right
		image_sample = slice(int(image_height*0), int(image_height))
		sr, sg, sb   = r[image_sample, :], g[image_sample, :], b[image_sample, :]	 #sampling
		track_list   = [sr, sg, sb]
		#tracks	   = map(lambda x: len(x[x > 200]), [sr, sg, sb])
		tracks	   = [len(x[x > 200]) for x in [sr, sg, sb]]
		#Check the left wall or right wall
		is_left_wall,is_right_wall=ImageProcessor.wall_detection(sr, sg, sb)

		#print("Left wall:{}, Right Wall:{}".format(is_left_wall, is_right_wall))
		#Keep the car in the center color region
		turn_index=ImageProcessor.turn_to_second_large_color_region(is_left_wall,is_right_wall,tracks)
		if turn_index>-1:
			ImageProcessor.track_index_current=turn_index

		#tracks_seen  = filter(lambda y: y > 50, tracks)
		tracks_seen = [track for track in tracks if track > 50]

		if len(tracks_seen) == 0:
			return 0.0
		#Keep the car in the same color region
		maximum_color_idx = ImageProcessor.keep_in_same_color_region(tracks)

		_target = track_list[maximum_color_idx]
		_y, _x = np.where(_target == 255)
		px = np.mean(_x)
		#Error handling for
		steering_angle = ImageProcessor.get_color_region_angle(ImageProcessor.current_angle,tracks,track_list,px,image_height,camera_x)
		ImageProcessor.current_angle=steering_angle
		if debug:
			#draw the steering direction
			r = 60
			x = int(image_width / 2 + int(r * math.cos(steering_angle)))
			y = int(image_height - int(r * math.sin(steering_angle)))
			#cv2.line(img, (int(image_width / 2), image_height), (x, y), (255, 0, 255), 2)
			logit("steering angle: %0.2f, last steering angle: %0.2f" % (ImageProcessor.rad2deg(steering_angle), ImageProcessor.rad2deg(np.pi/2-last_steering_angle)))

		return (np.pi/2 - steering_angle) * 2.0

class AutoDrive(object):
	#Two PID controllers (steer and throttle controller)
	STEERING_PID_Kp = 0.2 #0.3-0.06(0.2)
	STEERING_PID_Ki = 0.008 #0.01-0.003(0.008)
	STEERING_PID_Kd = 0.08 #0.1-0.03 (0.08)
	STEERING_PID_max_integral = 40
	THROTTLE_PID_Kp = 0.02
	THROTTLE_PID_Ki = 0.005
	THROTTLE_PID_Kd = 0.02
	THROTTLE_PID_max_integral = 0.5
	MAX_STEERING_HISTORY = 3
	MAX_THROTTLE_HISTORY = 3
	DEFAULT_SPEED = 0.5

	debug = True

	def __init__(self, car,car_training_data_collector, record_folder = None):
		self._record_folder	= record_folder
		self._steering_pid	 = PID("wheel",Kp=self.STEERING_PID_Kp  , Ki=self.STEERING_PID_Ki  , Kd=self.STEERING_PID_Kd  , max_integral=self.STEERING_PID_max_integral)
		self._throttle_pid	 = PID("throttle",Kp=self.THROTTLE_PID_Kp  , Ki=self.THROTTLE_PID_Ki  , Kd=self.THROTTLE_PID_Kd  , max_integral=self.THROTTLE_PID_max_integral)
		self._throttle_pid.assign_set_point(self.DEFAULT_SPEED)
		self.car_training_data_collector=car_training_data_collector
		#Historical data
		self._steering_history = []
		self._throttle_history = []
		#Register card to auto driving
		self._car = car
		self._car.register(self)
		self.current_lap=1

	#When you get the input data
	def on_dashboard(self, src_img, last_steering_angle, speed, throttle, info):
		track_img	 = ImageProcessor.preprocess(src_img) #get track image
		#cur_radian, line_results = self.m_twQTeamImageProcessor.findSteeringAngle(src_img, proc_img)

		current_angle = ImageProcessor.find_steering_angle_by_color(track_img, last_steering_angle, debug = self.debug)
		#current_angle = ImageProcessor.find_steering_angle_by_line(track_img, last_steering_angle, debug = self.debug)
		steering_angle,Kp,Ki,Kd = self._steering_pid.update(-current_angle,-current_angle) #Current angle
		throttle,_,_,_	   = self._throttle_pid.update(-current_angle,speed) #current speed
		message=str(info["lap"])+","+str(steering_angle)+","+str(Kp)+","+str(Ki)+","+str(Kd)+","
		message+=str(throttle)
		if info["lap"]>self.current_lap:
			self.current_lap=info["lap"]
			self._steering_pid.pid_reset()
			self._throttle_pid.pid_reset()
			#throttle=0.3
			#steering_angle=0.0
		total_time = info["time"].split(":") if "time" in info else []  # spend time
		seconds = float(total_time.pop()) if len(total_time) > 0 else 0.0
		minutes = int(total_time.pop()) if len(total_time) > 0 else 0
		hours = int(total_time.pop()) if len(total_time) > 0 else 0
		elapsed = ((hours * 60) + minutes) * 60 + seconds
		# if hours==0 and minutes==0 and seconds==0:
		#	 ImageProcessor.switch_color(ImageProcessor.AUTO_DETECT)
		#	 ImageProcessor.ALREADY_TRANSED=False
		#	 ImageProcessor.current_step = 0
		#	 ImageProcessor.is_translating = False
		self.car_training_data_collector.save_data_direct(message)
		#debug and save captured images
		if self.debug:
			ImageProcessor.show_image(src_img, "source")
			ImageProcessor.show_image(track_img, "track")
			ImageProcessor.save_image('images', src_img, 'src_img', 'current_angle_%f'%(current_angle))
			ImageProcessor.save_image('images', track_img, 'track_img', 'current_angle_%f' % (current_angle))
			logit("steering PID: %0.2f (%0.2f) => %0.2f (%0.2f)" % (current_angle, ImageProcessor.rad2deg(current_angle), steering_angle, ImageProcessor.rad2deg(steering_angle)))
			logit("throttle PID: %0.4f => %0.4f" % (speed, throttle))
			logit("info: %s" % repr(info))

		if self._record_folder:
			suffix = "-deg%0.3f" % (ImageProcessor.rad2deg(steering_angle))
			ImageProcessor.save_image(self._record_folder, src_img  , prefix = "cam", suffix = suffix)
			ImageProcessor.save_image(self._record_folder, track_img, prefix = "trk", suffix = suffix)

		#smooth the control signals
		self._steering_history.append(steering_angle)
		self._steering_history = self._steering_history[-self.MAX_STEERING_HISTORY:]
		self._throttle_history.append(throttle)
		self._throttle_history = self._throttle_history[-self.MAX_THROTTLE_HISTORY:]
		if ImageProcessor.is_crash(track_img):
			self._car.control(sum(self._steering_history)/self.MAX_STEERING_HISTORY, sum(self._throttle_history)/self.MAX_THROTTLE_HISTORY, cmd='restart')
		else:
			self._car.control(sum(self._steering_history)/self.MAX_STEERING_HISTORY, sum(self._throttle_history)/self.MAX_THROTTLE_HISTORY)

class Car(object):
	MAX_STEERING_ANGLE = 40.0

	def __init__(self, control_function):
		self._driver = None
		self._control_function = control_function


	def register(self, driver):
		self._driver = driver

	def on_dashboard(self, dashboard):
		#normalize the units of all parameters
		last_steering_angle = np.pi/2 - float(dashboard["steering_angle"]) / 180.0 * np.pi #steel wheel angle
		throttle			= float(dashboard["throttle"]) #speed control
		speed			   = float(dashboard["speed"]) # current speed
		img				 = ImageProcessor.bgr2rgb(np.asarray(Image.open(BytesIO(base64.b64decode(dashboard["image"])))))#current image

		total_time = dashboard["time"].split(":") if "time" in dashboard else []#spend time
		seconds	= float(total_time.pop()) if len(total_time) > 0 else 0.0
		minutes	= int(total_time.pop())   if len(total_time) > 0 else 0
		hours	  = int(total_time.pop())   if len(total_time) > 0 else 0
		elapsed	= ((hours * 60) + minutes) * 60 + seconds

		info = {
			"lap"	: int(dashboard["lap"]) if "lap" in dashboard else 0,
			"elapsed": elapsed,
			"status" : int(dashboard["status"]) if "status" in dashboard else 0,
		}
		self._driver.on_dashboard(img, last_steering_angle, speed, throttle, info)

	def control(self, steering_angle, throttle, cmd=None):
		#convert the values with proper units
		steering_angle = min(max(ImageProcessor.rad2deg(steering_angle), -Car.MAX_STEERING_ANGLE), Car.MAX_STEERING_ANGLE)
		self._control_function(steering_angle, throttle, cmd)

if __name__ == "__main__":
	import shutil
	import argparse
	from datetime import datetime

	import socketio
	import eventlet
	import eventlet.wsgi
	from flask import Flask

	parser = argparse.ArgumentParser(description='AutoDriveBot')
	parser.add_argument(
			'record',
			type=str,
			nargs='?',
			default='',
			help='Path to image folder to record the images.'
	)
	track_name="Track_5"
	text_file_path = "logs/car_training_data_" + track_name + ".txt"
	car_training_data_collector=TrainingDataCollector(text_file_path)
	message = "lap,steering_angle,Kp,Ki,Kd,throttle"
	car_training_data_collector.save_data_direct(message)
	args = parser.parse_args()
	ImageProcessor.switch_color(ImageProcessor.AUTO_DETECT)
	#Inpiut arguments
	if args.record:
		if not os.path.exists(args.record):
			os.makedirs(args.record)
		logit("Start recording images to %s..." % args.record)
	sio = socketio.Server()

	def send_control(steering_angle, throttle, cmd=None):
		if cmd is None:
			sio.emit(
				"steer",
				data={
					'steering_angle': str(steering_angle),
					'throttle': str(throttle)
				},
				skip_sid=True)
		else:
			sio.emit("restart", data={}, skip_sid=True)
			print('restart')
	car = Car(control_function = send_control)
	drive = AutoDrive(car,car_training_data_collector, args.record)

	@sio.on('telemetry')
	def telemetry(sid, dashboard):
		if dashboard:
			car.on_dashboard(dashboard)
		else:
			sio.emit('manual', data={}, skip_sid=True)

	@sio.on('connect')
	def connect(sid, environ): # First time connect to car and environment
		car.control(0, 0)
	app = socketio.Middleware(sio, Flask(__name__))
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

# vim: set sw=4 ts=4 et :


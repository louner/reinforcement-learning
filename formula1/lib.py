from sample_bot_car_TI6 import *
from count_angle import *
import pyautogui

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

import math
from queue import Queue

import tensorflow as tf

import tensorflow.contrib.slim as slim
from random import random, choice

from time import sleep
from random import randint, random, sample
from IPython import display
import heapq

import traceback
import matplotlib.pyplot as plt

import logging
handler = logging.FileHandler('logs/drive.log', mode='w')
handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s'))
logger = logging.getLogger('drive')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

batch_size = 128
max_size = batch_size*10
t_max = 100
reload_window = batch_size
reward_decay = 0.99
GAMMA = 0.99
last_state = None
last_action = None
last_reward = None
learning_rate = 0.001
MU = 0.999
crashing_reward = -1

ANGLES = [0, 0.3490658503988659, -0.3490658503988659, 0.6981317007977318, -0.6981317007977318]
THROTTLES = [0.1]
ACTIONS = [(angle, throttle) for angle in ANGLES for throttle in THROTTLES]

def calculate_avg(series):
    return [np.mean(serie) for serie in series]

def plot(series):
    colors = ['r', 'y', 'g', 'b', 'p']
    to_plot = []
    for i in range(len(series)):
        to_plot += [series[i], colors[i]]
        
    plt.plot(*to_plot)
    display.display(plt.gcf())
    display.clear_output(wait=True)

def discount_reward(one_round):
    reward = crashing_reward
    for transaction in reversed(one_round):
        reward += transaction['reward']
        reward *= reward_decay
        transaction['reward'] = reward

class Action:
    def __init__(self, action_idx):
        self.action_idx = action_idx
        self.steering_angle = ACTIONS[action_idx][0]
        self.throttle = ACTIONS[action_idx][1]
        
    def __repr__(self):
        return str({'throttle': self.throttle, 'angle': ImageProcessor.rad2deg(self.steering_angle)})

class Transaction:
    columns = ['state', 'action', 'reward', 'next_state', 'advantage']
    
    def __init__(self, item):
        self.item = dict(zip(Transaction.columns, item))
    
    def  __lt__(self, other):
        return self.item['id'] < other.item['id']

    def __repr__(self):
        return str(self.item)
        
    def __getitem__(self, key):
        return self.item[key]
    
    def __setitem__(self, key, value):
        self.item[key] = value

class Memory:
    def __init__(self, max_size):
        MAX_TRANSACTION = Transaction(Transaction.columns)
        MAX_TRANSACTION.item['id'] = -1
        self.d = []
        self.id = 0
        self.max_size = max_size

    def insert(self, item):
        self.id += 1
        item['id'] = self.id

        if len(self.d) == self.max_size:
            heapq.heapreplace(self.d, item)
        else:
            heapq.heappush(self.d, item)

    def batch(self, n):
        if len(self.d) > n:
            batch = np.array(sample(self.d, n))
            return batch
        else:
            return np.array(self.d)


# In[6]:


# policy & value network only different at last layer
# common part is cnn1-cnn1-fc
def build_network():
    with tf.variable_scope('Action-Critic-Network-Common', reuse=tf.AUTO_REUSE):
        inputs = tf.placeholder(shape=(None, 108, 320, 1), dtype=tf.float32) # track image
        #inputs = tf.placeholder(shape=[None, 108, 320, 1], dtype=tf.float32, name='input_img') # src image
        layer_1 = slim.conv2d(inputs=inputs, num_outputs=16, kernel_size=(8, 8), stride=(4, 4), padding='VALID', activation_fn=tf.nn.tanh)
        layer_2 = slim.conv2d(inputs=layer_1, num_outputs=32, kernel_size=(4, 4), stride=(2, 2), padding='VALID', activation_fn=tf.nn.tanh)
        fc_layer = tf.layers.dense(inputs=slim.flatten(layer_2), units=256, activation=tf.nn.tanh)
    return inputs, fc_layer

class ACNetwork:
    def __init__(self):
        self.action_size = len(ACTIONS)
        self.output_size = 1
        self.model_folder = './model/actor_critic_network'
        
        self.advantage = tf.placeholder(shape=(None, self.action_size), dtype=tf.float32, name='advantage')
        self.rewards = tf.placeholder(shape=(None), dtype=tf.float32, name='rewards')
        
        inputs, output = build_network()
        self.inputs = inputs
        self.common = output
    
        self.policy_metrics = {}
        self.policy_metrics['loss'] = []
        
        self.value_metrics = {}
        self.value_metrics['loss'] = []
        
        with tf.variable_scope('policy-network'):
            self.actions_values = tf.layers.dense(output, units=self.action_size, activation=tf.nn.softmax) # batch_size, action_size
            self.best_action = tf.argmax(self.actions_values, axis=1)

            self.policy_loss = tf.reduce_sum(tf.log(self.actions_values) * self.advantage)*-1
            
        with tf.variable_scope('value-network'):
            self.state_value = tf.layers.dense(output, units=self.output_size)
            self.value_loss = tf.reduce_sum(tf.square(self.state_value - self.rewards))
        
        self.loss = self.policy_loss + self.value_loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.target_sess = tf.Session()
        
        self.sess.run(tf.global_variables_initializer())
        self.target_sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
    
    def get_action(self, state):
        best_action = self.target_sess.run(self.best_action, feed_dict={self.inputs: state})[0]
        return best_action
        
    def get_state_values(self, states):
        state_values = self.target_sess.run(self.state_value, feed_dict={self.inputs: states})
        return state_values
    
    def get_state_value(self, state):
        return self.get_state_values(state)[0]
    
    def get_advantages(self, batch):
        advantages = []
        for transaction in batch:
            # unselected actions get 0 advantage, selected action get reward - current_state_value advantage
            advantage = [0]*self.action_size
            advantage[transaction['action']] = transaction['advantage']

            advantages.append(advantage)
            
        advantages = np.array(advantages)
        return advantages

    def update(self, batch):
        states = [transaction['state'] for transaction in batch]
        advantages = self.get_advantages(batch)
        rewards = [transaction['reward'] for transaction in batch]
        
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.inputs: states, self.advantage: advantages, self.rewards: rewards})
        
    def reload(self, model_name):
        self.saver.save(self.sess, '%s/%s'%(self.model_folder, model_name))
        self.saver.restore(self.target_sess, tf.train.latest_checkpoint(self.model_folder))

class QNetwork:
    def __init__(self):
        self.state_size = (None, 108, 320, 3)
        self.action_size = len(ACTIONS)
        self.model_folder = './model'

        inputs, outputs = build_network()
        self.input = inputs

        with tf.variable_scope('QNetwork'):
            self.label = tf.placeholder(shape=(None, self.action_size), dtype=tf.float32)
            self.global_step = tf.Variable(0, dtype=tf.int32)

            self.qvalue = tf.layers.dense(outputs, units=self.action_size)  # batch_size, action_size
            self.best_action = tf.argmax(self.qvalue, axis=1)

            self.loss = tf.losses.mean_squared_error(labels=self.label, predictions=self.qvalue)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_step = self.do_clipping(self.loss, optimizer)

        self.sess = sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))
        self.target_Q_sess = sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))

        self.sess.run(tf.global_variables_initializer())
        self.target_Q_sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def do_clipping(self, loss, optimizer):
        grad_vars = optimizer.compute_gradients(loss)
        self.grad_vars = [
            (tf.clip_by_norm(grad, clip_norm=10), var)
            for grad, var in grad_vars if grad is not None
        ]
        self.grad_dict = dict([
            [var.name, grad]
            for grad, var in self.grad_vars
        ])

        train_step = optimizer.apply_gradients(self.grad_vars, global_step=self.global_step)
        return train_step

    def get_Q_value(self, states, sess='target'):
        sess = self.decide_sess(sess)
        states = np.array(states).reshape((-1, 108, 320, 1))
        return sess.run(self.qvalue, feed_dict={self.input: states})

    def get_best_Q_value(self, states, sess='target'):
        sess = self.decide_sess(sess)
        states = np.array(states).reshape((-1, 108, 320, 1))
        return sess.run(self.best_action, feed_dict={self.input: states})

    def get_action(self, state, sess='target'):
        sess = self.decide_sess(sess)
        state = np.array([state]).reshape((-1, 108, 320, 1))
        return int(sess.run(self.best_action, feed_dict={self.input: state}))

    def update(self, batch):
        states = np.array([transaction['state'] for transaction in batch]).reshape((-1, 108, 320, 1))
        labels = np.array([transaction['label'] for transaction in batch])

        return self.sess.run([self.train_step, self.loss], feed_dict={self.input: states, self.label: labels})

    def reload(self, name):
        self.saver.save(self.sess, '%s/%s' % (self.model_folder, name), global_step=self.global_step)
        self.saver.restore(self.target_Q_sess, tf.train.latest_checkpoint(self.model_folder))

    def decide_sess(self, sess):
        if sess == 'target':
            sess = self.target_Q_sess
        elif sess == 'online':
            sess = self.sess
        else:
            raise Exception('unknown sess %s' % (sess))
        return sess

class ACNDrive(AutoDrive):
    def __init__(self, *args, **kwargs):
        AutoDrive.__init__(self, *args, **kwargs)
        
        #self.policy_network = PolicyNetwork()
        #self.value_network = ValueNetwork()
        self.driver = ACNetwork()
        self.memory = Memory(max_size=max_size)
        self.total_steps = 0
        
        self.init_game()
        self.take_random_action_prob = 1
        
        self.reward_history = []
        self.metrics = {}
        self.metrics['round_length'] = [] # number of games, 1
        self.metrics['rewards'] = [] # number of games, game round length
        self.metrics['speeds'] = [] # number of games, game round length

    def init_game(self):
        self.one_round = []
        self.round_id = 1
        
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.last_elapsed = None
        
        self.speeds_batch = [1]*10
        
        self.rewards = []
        self.speeds = []
        
    def collect_game_metrics(self):
        self.metrics['round_length'].append(self.round_id)
        self.metrics['rewards'].append(self.rewards)
        self.metrics['speeds'].append(self.speeds)
        
        self.reward_history += self.rewards
        
    def plot_metrics(self):
        print(self.take_random_action_prob)
        avg_reward = calculate_avg(self.metrics['rewards'])
        avg_speed = calculate_avg(self.metrics['speeds'])
        avg_round_length = calculate_avg(self.metrics['round_length'])
        
        #plot([avg_reward, avg_speed, avg_round_length])
        plot([avg_reward, avg_speed])
        
    def update_speeds(self, speed):
        self.speeds_batch = self.speeds_batch[1:]
        self.speeds_batch.append(speed)

    def low_speed(self):
        return sum(self.speeds_batch) <= 0.01

    def is_crash(self, src_img):
        return ImageProcessor.is_crash(src_img)# and self.low_speed()
    
    def calculate_reward(self, current_aggle, speed):
        #return speed * math.cos(current_aggle)
        return math.cos(current_aggle)

    def add_label(self):
        transactions = [transaction for transaction in reversed(self.one_round)]

        states = [transaction['state'] for transaction in transactions]
        next_states = [transaction['next_state'] for transaction in transactions]

        state_values = self.driver.get_state_values(states)
        next_state_values = self.driver.get_state_values(next_states)

        reward = crashing_reward
        for t, state_value, next_state_value in zip(transactions, state_values, next_state_values):
            reward += t['reward']
            reward *= reward_decay
            self.rewards.append(reward)
            t['advantage'] = reward + GAMMA * next_state_value - state_value
    
    def update_memory(self):
        if len(self.one_round) == 0:
            return
        self.add_label()
        for transaction in self.one_round:
            self.memory.insert(transaction)

    def resume_game(self):
        pyautogui.press('esc')
        # my mouse ...
        for _ in range(10):
            pyautogui.click()

    def end_and_restart_new_game(self):
        self.update_memory()
        self.collect_game_metrics()
        self.plot_metrics()
        self.init_game()
        self.take_random_action_prob *= MU
        
        #print('crashed, restarting...')
        self.resume_game()

    def on_dashboard(self, src_img, last_steering_angle, speed, throttle, info):
        track_img = ImageProcessor.preprocess(src_img)

        self.src_img = src_img
        self.track_img = track_img

        # get angle to calculate reward
        self.update_speeds(speed)
        if self.is_crash(track_img):
            self.end_and_restart_new_game()
            return

        try:
            target_angle = get_car_road_angle(track_img)
            current_angle = ImageProcessor.find_steering_angle_by_color(track_img, last_steering_angle, debug = self.debug)
            #current_angle = ImageProcessor.find_steering_angle_by_line(track_img, last_steering_angle, debug = self.debug)
            #steering_angle = self._steering_pid.update(-current_angle)
            throttle, _, _, _ = self._throttle_pid.update(-current_angle, speed)  # current speed

            # update last transaction
            # because can't get next state right after executing action QQ
            #state = rgb2gray(track_img)/255.0
            state = rgb2gray(track_img)
            self.state = state
            if self.last_state is not None:
                transaction = Transaction([self.last_state, self.last_action, self.last_reward, state, None])
                self.one_round.append(transaction)

            # select action according to current state
            action = self.driver.get_action(state)

            if random() < self.take_random_action_prob:
                action = randint(a=0, b=len(ACTIONS)-1)
            else:
                logger.info('select' + str(action))
                a = Action(action)

            action = Action(action)
            steering_angle, _ = action.steering_angle, action.throttle
            logger.info(str(ImageProcessor.rad2deg(target_angle))+str(action)+str(self.take_random_action_prob)+' '+str(self.last_action))

            self.last_action = action.action_idx
            #self.last_reward = self.calculate_reward(steering_angle, speed)
            self.last_reward = self.calculate_reward(target_angle, speed)
            self.last_state = state
            self.last_elapsed = info['elapsed']

            self.speeds.append(speed)

            msg = '%.2f %.2f %.2f %.2f'%( target_angle*180/np.pi, self.last_reward, steering_angle*180/np.pi, throttle)
            img = put_meta(track_img, msg)
            ImageProcessor.save_image('images/normal', img, "track")
            '''
            img = draw_direction(track_img)
            ImageProcessor.save_image('images/normal', img, "track")
            ImageProcessor.show_image(src_img, "source")
            ImageProcessor.show_image(track_img, "track")
            logit("steering PID: %0.2f (%0.2f) => %0.2f (%0.2f)" % (current_angle, ImageProcessor.rad2deg(current_angle), steering_angle, ImageProcessor.rad2deg(steering_angle)))
            logit("throttle PID: %0.4f => %0.4f" % (speed, throttle))
            logit("info: %s" % repr(info))
            '''


            #plot([self.speeds, self.rewards])

            # execute action
            info['speed'] = speed
            #print(info)
            self._throttle_history.append(throttle)
            self._throttle_history = self._throttle_history[-self.MAX_THROTTLE_HISTORY:]

            self._car.control(steering_angle, sum(self._throttle_history) / self.MAX_THROTTLE_HISTORY)

            #self._car.control(steering_angle, throttle)

            # update policy & value network
            batch = self.memory.batch(batch_size)
            if batch.any():
                self.driver.update(batch)

            # reload policy & value network periodically
            if self.total_steps % reload_window == 0:
                self.driver.reload(str(info))

            self.round_id += 1
            self.total_steps += 1

        except:
            ImageProcessor.save_image('images/finish', src_img, "source")
            ImageProcessor.save_image('images/finish', track_img, "track")
            logger.error(traceback.format_exc())
            print(traceback.format_exc())
            self.end_and_restart_new_game()
            self._car.control(0, 0)
            return


class QDrive(ACNDrive):
    def __init__(self, *args, **kwargs):
        ACNDrive.__init__(self, *args, **kwargs)

        self.driver = QNetwork()

    def add_label(self):
        discount_reward(self.one_round)

        states = [transaction['state'] for transaction in self.one_round]
        states_qvalues = self.driver.get_Q_value(states, 'target')  # batch_size,action_size

        next_states = [transaction['next_state'] for transaction in self.one_round]
        next_states_best_qvalues = self.driver.get_best_Q_value(next_states)  # batch_size,1

        for transaction, state_qvalue, next_state_best_qvalue in zip(self.one_round, states_qvalues, next_states_best_qvalues):
            label = transaction['reward'] + GAMMA * next_state_best_qvalue

            labels = state_qvalue
            labels[transaction['action']] = label

            transaction['label'] = labels

# In[35]:

def setup():
    tf.reset_default_graph()

    sio = socketio.Server()
    record_folder = './records/'

    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)


    car = Car(control_function = send_control)
    #drive = AutoDrive(car, record_folder=record_folder)

    car_training_data_collector = TrainingDataCollector('./logs/tracks.log')
    #drive = ACNDrive(car, car_training_data_collector, record_folder=record_folder)
    drive = QDrive(car, car_training_data_collector, record_folder=record_folder)

    @sio.on('telemetry')
    def telemetry(sid, dashboard):
        if dashboard:
            car.on_dashboard(dashboard)
        else:
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        car.control(0, 0)

    return sio, car, drive


def run(sio, car, drive):
    app = socketio.Middleware(sio, Flask(__name__))
    print('starting ...')
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app, log_output=False)



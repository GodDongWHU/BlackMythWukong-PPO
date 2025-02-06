import cv2
import gym
import os
import mss
import time
import numpy as np
from gym import spaces
import pydirectinput
import pytesseract                              # Pytesseract is not just a simple pip install.
from WukongReward import WukongReward
from walkToBoss import walkToBoss

import re
from PIL import Image


N_CHANNELS = 3                                  #Image format
IMG_WIDTH = 1680                                #Game capture resolution
IMG_HEIGHT = 1050                             
MODEL_WIDTH = int(800 / 2)                      #Ai vision resolution
MODEL_HEIGHT = int(450 / 2)


'''Ai action list'''
DISCRETE_ACTIONS = {'release_wasd': 'release_wasd',
                    'w': 'run_forwards',                
                    's': 'run_backwards',
                    'a': 'run_left',
                    'd': 'run_right',
                    'w+shift': 'dodge_forwards',
                    's+shift': 'dodge_backwards',
                    'a+shift': 'dodge_left',
                    'd+shift': 'dodge_right',
                    'j': '1 light_attack',
                    'jjj': '3 light_attack',
                    'jjjjj': '5 light_attack',
                    'k': 'heavy_attack',
                    'j+k': 'block',
                    'r': 'heal'}


NUMBER_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)
NUM_ACTION_HISTORY = 10                         #Number of actions the agent can remember


class WukongEnv(gym.Env):


    def __init__(self, config):
        '''Setting up the environment'''
        super(WukongEnv, self).__init__()
        logdir = config.get("logdir", ".")  # 默认使用当前目录
        log_file_path = os.path.join(logdir, "Wukongenv.log")
        self.log_file = open(log_file_path, "a")
        '''Setting up the gym spaces'''
        self.action_space = spaces.Discrete(NUMBER_DISCRETE_ACTIONS)                                                            #Discrete action space with NUM_ACTION_HISTORY actions to choose from
        spaces_dict = {                                                                                                         #Observation space (img, prev_actions, state)
            'img': spaces.Box(low=0, high=255, shape=(MODEL_HEIGHT, MODEL_WIDTH, N_CHANNELS), dtype=np.uint8),                      #Image of the game
            'prev_actions': spaces.Box(low=0, high=1, shape=(NUM_ACTION_HISTORY, NUMBER_DISCRETE_ACTIONS, 1), dtype=np.uint8),      #Last 10 actions as one hot encoded array
            'state': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),                                                       #Stamina and health of the player in percent
        }
        self.observation_space = gym.spaces.Dict(spaces_dict)
    

        '''Setting up the variables'''''
        pytesseract.pytesseract.tesseract_cmd = config["PYTESSERACT_PATH"]          #Setting the path to pytesseract.exe            
        self.sct = mss.mss()                                                        #Initializing CV2 and MSS (used to take screenshots)
        self.reward = 0                                                             #Reward of the previous step
        self.rewardGen = WukongReward(config)                                        #Setting up the reward generator class
        self.death = False                                                          #If the agent died
        self.boss_death = False
        self.t_start = time.time()                                                  #Time when the training started
        self.done = False                                                           #If the game is done
        self.step_iteration = 0                                                     #Current iteration (number of steps taken in this fight)
        self.first_step = True                                                      #If this is the first step
        self.max_reward = None
        self.last_action = 0
        self.reward_history = []                                                    #Array of the rewards to calculate the average reward of fight
        self.action_history = []                                                    #Array of the actions that the agent took.
        self.time_since_heal = time.time()                                          #Time since the last heal
        self.action_name = ''                                                       #Name of the action for logging
        self.MONITOR = config["MONITOR"]                                            #Monitor to use
        self.DEBUG_MODE = config["DEBUG_MODE"]                                      #If we are in debug mode
        self.GAME_MODE = config["GAME_MODE"]                                        #If we are in PVP or PVE mode
        self.DESIRED_FPS = config["DESIRED_FPS"]                                    #Desired FPS (not implemented yet)
        self.BOSS_HAS_SECOND_PHASE = config["BOSS_HAS_SECOND_PHASE"]                #If the boss has a second phase
        self.are_in_second_phase = False                                            #If we are in the second phase of the boss
        self.walk_to_boss = walkToBoss(config["BOSS"])  #Class to walk to the boss


    '''One hot encoding of the last 10 actions'''
    def oneHotPrevActions(self, actions):
        oneHot = np.zeros(shape=(NUM_ACTION_HISTORY, NUMBER_DISCRETE_ACTIONS, 1))
        for i in range(NUM_ACTION_HISTORY):
            if len(actions) >= (i + 1):
                oneHot[i][actions[-(i + 1)]][0] = 1
        #print(oneHot)
        return oneHot 


    '''Grabbing a screenshot of the game'''
    def grab_screen_shot(self):
        monitor = self.sct.monitors[self.MONITOR]
        sct_img = self.sct.grab(monitor)
        frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)
        frame = frame[120:IMG_HEIGHT+6, 12:IMG_WIDTH-6]    #cut the frame to the size of the game
        if self.DEBUG_MODE:
            self.render_frame(frame)
        return frame
    

    '''Rendering the frame for debugging'''
    def render_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                
        cv2.imshow('debug-render', image_rgb)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
    
    def detect_huashen(self, frame):
        x = 1291
        y = 851
        hsv_value = frame[y,x]
        huashen_available = False
        if hsv_value[2] >= 140:
            huashen_available = True
        return huashen_available
    
    def detect_fabao(self, frame):
        x = 1368
        y = 837
        hsv_value = frame[y,x]
        fabao_available = False
        if hsv_value[2] >= 140:
            fabao_available = True
        return fabao_available

    def detect_dingshen(self, frame):
        x = 1442
        y = 712
        hsv_value = frame[y,x]
        fabao_available = False
        if hsv_value[2] >= 140:
            fabao_available = True
        return fabao_available
        
    
    '''Defining the actions that the agent can take'''
    def take_action(self, action):
        #action = -1 #Uncomment this for emergency block all actions
        if action == 0:
            pydirectinput.keyUp('w')
            pydirectinput.keyUp('s')
            pydirectinput.keyUp('a')
            pydirectinput.keyUp('d')
            self.action_name = 'Stand still'
        elif action == 1:
            pydirectinput.keyUp('w')
            pydirectinput.keyUp('s')
            pydirectinput.keyDown('w')
            self.action_name = 'W'
        elif action == 2:
            pydirectinput.keyUp('w')
            pydirectinput.keyUp('s')
            pydirectinput.keyDown('s')
            self.action_name = 'S'
        elif action == 3:
            pydirectinput.keyUp('a')
            pydirectinput.keyUp('d')
            pydirectinput.keyDown('a')
            self.action_name = 'A'
        elif action == 4:
            pydirectinput.keyUp('a')
            pydirectinput.keyUp('d')
            pydirectinput.keyDown('d')
            self.action_name = 'D'
        elif action == 5:
            pydirectinput.keyDown('w')
            pydirectinput.press('shift')
            self.action_name = 'Roll W'
        elif action == 6:
            pydirectinput.keyDown('s')
            pydirectinput.press('shift')
            self.action_name = 'Roll S'
        elif action == 7:
            pydirectinput.keyDown('a')
            pydirectinput.press('shift')
            self.action_name = 'Roll A'
        elif action == 8:
            pydirectinput.keyDown('d')
            pydirectinput.press('shift')
            self.action_name = 'Roll D'
        elif action == 9:
            pydirectinput.press('j')
            self.action_name = '1 light attack'
        elif action == 10:
            pydirectinput.press('j')
            time.sleep(0.1)
            pydirectinput.press('j')
            time.sleep(0.3)
            pydirectinput.press('j')
            time.sleep(0.1)
            self.action_name = '3 light attack'
        elif action == 11:
            pydirectinput.press('j')
            time.sleep(0.1)
            pydirectinput.press('j')
            time.sleep(0.3)
            pydirectinput.press('j')
            time.sleep(0.3)
            pydirectinput.press('j')
            time.sleep(1.1)
            pydirectinput.press('j')
            time.sleep(0.4)
            self.action_name = '5 light attack'
        elif action == 12:
            pydirectinput.keyDown('k')
            time.sleep(0.2)
            pydirectinput.keyUp('k')
            time.sleep(0.5)
            self.action_name = 'Heavy attack'
        elif action == 13:
            pydirectinput.press('j')
            time.sleep(0.1)
            pydirectinput.press('k')
            time.sleep(0.5)
            pydirectinput.press('k')
            time.sleep(0.8)
            # pydirectinput.press('j')
            self.action_name = 'Penetrate'
        elif action == 14:
            pydirectinput.press('r')
            time.sleep(1.6)
            self.action_name = 'Heal'
        elif action == 99:
            pydirectinput.press('q')
            self.action_name = '回土地庙'
        elif action == 100:
            time.sleep(30)
            pydirectinput.press('q')
            self.action_name = '打赢回土地庙'

    
    '''Checking if we are in the boss second phase'''
    def check_for_second_phase(self):
        frame = self.grab_screen_shot()
        self.reward, self.death, self.boss_death= self.rewardGen.update(frame, self.first_step)

        if not self.boss_death:                 #if the boss is not dead when we check for the second phase, we are in the second phase
            self.are_in_second_phase = True
        else:                                   #if the boss is dead we can simply warp back to the bonfire
            self.are_in_second_phase = False


    '''Waiting for the loading screen to end'''
    def wait_for_loading_screen(self):
        in_loading_screen = False           #If we are in a loading screen right now
        have_been_in_loading_screen = False #If a loading screen was detected
        t_begin = time.time()
        t_check_frozen_start = time.time()  #Timer to check the length of the loading screen
        t_since_seen_next = None            #We detect the loading screen by reading the text "next" in the bottom left corner of the loading screen.
        while True: #We are forever taking a screenshot and checking if it is a loading screen.
            frame = self.grab_screen_shot()
            in_loading_screen = self.check_for_loading_screen(frame)
            if in_loading_screen:
                print("⌛ Loading Screen:", in_loading_screen) #Loading Screen: True
                have_been_in_loading_screen = True
                t_since_seen_next = time.time()
            else:   #If we dont see "next" on the screen we are not in the loading screen [anymore]
                if have_been_in_loading_screen:
                    print('⌛ After loading screen...')
                else:
                    t = time.time() - t_begin
                    if t > 60:
                        pydirectinput.press('q')
                        time.sleep(2)
                    print('⌛ Waiting for loading screen...')
                
            if have_been_in_loading_screen and (time.time() - t_since_seen_next) > 2.5:             #We have been in a loading screen and left it for more than 2.5 seconds
                print('⌛✔️ Left loading screen #1')
                break
            elif have_been_in_loading_screen and  ((time.time() - t_check_frozen_start) > 60):      #We have been in a loading screen for 60 seconds. We assume the game is frozen
                print('⌛❌ Did not leave loading screen #2 (Frozen)')
                #some sort of error handling here...
                #break
            elif not have_been_in_loading_screen and ((time.time() - t_check_frozen_start) > 20):   #We have not entered a loading screen for 25 seconds. (return to bonfire and walk to boss) #⚔️ in pvp we use this for waiting for matchmaking
                if self.GAME_MODE == "PVE":
                    if self.BOSS_HAS_SECOND_PHASE:
                        self.check_for_second_phase()
                        if self.are_in_second_phase:
                            print('⌛👹 Second phase found #3')
                            break
                        else:
                            print('⌛🔥 No loading screen found #3')
                            self.take_action(99)                #warp back to bonfire
                            t_check_frozen_start = time.time()  #reset the timer
                    else:
                        print('⌛👹 No second phase #3')
                        # self.take_action(99)                #warp back to bonfire
                        t_check_frozen_start = time.time()  #reset the timer
                                                            #try again by not breaking the loop (waiting for loading screen then walk to boss)
                else:
                    print('⌛❌ No loading screen found #3')
                    t_check_frozen_start = time.time()  #reset the timer
                                                        #continue waiting for loading screen (matchmaking)
        

    '''Checking if we are in a loading screen'''
    def check_for_loading_screen(self, frame):
        x = 10
        y = 670
        next_text_image = frame[y:y + 100, x:x + 100]

        # 使用 Pillow 显示
        # image = Image.fromarray(next_text_image, 'RGB')
        # image.show()

        gray = cv2.cvtColor(next_text_image, cv2.COLOR_BGR2GRAY)
        threshold = 3
        dark_pixels = np.sum(gray < threshold)
        total_pixels = gray.size
        dark_ratio = dark_pixels / total_pixels
        in_loading_screen = dark_ratio > 0.9
        return in_loading_screen

    def log_and_print(self, message):
        print(message)
        message = re.sub(r'[^\u0000-\uFFFF]', '', message)
        self.log_file.write(message + "\n")
        self.log_file.flush()



    '''Step function that is called by train.py'''
    def step(self, action):
        #📍 Lets look at what step does
        #📍 1. Collect the current observation 
        #📍 2. Collect the reward based on the observation (reward of previous step)
        #📍 3. Check if the game is done (player died, boss died, 10minute time limit reached)
        #📍 4. Take the next action (based on the decision of the agent)
        #📍 5. Ending the step
        #📍 6. Returning the observation, the reward, if we are done, and the info
        #📍 7*. train.py decides the next action and calls step again


        if self.first_step: print("🐾#1 first step")
        
        '''Grabbing variables'''
        t_start = time.time()    #Start time of this step
        frame = self.grab_screen_shot()                                         #📍 1. Collect the current observation
        
        #📍 2. Collect the reward based on the observation (reward of previous step)
        self.reward, curr_hp, self.death, self.boss_death, self.game_won = self.rewardGen.update(frame, self.first_step, self.last_action) 
        

        if self.DEBUG_MODE:
            print('🎁 Reward: ', self.reward)
            print('🎁 self.death: ', self.death)
            print('🎁 self.boss_death: ', self.boss_death)


        '''📍 3. Checking if the game is done'''
        if self.death:
            self.done = True
            print('🐾✔️ Step done (player death)') 
        else:
            if (time.time() - self.t_start) > 600:  #If the agent has been in control for more than 10 minutes we give up
                self.done = True
                self.take_action(99)                #warp back to bonfire
                print('🐾✔️ Step done (time limit)')
            elif self.boss_death:
                self.done = True  
                self.take_action(100)                #warp back to bonfire
                print('🐾✔️ Step done (boss death)')    

            

        '''📍 4. Taking the action'''    
        if not self.done:
            # 有cd的技能好了就放
            cd_skill = False
            if self.detect_huashen(frame):
                pydirectinput.press('f')
                time.sleep(2.3)
                cd_skill = True
            if self.detect_dingshen(frame):
                pydirectinput.press('1')
                time.sleep(0.1)
                self.take_action(5)
                time.sleep(0.4)
                self.take_action(11)
                cd_skill = True
            # if self.detect_fabao(frame):
            #     pydirectinput.press('t')
            #     time.sleep(0.2)
            if cd_skill:    # 放完技能后再计算一次reward
                self.reward, curr_hp, self.death, self.boss_death, self.game_won = self.rewardGen.update(frame, self.first_step, self.last_action) 

            if curr_hp < 0.45 and self.last_action != 14:
                action = 14
            self.take_action(action)
            self.last_action = action

        '''📍 5. Ending the steap'''

        '''Return values'''
        info = {}                                                       #Empty info for gym
        observation = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))    #We resize the frame so the agent doesn't have to deal with a 1920x1080 image (400x225)
        if self.DEBUG_MODE: self.render_frame(observation)              #🐜 If we are in debug mode we render the frame
        if self.max_reward is None:                                     #Max reward
            self.max_reward = self.reward
        elif self.max_reward < self.reward:
            self.max_reward = self.reward
        self.reward_history.append(self.reward)                         #Reward history
        spaces_dict = {                                                 #Combining the observations into one dictionary like gym wants it
            'img': observation,
            'prev_actions': self.oneHotPrevActions(self.action_history),
            'state': np.asarray([self.rewardGen.curr_hp, self.rewardGen.curr_stam])
        }


        '''Other variables that need to be updated'''
        self.first_step = False
        self.step_iteration += 1
        self.action_history.append(int(action))                         #Appending the action to the action history


        '''FPS LIMITER'''
        t_end = time.time()                                             
        desired_fps = (1 / self.DESIRED_FPS)                            #My CPU (i9-13900k) can run the training at about 2.4SPS (steps per secons)
        time_to_sleep = desired_fps - (t_end - t_start)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        '''END FPS LIMITER'''


        current_fps = str(round(((1 / ((t_end - t_start) *10)) * 10), 1))     #Current SPS (steps per second)


        '''Console output of the step'''
        if not self.done: #Losts of python string formatting to make the console output look nice
            self.reward = round(self.reward, 0)
            reward_with_spaces = str(self.reward)
            for i in range(5 - len(reward_with_spaces)):
                reward_with_spaces = ' ' + reward_with_spaces
            max_reward_with_spaces = str(self.max_reward)
            for i in range(5 - len(max_reward_with_spaces)):
                max_reward_with_spaces = ' ' + max_reward_with_spaces
            for i in range(18 - len(str(self.action_name))):
                self.action_name = ' ' + self.action_name
            for i in range(5 - len(current_fps)):
                current_fps = ' ' + current_fps
            if not self.done:
                message = (f'👣 Iteration: {self.step_iteration} | FPS: {current_fps} | '
                           f'Reward: {reward_with_spaces} | Max Reward: {max_reward_with_spaces} | '
                           f'Action: {self.action_name}')
                self.log_and_print(message)
            else:
                message = f'👣✔️ Reward: {self.reward} | Max Reward: {self.max_reward}'
                self.log_and_print(message)
        else:           #If the game is done (Logging Reward for dying or winning)
            print('👣✔️ Reward: ' + str(self.reward) + '| Max Reward: ' + str(self.max_reward))


        #📍 6. Returning the observation, the reward, if we are done, and the info
        return spaces_dict, self.reward, self.done, info
    

    '''Reset function that is called if the game is done'''
    def reset(self):
        #📍 1. Clear any held down keys
        #📍 2. Print the average reward for the last run
        #📍 3. Wait for loading screen                      #⚔️3-4 PvP: wait for loading screen - matchmaking - wait for loading screen - lock on
        #📍 4. Walking back to the boss
        #📍 5. Reset all variables
        #📍 6. Create the first observation for the first step and return it


        print('🔄 Reset called...')


        '''📍 1.Clear any held down keys'''
        self.take_action(0)
        self.last_action = 0
        print('🔄🔪 Unholding keys...')
        time.sleep(1.5)

        '''📍 2. Print the average reward for the last run'''
        if len(self.reward_history) > 0:
            total_r = 0
            for r in self.reward_history:
                total_r += r
            avg_r = total_r / len(self.reward_history)                              
            print('🔄🎁 Average reward for last run:', avg_r) 


        '''📍 3. Checking for loading screen / waiting some time for sucessful reset'''
        if self.GAME_MODE == "PVE": self.wait_for_loading_screen()
        else:   #⚔️
            #wait for loading screen (after the duel) - matchmaking - wait for loading screen (into the duel) - lock on
            if not self.first_reset:            #handle the first reset differently (we want to start with the matchmaking, not with losing a duel)
                self.wait_for_loading_screen() 
                self.matchmaking.perform()
            self.first_reset = False
            self.wait_for_loading_screen()

            

        '''📍 4. Walking to the boss'''         #⚔️we already did this in 📍 3. for PVP
        if self.GAME_MODE == "PVE":
            if self.BOSS_HAS_SECOND_PHASE:
                if self.are_in_second_phase:
                    print("🔄👹 already in arena")
                else:
                    print("🔄👹 walking to boss")
                    self.walk_to_boss.perform()
            else:                
                print("🔄👹 walking to boss")
                self.walk_to_boss.perform()          #This is hard coded in walkToBoss.py

        if self.death:                           #Death counter in txt file
            f = open("deathCounter.txt", "r")
            deathCounter = int(f.read())
            f.close()
            deathCounter += 1
            f = open("deathCounter.txt", "w")
            f.write(str(deathCounter))
            f.close()


        '''📍 5. Reset all variables'''
        self.step_iteration = 0
        self.reward_history = [] 
        self.done = False
        self.first_step = True
        self.max_reward = None
        self.rewardGen.prev_hp = 1
        self.rewardGen.curr_hp = 1
        self.rewardGen.time_since_dmg_taken = time.time()
        self.rewardGen.curr_boss_hp = 1
        self.rewardGen.prev_boss_hp = 1
        self.action_history = []
        self.t_start = time.time()


        '''📍 6. Return the first observation'''
        frame = self.grab_screen_shot()
        observation = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))    #Reset also returns the first observation for the agent
        spaces_dict = { 
            'img': observation,                                         #The image
            'prev_actions': self.oneHotPrevActions(self.action_history),#The last 10 actions (empty)
            'state': np.asarray([1.0, 0])                             #Full hp and full stamina
        }
        
        print('🔄✔️ Reset done.')
        return spaces_dict                                              #return the new observation




    '''Closing the environment (not used)'''

    def close(self):
        self.cap.release()
        if hasattr(self, 'log_file'):
            self.log_file.close()


if __name__ == "__main__":
    # test
    time.sleep(3)
    env_config = {
        "PYTESSERACT_PATH": r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',    # Set the path to PyTesseract
        "MONITOR": 1,           #Set the monitor to use (1,2,3)
        "DEBUG_MODE": False,    #Renders the AI vision (pretty scuffed)
        "GAME_MODE": "PVE",     #PVP or PVEx
        "BOSS": 1,              #1-6 for PVE (look at walkToBoss.py for boss names) | Is ignored for GAME_MODE PVP
        "BOSS_HAS_SECOND_PHASE": False,  #Set to True if the boss has a second phase (only for PVE)
        "PLAYER_HP": 150,      #Set the player hp (used for hp bar detection)
        "PLAYER_STAMINA": 10,  #Set the player stamina (used for stamina bar detection)
        "DESIRED_FPS": 20       #Set the desired fps (used for actions per second) (24 = 2.4 actions per second) #not implemented yet       #My CPU (i9-13900k) can run the training at about 2.4SPS (steps per secons)
    }
    env = WukongEnv(env_config)
    frame = env.grab_screen_shot()
    env.detect_dingshen(frame)
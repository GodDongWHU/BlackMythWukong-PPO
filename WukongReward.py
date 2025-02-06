import cv2
import numpy as np
import time
import pytesseract 
import pydirectinput

from PIL import Image

from matplotlib import pyplot as plt
class WukongReward:
    '''Reward Class'''


    '''Constructor'''
    def __init__(self, config):
        pytesseract.pytesseract.tesseract_cmd = config["PYTESSERACT_PATH"]        #Setting the path to pytesseract.exe
        self.GAME_MODE = config["GAME_MODE"]
        self.DEBUG_MODE = config["DEBUG_MODE"]
        self.max_hp = config["PLAYER_HP"]                             #This is the hp value of your character. We need this to capture the right length of the hp bar.
        self.prev_hp = 1.0     
        self.curr_hp = 1.0
        self.time_since_dmg_taken = time.time()
        self.death = False
        self.max_stam = config["PLAYER_STAMINA"]    
        self.prev_stam = 0                 
        self.curr_stam = 0
        self.prev_boss_hp = 1.0
        self.curr_boss_hp = 1.0
        self.time_since_boss_dmg = time.time() 
        self.time_since_pvp_damaged = time.time()
        self.time_alive = time.time()
        self.boss_death = False    
        self.game_won = False    
        self.image_detection_tolerance = 0.02               #The image detection of the hp bar is not perfect. So we ignore changes smaller than this value. (0.02 = 2%)
        

    '''Detecting the current player hp'''
    def get_current_hp(self, frame):
        x = 173
        y = 848
        hp_image=frame[y:y+14, x:x+243]

        # 两种方法一起检测current_hp
        # 第一种方法：检测canny_edges的宽度
        # height, width, channels = hp_image.shape
        # black_border = np.zeros((height, 1, channels), dtype=hp_image.dtype)
        # image_with_border = np.hstack((hp_image, black_border))   #右边拼接一个黑色边界
        # image_gray = cv2.cvtColor(image_with_border, cv2.COLOR_BGR2GRAY)
        # blurred_img = cv2.GaussianBlur(image_gray, (3, 3), 0)
        # canny_edges = cv2.Canny(blurred_img, 30, 100)
        # value = canny_edges.argmax(axis=-1)
        # curr_hp1=np.max(value) / width
        # curr_hp1 += 0.02
        # if curr_hp1 >= 0.96:     #If the hp is above 96% we set it to 100% (also color noise fix)
        #     curr_hp1 = 1.0

        # 第二种方法：一共有多少个白色像素a
        lower_red = np.array([140,80,80])                                         #Filter the image for the correct shade of red
        lower_green = np.array([75,140,80])
        upper = np.array([230,230,230])                                         #Also Filter
        mask1 = cv2.inRange(hp_image, lower_red, upper)
        mask2 = cv2.inRange(hp_image, lower_green, upper)
        # if self.DEBUG_MODE: self.render_frame(mask)
        matches = max(len(np.argwhere(mask1==255)), len(np.argwhere(mask2==255)))                                        #Number for all the white pixels in the mask
        curr_hp = matches / (hp_image.shape[1] * hp_image.shape[0])        #Calculating percent of white pixels in the mask (current hp in percent)
        curr_hp += 0.02         #Adding +2% of hp for color noise
        if curr_hp >= 0.96:     #If the hp is above 96% we set it to 100% (also color noise fix)
            curr_hp = 1.0
        # if self.DEBUG_MODE: 
        # curr_hp = max(curr_hp1, curr_hp2)
        print('💊 Health: ', curr_hp)

        # if curr_hp < 0.03:
        #     # 使用 Pillow 显示
        #     image = Image.fromarray(hp_image)
        #     image.show()

        return curr_hp


    '''Detecting the current player stamina'''
    def get_current_stamina(self, frame):
        # 棍势条的值
        x = 1473
        y = 821
        st_image = frame[y:y + 79, x:x + 33]

        lower_white = np.array([220, 220, 200])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(st_image, lower_white, upper_white)
        rows = np.any(mask == 255, axis=1)  # 找到每行是否有白色像素
        white_indices = np.where(rows)[0]  # 获取所有白色像素行的索引
        if len(white_indices) > 0:
            max_white_height = white_indices[-1] - white_indices[0] + 1  # 最大高度
        else:
            max_white_height = 0  # 如果没有白色部分，高度为 0
        height_ratio = max_white_height / mask.shape[0]

        # 棍势点的数量
        x = 1550
        y = 855
        st_image = frame[y:y + 52, x:x + 46]

        lower_white = np.array([250, 245, 140])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(st_image, lower_white, upper_white)

        # 使用 Pillow 显示
        # image = Image.fromarray(mask)
        # image.show()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_regions = min(sum(1 for contour in contours if cv2.contourArea(contour) >= 15), 4)

        self.curr_stam = max(height_ratio + valid_regions, 4)
        return self.curr_stam
    

    '''Detecting the current boss hp'''
    def get_boss_hp(self, frame):
        x = 587
        y = 789
        boss_hp_image = frame[y:y + 10,x:x + 507]

        lower = np.array([170,100,100])                                         #Filter the image for the correct shade of red
        upper = np.array([230,230,230])                                         #Also Filter
        mask = cv2.inRange(boss_hp_image, lower, upper)

        matches = np.argwhere(mask==255)                                        #Number for all the white pixels in the mask
        boss_hp = len(matches) / (boss_hp_image.shape[1] * boss_hp_image.shape[0])        #Calculating percent of white pixels in the mask (current hp in percent)

        boss_hp += 0.02         #Adding +2% of hp for color noise
        
        return boss_hp
    

    '''Detecting if the boss is damaged in PvE'''           #🚧 This is not implemented yet!!
    def detect_boss_damaged(self, frame):
        cut_frame = frame[863:876, 462:1462]
        
        lower = np.array([23,210,0])                                            #This filter really inst perfect but its good enough bcause stamina is not that important
        upper = np.array([25,255,255])                                           #Also Filter
        hsv = cv2.cvtColor(cut_frame, cv2.COLOR_RGB2HSV)                       #Apply the filter
        mask = cv2.inRange(hsv, lower, upper)                                   #Also apply
        matches = np.argwhere(mask==255)                                        #Number for all the white pixels in the mask


        if len(matches) > 30:                                                   #if there are more than 30 white pixels in the mask, return true
            return True
        else:
            return False

    

    '''Detecting if the enemy is damaged in PvP'''          #🚧 This is not implemented yet!!
    def detect_pvp_damaged(self, frame):
        cut_frame = frame[150:400, 350:1700]
        
        lower = np.array([24,210,0])                                            #This filter really inst perfect but its good enough bcause stamina is not that important
        upper = np.array([25,255,255])                                           #Also Filter
        hsv = cv2.cvtColor(cut_frame, cv2.COLOR_RGB2HSV)                       #Apply the filter
        mask = cv2.inRange(hsv, lower, upper)                                   #Also apply
        matches = np.argwhere(mask==255)                                        #Number for all the white pixels in the mask
        if len(matches) > 30:                                                   #if there are more than 30 white pixels in the mask, return true
            return True
        else:
            return False
    

    '''Detecting if the duel is won in PvP'''
    def detect_win(self, frame):
        cut_frame = frame[420:470, 600:1080]
        gray = cv2.cvtColor(cut_frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pytesseract_output = pytesseract.image_to_string(mask, lang='eng',config='--psm 6 --oem 3') #reading text from the image cutout
        game_won = "DEFEATED" in pytesseract_output           #Boolean if we see "DEFEATED" on the screen
        return game_won
    

    '''Debug function to render the frame'''
    def render_frame(self, frame):
        cv2.imshow('debug-render', frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

 
    '''Update function that gets called every step and returns the total reward and if the agent died or the boss died'''
    def update(self, frame, first_step, last_action):
        #📍 1 Getting current values
        #📍 2 Hp Rewards
        #📍 3 Boss Rewards
        #📍 4 PvP Rewards
        #📍 5 Total Reward / Return


        '''📍1 Getting/Setting current values'''
        self.curr_hp = self.get_current_hp(frame)                   
        self.curr_stam = self.get_current_stamina(frame)            
        self.curr_boss_hp = self.get_boss_hp(frame)
        if first_step:
            self.time_since_dmg_taken = time.time() - 10 #Setting the time_since_dmg_taken to 10 seconds ago so we dont get a negative reward at the start of the game     
            self.prev_hp = self.curr_hp
            self.prev_boss_hp = self.curr_boss_hp
        
        self.death = False
        if self.curr_hp <= 0.01 + self.image_detection_tolerance:   #If our hp is below 1% we are dead
            self.death = True
            self.curr_hp = 0.0

        self.boss_death = False
        if self.GAME_MODE == "PVE":                                 #Only if we are in PVE mode
            self.boss_death = self.detect_win(frame)             # If we defeat the boss
            self.game_won = self.boss_death

        
        '''📍2 Hp Rewards'''
        hp_reward = 0
        if not self.death:                           
            if self.curr_hp - self.prev_hp > 0.3:        #Reward if we healed
                hp_reward = 120   
            elif self.curr_hp - self.prev_hp < 0.25 and last_action == 14:    # The replied health is too little
                hp_reward = -80
            elif self.curr_hp < self.prev_hp - self.image_detection_tolerance:      #Negative reward if we took damage
                hp_reward = min((self.curr_hp - self.prev_hp + self.image_detection_tolerance) * 140, 0)
                self.time_since_dmg_taken = time.time()
        else:
            hp_reward = -420                                                        #Large negative reward for dying

        time_since_taken_dmg_reward = 0                                  
        if time.time() - self.time_since_dmg_taken > 4:                             #Reward if we have not taken damage for 5 seconds (every step for as long as we don't take damage)
            time_since_taken_dmg_reward = 15

        self.prev_hp = self.curr_hp     #Update prev_hp to curr_hp


        '''📍3 Boss Rewards'''
        if self.GAME_MODE == "PVE":                                             #Only if we are in PVE mode
            boss_dmg_reward = 0
            if self.boss_death:                                                     #Large reward if the boss is dead
                boss_dmg_reward = 420
                pydirectinput.keyUp('w')
                pydirectinput.keyUp('s')
                pydirectinput.keyUp('a')
                pydirectinput.keyUp('d')
            else:
                if self.curr_boss_hp < self.prev_boss_hp - self.image_detection_tolerance/2:  #Reward if we damaged the boss (small tolerance because its a large bar)
                    if self.prev_boss_hp - self.curr_boss_hp < 0.2:
                        boss_dmg_reward = max((self.prev_boss_hp - self.curr_boss_hp + self.image_detection_tolerance/2) * 2000, 0)
                        self.time_since_boss_dmg = time.time()
                elif self.prev_stam - self.curr_stam > 0.8:         # 消耗了棍势点但没打到boss
                    boss_dmg_reward = -20
                if time.time() - self.time_since_boss_dmg > 3:                      #Negative reward if we have not damaged the boss for 5 seconds (every step for as long as we dont damage the boss)
                    boss_dmg_reward = -30                                                

            percent_through_fight_reward = 0
            if self.curr_boss_hp < 0.97:                                            #Increasing reward for every step we are alive depending on how low the boss hp is
                percent_through_fight_reward = self.curr_boss_hp * 15            

        if self.prev_boss_hp - self.curr_boss_hp < 0.2:         # boss一次不太可能掉血过多，可能是检测出了问题
            self.prev_boss_hp = self.curr_boss_hp     #Update prev_boss_hp to curr_boss_hp
        self.prev_stam = self.curr_stam

        '''📍4 PVP rewards'''
        pvp_reward = 0
        if self.GAME_MODE != "PVE":                                              #Only if we are in PVP mode
            enemy_damaged = self.detect_pvp_damaged(frame)                          #Detect if the enemy is damaged
            if enemy_damaged:                                                       #Reward if the enemy is damaged
                pvp_reward = 69                         
                self.time_since_pvp_damaged = time.time()
            else:
                if time.time() - self.time_since_pvp_damaged > 4:                   #Negative reward if we have not damaged the enemy for 5 seconds (every step for as long as we dont damage the enemy)
                    pvp_reward = -25
                    #print("🔫 Duelist not damaged for 4s")
                else:
                    pvp_reward = 0

            #staying alive reward
            '''                     #a time alive reward could cause problems because the agent will still get rewarded even if performing bad when the time alive reward is higher than the other punishments
            time_alive_reward = 0
            if time.time() - self.time_alive > 5:                                   #Reward if we have been alive for 5 seconds (we give an increasinig reward for every second we are alive)
                time_alive_reward = time.time() - self.time_alive - 5
                print("🕒 Time alive reward: ", time_alive_reward)
                pvp_reward += time_alive_reward
            '''

            #winning
            self.game_won = self.detect_win(frame)    #not implemented yet
            if self.game_won:
                pvp_reward = 420
                pydirectinput.keyUp('w')
                pydirectinput.keyUp('s')
                pydirectinput.keyUp('a')
                pydirectinput.keyUp('d')


        '''📍5 Total Reward / Return'''
        if self.GAME_MODE == "PVE":                                                 #Only if we are in PVE mode
            total_reward = hp_reward + boss_dmg_reward + time_since_taken_dmg_reward + percent_through_fight_reward
        else:
            total_reward = hp_reward + time_since_taken_dmg_reward + pvp_reward
        
        total_reward = round(total_reward, 3)

        return total_reward, self.curr_hp, self.death, self.boss_death, self.game_won

    


'''Testing code'''
if __name__ == "__main__":
    env_config = {
        "PYTESSERACT_PATH": r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',    # Set the path to PyTesseract
        "MONITOR": 1,           #Set the monitor to use (1,2,3)
        "DEBUG_MODE": False,    #Renders the AI vision (pretty scuffed)
        "GAME_MODE": "PVE",     #PVP or PVE
        "BOSS": 8,              #1-6 for PVE (look at walkToBoss.py for boss names) | Is ignored for GAME_MODE PVP
        "BOSS_HAS_SECOND_PHASE": True,  #Set to True if the boss has a second phase (only for PVE)
        "PLAYER_HP": 1679,      #Set the player hp (used for hp bar detection)
        "PLAYER_STAMINA": 121,  #Set the player stamina (used for stamina bar detection)
        "DESIRED_FPS": 24       #Set the desired fps (used for actions per second) (24 = 2.4 actions per second) #not implemented yet       #My CPU (i9-13900k) can run the training at about 2.4SPS (steps per secons)
    }
    reward = WukongReward(env_config)

    IMG_WIDTH = 1680                                #Game capture resolution
    IMG_HEIGHT = 1050

    import mss
    sct = mss.mss()
    monitor = sct.monitors[1]

    sct_img = sct.grab(monitor)
    frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)
    frame = frame[120:IMG_HEIGHT+6, 12:IMG_WIDTH-6]    #cut the frame to the size of the game

    
    print(reward.get_current_stamina(frame))
    # print(reward.get_current_hp(frame))

    # reward.update(frame, True)

    # time.sleep(1)
    # reward.update(frame, False)

        
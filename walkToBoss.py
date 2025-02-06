import pydirectinput
import time


class walkToBoss:
        '''Walk to boss class - hard coded paths from the bonfire to the boss'''

        '''Constructor'''
        def __init__(self, BOSS):
                self.BOSS = BOSS        #Boss number | 99/100 reserved for PVP


        '''Walk to boss function'''
        def perform(self):
                '''PVE'''
                if self.BOSS == 1:
                        self.boss1()
                # elif self.BOSS == 2:
                #         self.boss2()
                # elif self.BOSS == 3:
                #         self.boss3()
                # elif self.BOSS == 4:
                #         self.boss4()
                # elif self.BOSS == 5:
                #         self.boss5()
                # elif self.BOSS == 6:
                #         self.boss6()
                # elif self.BOSS == 7:
                #         self.boss7()
                # elif self.BOSS == 8:
                #         self.boss8()
                # elif self.BOSS == 9:
                #         self.boss9()
                # elif self.BOSS == 10:
                #         self.boss10()
                # elif self.BOSS == 11:






        '''Hu xianfeng'''
        def boss1(self):
                time.sleep(4)
                pydirectinput.press('e')
                time.sleep(4)
                pydirectinput.press('esc')
                time.sleep(1.5)
                pydirectinput.press('z')

                pydirectinput.keyDown('s')
                time.sleep(0.1)
                pydirectinput.keyDown('ctrl')
                time.sleep(0.8)
                pydirectinput.keyDown('d')
                time.sleep(0.6)
                pydirectinput.keyUp('s')
                time.sleep(0.2)
                pydirectinput.keyDown('w')
                time.sleep(3)
                pydirectinput.keyUp('d')
                time.sleep(3.1)
                pydirectinput.keyDown('d')
                time.sleep(1.5)
                pydirectinput.keyUp('d')
                pydirectinput.keyDown('a')
                time.sleep(1.15)
                pydirectinput.keyUp('a')
                time.sleep(2.05)

                pydirectinput.keyUp('w')
                pydirectinput.press('m')
                time.sleep(0.5)
                pydirectinput.keyDown('k')
                time.sleep(4)
                pydirectinput.keyDown('w')
                time.sleep(1.2)

                pydirectinput.press('c')
                pydirectinput.keyUp('w')
                pydirectinput.keyUp('k')
                pydirectinput.keyUp('ctrl')
                time.sleep(3.5)

                pydirectinput.press('f')
                time.sleep(2.3)

                pydirectinput.press('1')
                time.sleep(0.1)
                pydirectinput.keyDown('w')
                pydirectinput.press('shift')
                time.sleep(0.4)
                pydirectinput.press('j')
                time.sleep(0.1)
                pydirectinput.press('j')
                time.sleep(0.3)
                pydirectinput.press('j')
                time.sleep(0.3)
                pydirectinput.press('j')
                time.sleep(1.1)
                pydirectinput.press('j')
                pydirectinput.keyUp('w')
                # time.sleep(1.4)
                
                # pydirectinput.keyDown('w')
                # pydirectinput.press('t')
                # time.sleep(0.3)
                # pydirectinput.press('j')
                # pydirectinput.keyUp('w')

#Run the function to test it
def test():
#     print("👉👹 3")
#     time.sleep(1)
#     print("👉👹 2")
#     time.sleep(1)
#     print("👉👹 1")
#     time.sleep(1)
    walkToBoss(1).boss1()
if __name__ == "__main__":
    test()

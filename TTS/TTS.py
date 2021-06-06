import pyttsx3
import threading
import random
import time
english = {
    1: "car ahead.",
    2: "car in your left",
    3: "car in your right",
    4: "Truck ahead.",
    5: "Truck in your left",
    6: "Truck in your right",
    7: "Bus ahead.",
    8: "Bus in your left",
    9: "Bus in your right",
    10: "person ahead.",
    11: "person in your left",
    12: "person in your right",
    13: "Bicycle ahead.",
    14: "Bicycle in your left",
    15: "Bicycle in your right",
    16: "Bike ahead.",
    17: "Bike in your left",
    18: "Bike in your right",
}
nepali = {
    1: "car ahead.",
    2: "car in your left",
    3: "car in your right",
    4: "Truck ahead.",
    5: "Truck in your left",
    6: "Truck in your right",
    7: "Bus ahead.",
    8: "Bus in your left",
    9: "Bus in your right",
    10: "अगाडी व्यक्ति.",
    11: "तपाईंको बायाँ मा व्यक्ति",
    12: "तपाईंको दाहिने व्यक्ति",
    13: "Bicycle ahead.",
    14: "Bicycle in your left",
    15: "Bicycle in your right",
    16: "Bike ahead.",
    17: "Bike in your left",
    18: "Bike in your right",
}
class TTS (threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 100)    
        self.patience = 10
        self.command_list = []
        self.is_exit = False
        

    def addCommand(self,command):
        try:
            self.command_list.index(command)
        except:
            self.command_list.append(command)

    def exitTTS(self):
        self.is_exit = True

    def run(self):
        while True:
            if self.command_list:
                command = self.command_list.pop(0)
                if self.is_nepali:
                    self.engine.say(nepali.get(command,"This is default Command"))
                else:
                    self.engine.say(english.get(command,"This is default Command"))

                self.engine.runAndWait()
                self.patience = 60
            else:
                self.patience-=1
                time.sleep(1)
                if not self.patience:
                    break
            if self.is_exit:
                break
    
    def setLanguage(self,is_nepali=False):
        self.is_nepali = is_nepali

        if is_nepali:
            self.engine.setProperty('voice','nepali')
            print(self.engine.getProperty('voice'))
# thread = TTS()
# thread.start()

# for i in range(10):
    # idx=random.randint(1,14)
    # thread.addCommand(idx)
    

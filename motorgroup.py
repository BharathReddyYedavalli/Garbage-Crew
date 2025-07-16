import ev3dev2.motor
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, SpeedPercent
from ev3dev2.sound import Sound



class GarbageSortController:
    # motor controller for sorting garbage
   def __init__(self):
       try:
           # adjust output ports for motors
           self.motor_recyclable = LargeMotor(OUTPUT_A)
           self.motor_compost = LargeMotor(OUTPUT_B)
           self.motor_trash = LargeMotor(OUTPUT_C)
           self.sound = Sound()
           self.motors_available = True
           print("EV3 motors initialized successfully")
       except Exception as e:
           print(f"EV3 motors not available: {e}")
           self.motors_available = False

       # motor settings
       self.default_speed = SpeedPercent(50)
       self.default_degrees = 90

   def handle_Battery(self):
       """Handle battery classification"""
       if self.motors_available:
           self.motor_recyclable.on_for_degrees(self.default_speed, self.default_degrees)
       else:
           print("Motor action: Battery -> Recyclable bin")
       # self.sound.speak("Battery")

   def handle_Glass(self):
       """Handle glass classification"""
       if self.motors_available:
           self.motor_recyclable.on_for_degrees(self.default_speed, self.default_degrees)
       else:
           print("Motor action: Glass -> Recyclable bin")
       # self.sound.speak("Glass")


   def handle_Metal(self):
       """Handle trash classification"""
       self.motor_trash.on_for_degrees(self.default_speed, self.default_degrees)
       # self.sound.speak("Trash")

   def handle_Organic_Waste(self):
       """Handle organic waste classification"""
       self.motor_compost.on_for_degrees(self.default_speed, self.default_degrees)
       # self.sound.speak("Organic Waste")


   def handle_Paper(self):
       """Handle paper classification"""
       self.motor_recyclable.on_for_degrees(self.default_speed, self.default_degrees)
       # self.sound.speak("Paper")

   def handle_Plastic(self):
       """Handle plastic classification"""
       self.motor_recyclable.on_for_degrees(self.default_speed, self.default_degrees)
       # self.sound.speak("Plastic")


   def handle_Textiles(self):
       """Handle textiles classification"""
       self.motor_recyclable.on_for_degrees(self.default_speed, self.default_degrees)
       # self.sound.speak("Textiles")

   def handle_Trash(self):
       """Handle trash classification"""
       self.motor_trash.on_for_degrees(self.default_speed, self.default_degrees)
       # self.sound.speak("Trash")


   def handle_classification(self, label):
       """Main handler that routes classifications to appropriate motors"""
       print(f"Handling classification: {label}")  # debug output
       
       if label == "battery":
           self.handle_Battery()
       elif label == "glass":
           self.handle_Glass()
       elif label == "metal":
           self.handle_Metal()
       elif label == "organic_waste":
           self.handle_Organic_Waste()
       elif label == "paper_cardboard":
           self.handle_Paper()
       elif label == "plastic":
           self.handle_Plastic()
       elif label == "textiles":
           self.handle_Textiles()
       elif label == "trash":
           self.handle_Trash()
       else:
           print(f"Unknown classification: {label}")  # debug for unknown labels
   def set_motor_settings(self, speed_percent=50, degrees=90):
       self.default_speed = SpeedPercent(speed_percent)
       self.default_degrees = degrees

   def stop_all_motors(self):
       if self.motors_available:
           self.motor_recyclable.stop()
           self.motor_compost.stop()
           self.motor_trash.stop()
       else:
           print("Motor action: Stop all motors")


# global instance for easy import
garbage_sort_controller = GarbageSortController()
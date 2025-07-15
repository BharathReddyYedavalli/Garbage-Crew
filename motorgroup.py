import ev3dev2.motor
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, SpeedPercent
from ev3dev2.sound import Sound
from model import labels


class WasteMotorController:
    """Motor controller for waste classification system"""
    
    def __init__(self):
        # Initialize motors (adjust OUTPUT_X to your wiring)
        self.motor_recyclable = LargeMotor(OUTPUT_A)
        self.motor_compost = LargeMotor(OUTPUT_B)
        self.motor_trash = LargeMotor(OUTPUT_C)
        self.sound = Sound()
        
        # motor settings
        self.default_speed = SpeedPercent(50)
        self.default_degrees = 90
        
    def handle_Battery(self):
        """Handle battery classification"""
        print("Battery detected!")
        self.motor_recycle.on_for_degrees(self.default_speed, self.default_degrees)
        self.sound.speak("Battery")

    def handle_Glass(self):
        """Handle glass classification"""
        print("Glass detected!")
        self.motor_recycle.on_for_degrees(self.default_speed, self.default_degrees)
        self.sound.speak("Glass")

    def handle_Metal(self):
        """Handle trash classification"""
        print("Trash detected!")
        self.motor_trash.on_for_degrees(self.default_speed, self.default_degrees)
        self.sound.speak("Trash")
        
    def handle_Organic_Waste(self):
        """Handle organic waste classification"""
        print("Organic Waste detected!")
        self.motor_compost.on_for_degrees(self.default_speed, self.default_degrees)
        self.sound.speak("Organic Waste")

    def handle_Paper(self):
        """Handle paper classification"""
        print("Paper detected!")
        self.motor_recyclable.on_for_degrees(self.default_speed, self.default_degrees)
        self.sound.speak("Paper")
        
    def handle_Plastic(self):
        """Handle plastic classification"""
        print("Plastic detected!")
        self.motor_recyclable.on_for_degrees(self.default_speed, self.default_degrees)
        self.sound.speak("Plastic")

    def handle_Textiles(self):
        """Handle textiles classification"""
        print("Textiles detected!")
        self.motor_recyclable.on_for_degrees(self.default_speed, self.default_degrees)
        self.sound.speak("Textiles")
    
    def handle_Trash(self):
        """Handle trash classification"""
        print("Trash detected!")
        self.motor_trash.on_for_degrees(self.default_speed, self.default_degrees)
        self.sound.speak("Trash")

    def handle_classification(self, label):
        """Handle classification based on label"""
        if label == "Battery":
            self.handle_Battery()
        elif label == "Glass":
            self.handle_Glass()
        elif label == "Metal":
            self.handle_Metal()
        elif label == "Organic Waste":
            self.handle_Organic_Waste()  
        elif label == "Paper":
            self.handle_Paper()
        elif label == "Plastic":
            self.handle_Plastic()
        elif label == "Textiles":
            self.handle_Textiles()
        elif label == "Trash":
            self.handle_Trash()
        else:
            pass # ignore unknown classifications
    def set_motor_settings(self, speed_percent=50, degrees=90):
        """Set default motor speed and rotation degrees"""
        self.default_speed = SpeedPercent(speed_percent)
        self.default_degrees = degrees
        
    def stop_all_motors(self):
        """Emergency stop for all motors"""
        self.motor_recyclable.stop()
        self.motor_compost.stop()
        self.motor_trash.stop()

# global instance for easy import
waste_motor_controller = WasteMotorController()
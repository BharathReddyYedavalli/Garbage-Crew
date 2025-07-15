import ev3dev2.motor
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, SpeedPercent
from ev3dev2.sound import Sound

class WasteMotorController:
    """Motor controller for waste classification system"""
    
    def __init__(self):
        # Initialize motors (adjust OUTPUT_X to your wiring)
        self.motor_recycle = LargeMotor(OUTPUT_A)
        self.motor_compost = LargeMotor(OUTPUT_B)
        self.motor_trash = LargeMotor(OUTPUT_C)
        self.sound = Sound()
        
        # motor settings
        self.default_speed = SpeedPercent(50)
        self.default_degrees = 90
        
    def handle_recycle(self):
        """Handle recycle classification"""
        print("Recycle detected!")
        self.motor_recycle.on_for_degrees(self.default_speed, self.default_degrees)
        self.sound.speak("Recycle")
        
    def handle_compost(self):
        """Handle compost classification"""
        print("Compost detected!")
        self.motor_compost.on_for_degrees(self.default_speed, self.default_degrees)
        self.sound.speak("Compost")
        
    def handle_trash(self):
        """Handle trash classification"""
        print("Trash detected!")
        self.motor_trash.on_for_degrees(self.default_speed, self.default_degrees)
        self.sound.speak("Trash")
        
    def handle_other(self):
        """Handle other classification"""
        print("Other detected!")
        # You can assign a motor or just play a sound
        self.sound.speak("Other")
        
    def handle_classification(self, label):
        """Handle classification based on label"""
        if label == "Recycle":
            self.handle_recycle()
        elif label == "Compost":
            self.handle_compost()
        elif label == "Trashes":
            self.handle_trash()
        elif label == "Other":
            self.handle_other()
        else:
            print(f"Unknown classification: {label}")
            
    def set_motor_settings(self, speed_percent=50, degrees=90):
        """Set default motor speed and rotation degrees"""
        self.default_speed = SpeedPercent(speed_percent)
        self.default_degrees = degrees
        
    def stop_all_motors(self):
        """Emergency stop for all motors"""
        self.motor_recycle.stop()
        self.motor_compost.stop()
        self.motor_trash.stop()

# global instance for easy import
waste_motor_controller = WasteMotorController()
# Garbage-Crew

# Smart Waste Sorter — Trash, Compost & Recycling Classifier

This project is an **AI-powered automated waste sorting system** that classifies waste items into **Trash**, **Compost**, or **Recycling** using a fine-tuned image classification model.

The system runs on a **Raspberry Pi** connected to a **LEGO EV3** controller that operates a **trapdoor mechanism** to direct items into the correct bin automatically. This project demonstrates how **AI + robotics** can increase sorting efficiency and reduce contamination in waste streams and is applicable in many real world scenarios.

---

## Features

- Real-time image classification using a pretrained **ResNet/EfficientNet/ViT** model.
- Fine-tuned on various datasets for **Trash**, **Compost**, and **Recycling**.
- Raspberry Pi controls an EV3 motorized trapdoor for physical sorting.
- Low-cost, portable, and easy to expand.

---

## Hardware Requirements - Tested (What we used)

- Raspberry Pi 4
- Raspberry Pi Camera Module V2-8 Megapixel,1080p
- LEGO Mindstorms EV3 brick
- Compatible motor(s) for trapdoor mechanism
- Respective connecting cables via the components

---

## Software Requirements

- Python 3.x
- PyTorch
- torchvision or timm
- transformers (if using ViT)
- OpenCV (for camera capture)
- ev3dev2 Python library (for EV3 motor control)

---

## How It Works

1. The Pi camera captures an image of the waste item.
2. The image is fed to the AI model for classification.
3. The predicted class determines which trapdoor to open.
4. The EV3 motor moves the trapdoor to direct the waste item to the correct bin.

---

## Project Structure

............to be completed...........


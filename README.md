[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center"> ~ Coral Server with Object Detection ~ </h1>

The first time I put my hands on a Coral board I was pretty amazed by the demo that they make you run at the end of the set-up procedure.
It's a simple object detector running on the board, but that's not the point. You can check the predictions made by the network comfortably with your web browser! I recognized that was a pretty nice trick and that could be helpful for some of my works either serious or playful. Let's say expecially playful :))
So, I looked at their code and I had to admit that they'd done a very good job at over complicate their code. It works just fine, but you can't really build on that shit :)

So, here we are! I put few line of codes, using the Flask library, to build a very simple server, copycat of the original one. It simply shows, on your localhost, frames coming from a thread where the detections are made. For this repository, I wanted to put the same object detector of the demo. So, it runs a [Single-Shot-multiBox Detector](https://arxiv.org/abs/1512.02325) (SSD) either on your CPU or TPU if you have an [accellerator](https://coral.ai/products/accelerator/) attached to your host. However, you can simply change the code in the detector and make your own class to show on your browser other magics :crystal_ball:

<p align="center">
  <img width="600" height="338" src="media/how_it_work.gif">
</p>

# 1.0 Getting Started

Clone this repository

   ```bash
   git clone https://github.com/EscVM/Coral_Server_with_Object_Detection
   ```
   
## 1.1 Installations for the hosting device

Install on the hosting device the following libraries:

- [opencv-python](https://pypi.org/project/opencv-python/)
- [numpy](https://pypi.org/project/numpy/)
- [Flask](https://pypi.org/project/Flask/)
- [TensorFlow Lite Interpreter](https://www.tensorflow.org/lite/guide/python). If you're using the Coral USB Accelerator with the Raspberry download ARM32.  

# 2.0 Run the Server
Open your terminal in the project folder and launch:

   ```bash
   python3 server.py
   ```
or

  ```bash
   python3 server_login.py
   ```
if you want a very (un)protected version with a login page. (for the credentials think at your router)

**N.B.** as default it runs on the CPU. Use 

  ```bash
   python3 server_login.py --tpu 1
   ```
if you have a Coral Accelerator.

# Bonus: Random Projects
Here a little list of stupid projects I built using this repository:

- Person Detector @ Edge [here](media/how_it_work.gif)
- Person Counter during Quarantine (lol) [here](media/quarantine_counter.gif)
- Pose Detection @ Edge [here](media/pose_edge.jpg)

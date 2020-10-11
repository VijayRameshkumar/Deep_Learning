# Deep_Learning
<h1>Website Screenshots Dataset</h1>
<a target="_blank" href="https://public.roboflow.com/object-detection/website-screenshots"> Link for the dataset </a><br>

<h2>About This Dataset:</h2>

The Roboflow Website Screenshots dataset is a synthetically generated dataset composed of screenshots from over 1000 of the world's top websites. They have been automatically annotated to label the following classes:

    button - navigation links, tabs, etc.
    heading - text that was enclosed in <h1> to <h6> tags.
    link - inline, textual <a> tags.
    label - text labeling form fields.
    text - all other text.
    image - <img>, <svg>, or <video> tags, and icons.
    iframe - ads and 3rd party content.

<p><b>Example:</b></p>

This is an example image and annotation from the dataset:
<img src="sample.png"><br><br><br>

<p><b>Usage:</b></p>

Annotated screenshots are very useful in Robotic Process Automation. But they can be expensive to label. This dataset would cost over $4000 for humans to label on popular labeling services.

<h2><b>custom_object_detection: (YOLOv4)</b></h2>

I've trained this model with YOLOv4 darknet implemetation.

it yields promising results with <b>30FPS</b> in RTX 2080 Nvidia GPU

This is the SOTA real-time object detection method.

ASAP, I'll upload object-tracking tutorial for real-time.
<br><br>

<img src="yolov4.png"> <br><br>

<p><b> below, I've attached sample output results: </b></p> <br><br>

![](website_text_detect_yolov4.gif)<br><br>

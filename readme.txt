learn_color_names.py and
generate_color_names.py: two-part tool for using natural language generation to create
names for arbitrary paint colors.

Overview:
The system learns properties of paint color names from several thousand commercial paint color
examples, provided by Andrew Weber's colorbot project https://github.com/waweber/colorbot.
To run it yourself:
1. Set up environment
2. Run learn_paint_colors.py
3. run generate_paint_colors.py and test it.

Details:
1. Set up environment:
a. Download and install anaconda with python 3.
b. create environment 'color_names', activate
b. pip install -r requirements.txt

2. learn paint colors:
a. From your anaconda prompt:
>python learn_paint_colors.py
(If successful, you'll see output messages about the number of paint color examples
the system learned from, and what kinds of models it creates with them.)

3. generate paint colors:
a. From your anaconda prompt:
>python generate_paint_colors.py
(Choose a color in the color picker, press ok, and check out the results in your anaconda prompt.)

# Converting a Jupyter notebook for the blog
# Very crude. Please don't look here!
# This is *not* the way!

command = r'jupyter nbconvert {} --to markdown --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags=[\"remove_cell\"]'

import os
import sys
import shutil
from urllib.parse import quote
import glob


POST_PATH = r"_posts/"
IMAGES_PATH = r"assets/images/"
GITHUB_PATH = r"https://github.com/znswanderer/znswanderer.github.io/blob/main/_jupyter"


def pacify_dollars(txt):
    """In Jekyll somehow the single dollar signs $ for math do not work properly
    We change them to double dollar signs
    """
    print("pacify_dollars...")
    # first make sure the $$ signs are left untouched
    txt = txt.replace("$$", "DOLLAR_DOLLAR")
    # now change the single $ signs
    txt = txt.replace("$", "$$")
    # and back the original $$
    txt = txt.replace("DOLLAR_DOLLAR", "$$")
    return txt

def remove_displaystyle(txt):
    """ipython adds a \\displaystyle command to LaTeX output.
    Let's remove that!
    """
    print("remove_displaystyle...")
    txt = txt.replace(r"\displaystyle ", "")
    return txt

def change_image_links(image_base_path, txt):
    """Use the correct path
    """
    print("change_image_links...")
    out = []
    for line in txt.splitlines():
        if line.startswith("![png]"):
            picture_name = line.split("/")[-1][:-1]
            out.append('{: style="text-align:center"}')
            line = r'![png]({}_files/{})'.format(image_base_path, picture_name)
            out.append(line)
        else:
            out.append(line)

    return "\n".join(out)


STATE_DEFAULT = 'default'
STATE_IN_PYTHON = 'python'
STATE_WAITING_FOR_DOLLARS = "money"

def left_align_math_output(txt):
    """LaTeX output from python is left aligned in jupyter,
    but nbconvert exports it as $$ $$ environment with
    line breaks, so it will be centered.
    """
    print("left_align_math_output...")
    out = []
    saved_lines = []
    state = STATE_DEFAULT
    for n, line in enumerate(txt.splitlines()):
        if state == STATE_DEFAULT and line.startswith("```"):
            state = STATE_IN_PYTHON
            out.append(line)
            continue
        elif state == STATE_IN_PYTHON and line.startswith("```"):
            state = STATE_WAITING_FOR_DOLLARS
            saved_lines = []
            out.append(line)
            continue

        if state == STATE_IN_PYTHON:
            line = line.replace("$$", "$")   # in python we do not need double dollars
        
        if state != STATE_WAITING_FOR_DOLLARS:
            out.append(line)
            continue
        
        if line.startswith("$$"):
            # output after python was Math!
            state = STATE_DEFAULT
            saved_lines = []
            out.append(line + "  ")
        elif line.startswith("```"):
            state = STATE_IN_PYTHON
            out.extend(saved_lines)
            out.append(line)
        elif line.strip() != "":
            # The output after not not Math
            state = STATE_DEFAULT
            out.extend(saved_lines)
            out.append(line)
        else:
            saved_lines.append(line)

    return "\n".join(out)

STATE_PRE_HEADER = 'preheader'
STATE_AFTER_HEADER = 'afterheader'
STATE_IN_HEADER = 'inheader'
ALLOWED_LIQUIDS = ['{{site.url}}']

def make_raw(txt):
    '''Make the whole post after the header 'raw' for liquid
    '''
    print("make raw...")
    state = STATE_PRE_HEADER
    out = []
    for line in txt.splitlines():
        if line.startswith("---") and state == STATE_PRE_HEADER:
            state = STATE_IN_HEADER
            out.append(line)
            continue
        elif line.startswith("---") and state == STATE_IN_HEADER:
            state = STATE_AFTER_HEADER
            out.append(line)
            out.append("{% raw %}") # for Jekyll's liquid
            continue
        if line.startswith("<!--more-->"):
            out.append("{% endraw %}")
            out.append(line)
            out.append("{% raw %}") # for Jekyll's liquid
        elif any(x in line for x in ALLOWED_LIQUIDS):
            out.append("{% endraw %}")
            out.append(line)
            out.append("{% raw %}") # for Jekyll's liquid
        else:
            out.append(line)

    if state == STATE_AFTER_HEADER:
        out.append(" {% endraw %}")
    return "\n".join(out)


def fix_video(image_base_path, txt):
    print("fix_video", image_base_path)
    video_files = []
    out = []
    for line in txt.splitlines():
        if line.startswith('<video src='):
            line_parts = line.split('"')    
            v_file = line_parts[1]
            video_files.append(v_file)
            line_parts[1] = r'{}_files/{}'.format(image_base_path, v_file)
            out.append("{% endraw %}")
            out.append('"'.join(line_parts))
            out.append("{% raw %}")
        else:
            out.append(line)

    return "\n".join(out), video_files

def move_files(md_path):
    print("move_files...", md_path)
    dir_path, md_file_name = os.path.split(md_path)
    bookname, _ = os.path.splitext(md_file_name)
    print(md_file_name)
    print(os.path.join(dir_path, bookname + "_files"))
    try:
        shutil.move(md_path, os.path.join(POST_PATH, md_file_name))
        dst_img_path = os.path.join(IMAGES_PATH, bookname + "_files")
        shutil.rmtree(dst_img_path, ignore_errors=True)
        shutil.move(os.path.join(dir_path, bookname + "_files"), dst_img_path)
    except FileNotFoundError:
        print("No plots to move")


def copy_videos(md_path, video_files):
    print("copy_videos...", md_path)
    dir_path, md_file_name = os.path.split(md_path)
    bookname, _ = os.path.splitext(md_file_name)
    for v_file in video_files:
        print("copying", v_file)
        dst_img_path = os.path.join(IMAGES_PATH, bookname + "_files")
        print(dst_img_path)
        shutil.copy(os.path.join(dir_path, v_file), dst_img_path)

        #shutil.copy(src, dst)


def add_jupyter_link(md_name, txt):
    print("add_jupyter_link...")
    txt += "\n\n"
    txt += "*The original Jupyter notebook can be found [here](<{}/{}>).*".format(GITHUB_PATH, quote(md_name))
    return txt


def convert(notebook):
    print("Running nbconvert...")
    bookname, extension = os.path.splitext(notebook)
    print(command.format(notebook))
    os.system(command.format(notebook))

    print("\nAdjusting output...")
    md_path = bookname + ".md"
    with open(md_path) as f:
        txt = f.read()

    pure_bookname = os.path.split(bookname)[-1]
    image_base_path = r"{{site.url}}/assets/images/" + quote(pure_bookname)
    print(image_base_path)

    txt = pacify_dollars(txt)
    #txt = remove_displaystyle(txt)
    txt = left_align_math_output(txt)
    txt = change_image_links(image_base_path, txt)
    txt = add_jupyter_link(pure_bookname + ".ipynb", txt)
    txt = make_raw(txt)
    txt, video_files = fix_video(image_base_path, txt)

    with open(md_path, "w") as f:
        f.write(txt)

    move_files(md_path)
    copy_videos(md_path, video_files)


if __name__ == '__main__':
    pattern = sys.argv[1]
    for notebook in glob.glob(pattern):
        convert(notebook)




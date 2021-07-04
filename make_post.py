# Converting a Jupyter notebook for the blog
# Very crude. Please don't look here!
# This is *not* the way!

command = r'jupyter nbconvert {} --to markdown --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags=[\"remove_cell\"]'

import os
import sys
import shutil

POST_PATH = r"_posts/"
IMAGES_PATH = r"assets/images/"
GITHUB_PATH = r"https://github.com/znswanderer/znswanderer.github.io/blob/main/_jupyter/"


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
            line = r'![png]({}_files/{}){{: width="80%"}}'.format(image_base_path, picture_name)
            out.append(line)
        else:
            out.append(line)

    return "\n".join(out)

def move_files(md_path):
    print("move_files...")
    dir_path, md_file_name = os.path.split(md_path)
    bookname, _ = os.path.splitext(md_file_name)
    shutil.move(md_path, os.path.join(POST_PATH, md_file_name))
    dst_img_path = os.path.join(IMAGES_PATH, bookname + "_files")
    shutil.rmtree(dst_img_path, ignore_errors=True)
    shutil.move(os.path.join(dir_path, bookname + "_files"), dst_img_path)


def add_jupyter_link(md_name, txt):
    print("add_jupyter_link...")
    txt += "\n\n"
    txt += "*The original Jupyter notebook can be found [here](<%s/%s>).*" % (GITHUB_PATH, md_name)
    return txt


if __name__ == '__main__':
    print("Running nbconvert...")
    notebook = sys.argv[1]
    bookname, extension = os.path.splitext(notebook)
    print(command.format(notebook))
    os.system(command.format(notebook))

    print("\nAdjusting output...")
    md_path = bookname + ".md"
    with open(md_path) as f:
        txt = f.read()

    pure_bookname = os.path.split(bookname)[-1]
    image_base_path = r"{{site.url}}/assets/images/" + pure_bookname
    print(image_base_path)

    txt = pacify_dollars(txt)
    txt = remove_displaystyle(txt)
    txt = change_image_links(image_base_path, txt)
    txt = add_jupyter_link(pure_bookname + ".ipynb", txt)

    with open(md_path, "w") as f:
        f.write(txt)

    move_files(md_path)




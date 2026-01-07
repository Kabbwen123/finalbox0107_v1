import os

# place 取 1/10/100/1000... 分别表示 个/十/百/千 位
def number_at_place(n: int, place: int):
    return (abs(n) // place) % 10


def str_append(str1, str2, seperatestr = "\n"):
    return str2 if str1 == "" else str1 + seperatestr + str2


# 从文件路径获取不带扩展名的文件名
def get_filename_without_extension(file_path):
    # 获取带扩展名的文件名
    filename_with_ext = os.path.basename(file_path)
    # 分割文件名和扩展名
    filename, ext = os.path.splitext(filename_with_ext)
    #
    return filename
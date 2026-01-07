# 数据集共三种类型，每种类型下可再分小类
from enum import Enum


class DATASET:
    OK = 0
    NG = 1
    UNKNOW = -1

# 项目类型
class TYPE(Enum):
    EXCEPTION_CHECK = (0, "异常检查")

    def __init__(self, code: int, label: str):
        self.code = code  # 数值
        self.label = label  # 中文名

    @classmethod
    def from_code(cls, code: int) -> "TYPE":
        for item in cls:
            if item.code == code:
                return item
        raise ValueError(f"未知的检查类型 code: {code}")

    # # 从数值反查枚举
    # t2 = TYPE.from_code(1)
    # print(t2 is TYPE.SELF_CHECK)  # True
    # print(t2.label)                    # "自检检测"
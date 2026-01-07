from .event_bus import EventBus
from .command_bus import CommandBus

"""
use_case中根据.py文件来区分用例

eventbus信号不复用 commandbus能力复用

commandbus中来发出新的eventbus信号，While True的业务代码使用迭代器或者回调函数带到commandbus来发出
"""

test_for_pylint = 0


class MyClass:
    """Test CLass"""

    _protected = 0

    def __init__(self):
        self.test = 0

    def set_attribute(self):
        self.attr = 0
        self.test += 1


obj = MyClass()
obj._protected = 233

from eckity.before_after_publisher import BeforeAfterPublisher


class Initializer(BeforeAfterPublisher):
    def __init__(self):
        self.algorithm = None
        events = ["after_init", "get_executor"]
        super().__init__(events)

    def initialization(self, algorithm):
        self.algorithm = algorithm
        self.publish("after_init")

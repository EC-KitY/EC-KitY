BEFORE_OPERATOR_EVENT_NAME = 'before_operator'
AFTER_OPERATOR_EVENT_NAME = 'after_operator'


class BeforeAfterPublisher:
    def __init__(self, events=None, event_names=None):
        ext_events_names = event_names if event_names is not None else []
        if events is None:
            # Initialize events dictionary with event names as keys and subscribers as values
            ext_events_names.extend([BEFORE_OPERATOR_EVENT_NAME, AFTER_OPERATOR_EVENT_NAME])
            self.events = {event: {} for event in ext_events_names}
        else:
            # Assign an already existing dictionary, in case of deserialization/clone
            self.events = events
        self.customers_id = 0

    def _get_subscribers(self, event_name):
        return self.events[event_name]

    def register(self, event, callback=None):
        # TODO warning if register to a fake event
        if callback is None:
            callback = {lambda _: None}
        self._get_subscribers(event)[self.customers_id] = callback
        self.customers_id += 1
        return self.customers_id - 1

    def unregister(self, event, customers_id):
        # TODO warn if unregister to a fake event
        del self._get_subscribers(event)[customers_id]

    def publish(self, event_name):
        struct = self.event_name_to_data(event_name)
        for subscriber, callback in self._get_subscribers(event_name).items():
            callback(self, struct)

    def event_name_to_data(self, event_name):
        return {}  # TODO abs?

    def act_and_publish_before_after(self, act_func: callable):
        self.publish(BEFORE_OPERATOR_EVENT_NAME)
        return_val = act_func()
        self.publish(AFTER_OPERATOR_EVENT_NAME)
        return return_val

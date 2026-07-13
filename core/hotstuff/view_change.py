class ViewChangeHandler:
    def __init__(self, n: int, f: int):
        self.n = n
        self.f = f
        self.current_view = 1

    def should_change_view(
        self, leader_id: str, timeout: bool = False
    ) -> bool:
        return timeout

    def next_leader(self) -> str:
        idx = (self.current_view - 1) % self.n
        self.current_view += 1
        return f"n{idx}"
